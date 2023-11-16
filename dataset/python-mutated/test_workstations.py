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
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.rpc import status_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.workstations_v1beta.services.workstations import WorkstationsAsyncClient, WorkstationsClient, pagers, transports
from google.cloud.workstations_v1beta.types import workstations

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        print('Hello World!')
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        while True:
            i = 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert WorkstationsClient._get_default_mtls_endpoint(None) is None
    assert WorkstationsClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert WorkstationsClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert WorkstationsClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert WorkstationsClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert WorkstationsClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(WorkstationsClient, 'grpc'), (WorkstationsAsyncClient, 'grpc_asyncio'), (WorkstationsClient, 'rest')])
def test_workstations_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('workstations.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://workstations.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.WorkstationsGrpcTransport, 'grpc'), (transports.WorkstationsGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.WorkstationsRestTransport, 'rest')])
def test_workstations_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(WorkstationsClient, 'grpc'), (WorkstationsAsyncClient, 'grpc_asyncio'), (WorkstationsClient, 'rest')])
def test_workstations_client_from_service_account_file(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_file') as factory:
        factory.return_value = creds
        client = client_class.from_service_account_file('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        client = client_class.from_service_account_json('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('workstations.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://workstations.googleapis.com')

def test_workstations_client_get_transport_class():
    if False:
        for i in range(10):
            print('nop')
    transport = WorkstationsClient.get_transport_class()
    available_transports = [transports.WorkstationsGrpcTransport, transports.WorkstationsRestTransport]
    assert transport in available_transports
    transport = WorkstationsClient.get_transport_class('grpc')
    assert transport == transports.WorkstationsGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(WorkstationsClient, transports.WorkstationsGrpcTransport, 'grpc'), (WorkstationsAsyncClient, transports.WorkstationsGrpcAsyncIOTransport, 'grpc_asyncio'), (WorkstationsClient, transports.WorkstationsRestTransport, 'rest')])
@mock.patch.object(WorkstationsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WorkstationsClient))
@mock.patch.object(WorkstationsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WorkstationsAsyncClient))
def test_workstations_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(WorkstationsClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(WorkstationsClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(WorkstationsClient, transports.WorkstationsGrpcTransport, 'grpc', 'true'), (WorkstationsAsyncClient, transports.WorkstationsGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (WorkstationsClient, transports.WorkstationsGrpcTransport, 'grpc', 'false'), (WorkstationsAsyncClient, transports.WorkstationsGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (WorkstationsClient, transports.WorkstationsRestTransport, 'rest', 'true'), (WorkstationsClient, transports.WorkstationsRestTransport, 'rest', 'false')])
@mock.patch.object(WorkstationsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WorkstationsClient))
@mock.patch.object(WorkstationsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WorkstationsAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_workstations_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [WorkstationsClient, WorkstationsAsyncClient])
@mock.patch.object(WorkstationsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WorkstationsClient))
@mock.patch.object(WorkstationsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WorkstationsAsyncClient))
def test_workstations_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(WorkstationsClient, transports.WorkstationsGrpcTransport, 'grpc'), (WorkstationsAsyncClient, transports.WorkstationsGrpcAsyncIOTransport, 'grpc_asyncio'), (WorkstationsClient, transports.WorkstationsRestTransport, 'rest')])
def test_workstations_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(WorkstationsClient, transports.WorkstationsGrpcTransport, 'grpc', grpc_helpers), (WorkstationsAsyncClient, transports.WorkstationsGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (WorkstationsClient, transports.WorkstationsRestTransport, 'rest', None)])
def test_workstations_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_workstations_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.workstations_v1beta.services.workstations.transports.WorkstationsGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = WorkstationsClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(WorkstationsClient, transports.WorkstationsGrpcTransport, 'grpc', grpc_helpers), (WorkstationsAsyncClient, transports.WorkstationsGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_workstations_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('workstations.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='workstations.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [workstations.GetWorkstationClusterRequest, dict])
def test_get_workstation_cluster(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_workstation_cluster), '__call__') as call:
        call.return_value = workstations.WorkstationCluster(name='name_value', display_name='display_name_value', uid='uid_value', reconciling=True, etag='etag_value', network='network_value', subnetwork='subnetwork_value', control_plane_ip='control_plane_ip_value', degraded=True)
        response = client.get_workstation_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.GetWorkstationClusterRequest()
    assert isinstance(response, workstations.WorkstationCluster)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'
    assert response.network == 'network_value'
    assert response.subnetwork == 'subnetwork_value'
    assert response.control_plane_ip == 'control_plane_ip_value'
    assert response.degraded is True

def test_get_workstation_cluster_empty_call():
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_workstation_cluster), '__call__') as call:
        client.get_workstation_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.GetWorkstationClusterRequest()

@pytest.mark.asyncio
async def test_get_workstation_cluster_async(transport: str='grpc_asyncio', request_type=workstations.GetWorkstationClusterRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_workstation_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.WorkstationCluster(name='name_value', display_name='display_name_value', uid='uid_value', reconciling=True, etag='etag_value', network='network_value', subnetwork='subnetwork_value', control_plane_ip='control_plane_ip_value', degraded=True))
        response = await client.get_workstation_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.GetWorkstationClusterRequest()
    assert isinstance(response, workstations.WorkstationCluster)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'
    assert response.network == 'network_value'
    assert response.subnetwork == 'subnetwork_value'
    assert response.control_plane_ip == 'control_plane_ip_value'
    assert response.degraded is True

@pytest.mark.asyncio
async def test_get_workstation_cluster_async_from_dict():
    await test_get_workstation_cluster_async(request_type=dict)

def test_get_workstation_cluster_field_headers():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.GetWorkstationClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_workstation_cluster), '__call__') as call:
        call.return_value = workstations.WorkstationCluster()
        client.get_workstation_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_workstation_cluster_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.GetWorkstationClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_workstation_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.WorkstationCluster())
        await client.get_workstation_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_workstation_cluster_flattened():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_workstation_cluster), '__call__') as call:
        call.return_value = workstations.WorkstationCluster()
        client.get_workstation_cluster(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_workstation_cluster_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_workstation_cluster(workstations.GetWorkstationClusterRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_workstation_cluster_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_workstation_cluster), '__call__') as call:
        call.return_value = workstations.WorkstationCluster()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.WorkstationCluster())
        response = await client.get_workstation_cluster(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_workstation_cluster_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_workstation_cluster(workstations.GetWorkstationClusterRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [workstations.ListWorkstationClustersRequest, dict])
def test_list_workstation_clusters(request_type, transport: str='grpc'):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_workstation_clusters), '__call__') as call:
        call.return_value = workstations.ListWorkstationClustersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_workstation_clusters(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.ListWorkstationClustersRequest()
    assert isinstance(response, pagers.ListWorkstationClustersPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_workstation_clusters_empty_call():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_workstation_clusters), '__call__') as call:
        client.list_workstation_clusters()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.ListWorkstationClustersRequest()

@pytest.mark.asyncio
async def test_list_workstation_clusters_async(transport: str='grpc_asyncio', request_type=workstations.ListWorkstationClustersRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_workstation_clusters), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.ListWorkstationClustersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_workstation_clusters(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.ListWorkstationClustersRequest()
    assert isinstance(response, pagers.ListWorkstationClustersAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_workstation_clusters_async_from_dict():
    await test_list_workstation_clusters_async(request_type=dict)

def test_list_workstation_clusters_field_headers():
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.ListWorkstationClustersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_workstation_clusters), '__call__') as call:
        call.return_value = workstations.ListWorkstationClustersResponse()
        client.list_workstation_clusters(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_workstation_clusters_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.ListWorkstationClustersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_workstation_clusters), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.ListWorkstationClustersResponse())
        await client.list_workstation_clusters(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_workstation_clusters_flattened():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_workstation_clusters), '__call__') as call:
        call.return_value = workstations.ListWorkstationClustersResponse()
        client.list_workstation_clusters(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_workstation_clusters_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_workstation_clusters(workstations.ListWorkstationClustersRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_workstation_clusters_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_workstation_clusters), '__call__') as call:
        call.return_value = workstations.ListWorkstationClustersResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.ListWorkstationClustersResponse())
        response = await client.list_workstation_clusters(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_workstation_clusters_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_workstation_clusters(workstations.ListWorkstationClustersRequest(), parent='parent_value')

def test_list_workstation_clusters_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_workstation_clusters), '__call__') as call:
        call.side_effect = (workstations.ListWorkstationClustersResponse(workstation_clusters=[workstations.WorkstationCluster(), workstations.WorkstationCluster(), workstations.WorkstationCluster()], next_page_token='abc'), workstations.ListWorkstationClustersResponse(workstation_clusters=[], next_page_token='def'), workstations.ListWorkstationClustersResponse(workstation_clusters=[workstations.WorkstationCluster()], next_page_token='ghi'), workstations.ListWorkstationClustersResponse(workstation_clusters=[workstations.WorkstationCluster(), workstations.WorkstationCluster()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_workstation_clusters(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, workstations.WorkstationCluster) for i in results))

def test_list_workstation_clusters_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_workstation_clusters), '__call__') as call:
        call.side_effect = (workstations.ListWorkstationClustersResponse(workstation_clusters=[workstations.WorkstationCluster(), workstations.WorkstationCluster(), workstations.WorkstationCluster()], next_page_token='abc'), workstations.ListWorkstationClustersResponse(workstation_clusters=[], next_page_token='def'), workstations.ListWorkstationClustersResponse(workstation_clusters=[workstations.WorkstationCluster()], next_page_token='ghi'), workstations.ListWorkstationClustersResponse(workstation_clusters=[workstations.WorkstationCluster(), workstations.WorkstationCluster()]), RuntimeError)
        pages = list(client.list_workstation_clusters(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_workstation_clusters_async_pager():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_workstation_clusters), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (workstations.ListWorkstationClustersResponse(workstation_clusters=[workstations.WorkstationCluster(), workstations.WorkstationCluster(), workstations.WorkstationCluster()], next_page_token='abc'), workstations.ListWorkstationClustersResponse(workstation_clusters=[], next_page_token='def'), workstations.ListWorkstationClustersResponse(workstation_clusters=[workstations.WorkstationCluster()], next_page_token='ghi'), workstations.ListWorkstationClustersResponse(workstation_clusters=[workstations.WorkstationCluster(), workstations.WorkstationCluster()]), RuntimeError)
        async_pager = await client.list_workstation_clusters(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, workstations.WorkstationCluster) for i in responses))

@pytest.mark.asyncio
async def test_list_workstation_clusters_async_pages():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_workstation_clusters), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (workstations.ListWorkstationClustersResponse(workstation_clusters=[workstations.WorkstationCluster(), workstations.WorkstationCluster(), workstations.WorkstationCluster()], next_page_token='abc'), workstations.ListWorkstationClustersResponse(workstation_clusters=[], next_page_token='def'), workstations.ListWorkstationClustersResponse(workstation_clusters=[workstations.WorkstationCluster()], next_page_token='ghi'), workstations.ListWorkstationClustersResponse(workstation_clusters=[workstations.WorkstationCluster(), workstations.WorkstationCluster()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_workstation_clusters(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [workstations.CreateWorkstationClusterRequest, dict])
def test_create_workstation_cluster(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_workstation_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_workstation_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.CreateWorkstationClusterRequest()
    assert isinstance(response, future.Future)

def test_create_workstation_cluster_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_workstation_cluster), '__call__') as call:
        client.create_workstation_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.CreateWorkstationClusterRequest()

@pytest.mark.asyncio
async def test_create_workstation_cluster_async(transport: str='grpc_asyncio', request_type=workstations.CreateWorkstationClusterRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_workstation_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_workstation_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.CreateWorkstationClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_workstation_cluster_async_from_dict():
    await test_create_workstation_cluster_async(request_type=dict)

def test_create_workstation_cluster_field_headers():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.CreateWorkstationClusterRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_workstation_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_workstation_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_workstation_cluster_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.CreateWorkstationClusterRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_workstation_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_workstation_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_workstation_cluster_flattened():
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_workstation_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_workstation_cluster(parent='parent_value', workstation_cluster=workstations.WorkstationCluster(name='name_value'), workstation_cluster_id='workstation_cluster_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].workstation_cluster
        mock_val = workstations.WorkstationCluster(name='name_value')
        assert arg == mock_val
        arg = args[0].workstation_cluster_id
        mock_val = 'workstation_cluster_id_value'
        assert arg == mock_val

def test_create_workstation_cluster_flattened_error():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_workstation_cluster(workstations.CreateWorkstationClusterRequest(), parent='parent_value', workstation_cluster=workstations.WorkstationCluster(name='name_value'), workstation_cluster_id='workstation_cluster_id_value')

@pytest.mark.asyncio
async def test_create_workstation_cluster_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_workstation_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_workstation_cluster(parent='parent_value', workstation_cluster=workstations.WorkstationCluster(name='name_value'), workstation_cluster_id='workstation_cluster_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].workstation_cluster
        mock_val = workstations.WorkstationCluster(name='name_value')
        assert arg == mock_val
        arg = args[0].workstation_cluster_id
        mock_val = 'workstation_cluster_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_workstation_cluster_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_workstation_cluster(workstations.CreateWorkstationClusterRequest(), parent='parent_value', workstation_cluster=workstations.WorkstationCluster(name='name_value'), workstation_cluster_id='workstation_cluster_id_value')

@pytest.mark.parametrize('request_type', [workstations.UpdateWorkstationClusterRequest, dict])
def test_update_workstation_cluster(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_workstation_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_workstation_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.UpdateWorkstationClusterRequest()
    assert isinstance(response, future.Future)

def test_update_workstation_cluster_empty_call():
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_workstation_cluster), '__call__') as call:
        client.update_workstation_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.UpdateWorkstationClusterRequest()

@pytest.mark.asyncio
async def test_update_workstation_cluster_async(transport: str='grpc_asyncio', request_type=workstations.UpdateWorkstationClusterRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_workstation_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_workstation_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.UpdateWorkstationClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_workstation_cluster_async_from_dict():
    await test_update_workstation_cluster_async(request_type=dict)

def test_update_workstation_cluster_field_headers():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.UpdateWorkstationClusterRequest()
    request.workstation_cluster.name = 'name_value'
    with mock.patch.object(type(client.transport.update_workstation_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_workstation_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'workstation_cluster.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_workstation_cluster_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.UpdateWorkstationClusterRequest()
    request.workstation_cluster.name = 'name_value'
    with mock.patch.object(type(client.transport.update_workstation_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_workstation_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'workstation_cluster.name=name_value') in kw['metadata']

def test_update_workstation_cluster_flattened():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_workstation_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_workstation_cluster(workstation_cluster=workstations.WorkstationCluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].workstation_cluster
        mock_val = workstations.WorkstationCluster(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_workstation_cluster_flattened_error():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_workstation_cluster(workstations.UpdateWorkstationClusterRequest(), workstation_cluster=workstations.WorkstationCluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_workstation_cluster_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_workstation_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_workstation_cluster(workstation_cluster=workstations.WorkstationCluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].workstation_cluster
        mock_val = workstations.WorkstationCluster(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_workstation_cluster_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_workstation_cluster(workstations.UpdateWorkstationClusterRequest(), workstation_cluster=workstations.WorkstationCluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [workstations.DeleteWorkstationClusterRequest, dict])
def test_delete_workstation_cluster(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_workstation_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_workstation_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.DeleteWorkstationClusterRequest()
    assert isinstance(response, future.Future)

def test_delete_workstation_cluster_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_workstation_cluster), '__call__') as call:
        client.delete_workstation_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.DeleteWorkstationClusterRequest()

@pytest.mark.asyncio
async def test_delete_workstation_cluster_async(transport: str='grpc_asyncio', request_type=workstations.DeleteWorkstationClusterRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_workstation_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_workstation_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.DeleteWorkstationClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_workstation_cluster_async_from_dict():
    await test_delete_workstation_cluster_async(request_type=dict)

def test_delete_workstation_cluster_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.DeleteWorkstationClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_workstation_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_workstation_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_workstation_cluster_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.DeleteWorkstationClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_workstation_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_workstation_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_workstation_cluster_flattened():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_workstation_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_workstation_cluster(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_workstation_cluster_flattened_error():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_workstation_cluster(workstations.DeleteWorkstationClusterRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_workstation_cluster_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_workstation_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_workstation_cluster(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_workstation_cluster_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_workstation_cluster(workstations.DeleteWorkstationClusterRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [workstations.GetWorkstationConfigRequest, dict])
def test_get_workstation_config(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_workstation_config), '__call__') as call:
        call.return_value = workstations.WorkstationConfig(name='name_value', display_name='display_name_value', uid='uid_value', reconciling=True, etag='etag_value', replica_zones=['replica_zones_value'], degraded=True, enable_audit_agent=True)
        response = client.get_workstation_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.GetWorkstationConfigRequest()
    assert isinstance(response, workstations.WorkstationConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'
    assert response.replica_zones == ['replica_zones_value']
    assert response.degraded is True
    assert response.enable_audit_agent is True

def test_get_workstation_config_empty_call():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_workstation_config), '__call__') as call:
        client.get_workstation_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.GetWorkstationConfigRequest()

@pytest.mark.asyncio
async def test_get_workstation_config_async(transport: str='grpc_asyncio', request_type=workstations.GetWorkstationConfigRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_workstation_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.WorkstationConfig(name='name_value', display_name='display_name_value', uid='uid_value', reconciling=True, etag='etag_value', replica_zones=['replica_zones_value'], degraded=True, enable_audit_agent=True))
        response = await client.get_workstation_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.GetWorkstationConfigRequest()
    assert isinstance(response, workstations.WorkstationConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'
    assert response.replica_zones == ['replica_zones_value']
    assert response.degraded is True
    assert response.enable_audit_agent is True

@pytest.mark.asyncio
async def test_get_workstation_config_async_from_dict():
    await test_get_workstation_config_async(request_type=dict)

def test_get_workstation_config_field_headers():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.GetWorkstationConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_workstation_config), '__call__') as call:
        call.return_value = workstations.WorkstationConfig()
        client.get_workstation_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_workstation_config_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.GetWorkstationConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_workstation_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.WorkstationConfig())
        await client.get_workstation_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_workstation_config_flattened():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_workstation_config), '__call__') as call:
        call.return_value = workstations.WorkstationConfig()
        client.get_workstation_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_workstation_config_flattened_error():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_workstation_config(workstations.GetWorkstationConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_workstation_config_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_workstation_config), '__call__') as call:
        call.return_value = workstations.WorkstationConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.WorkstationConfig())
        response = await client.get_workstation_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_workstation_config_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_workstation_config(workstations.GetWorkstationConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [workstations.ListWorkstationConfigsRequest, dict])
def test_list_workstation_configs(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_workstation_configs), '__call__') as call:
        call.return_value = workstations.ListWorkstationConfigsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_workstation_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.ListWorkstationConfigsRequest()
    assert isinstance(response, pagers.ListWorkstationConfigsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_workstation_configs_empty_call():
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_workstation_configs), '__call__') as call:
        client.list_workstation_configs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.ListWorkstationConfigsRequest()

@pytest.mark.asyncio
async def test_list_workstation_configs_async(transport: str='grpc_asyncio', request_type=workstations.ListWorkstationConfigsRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_workstation_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.ListWorkstationConfigsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_workstation_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.ListWorkstationConfigsRequest()
    assert isinstance(response, pagers.ListWorkstationConfigsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_workstation_configs_async_from_dict():
    await test_list_workstation_configs_async(request_type=dict)

def test_list_workstation_configs_field_headers():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.ListWorkstationConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_workstation_configs), '__call__') as call:
        call.return_value = workstations.ListWorkstationConfigsResponse()
        client.list_workstation_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_workstation_configs_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.ListWorkstationConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_workstation_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.ListWorkstationConfigsResponse())
        await client.list_workstation_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_workstation_configs_flattened():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_workstation_configs), '__call__') as call:
        call.return_value = workstations.ListWorkstationConfigsResponse()
        client.list_workstation_configs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_workstation_configs_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_workstation_configs(workstations.ListWorkstationConfigsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_workstation_configs_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_workstation_configs), '__call__') as call:
        call.return_value = workstations.ListWorkstationConfigsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.ListWorkstationConfigsResponse())
        response = await client.list_workstation_configs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_workstation_configs_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_workstation_configs(workstations.ListWorkstationConfigsRequest(), parent='parent_value')

def test_list_workstation_configs_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_workstation_configs), '__call__') as call:
        call.side_effect = (workstations.ListWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig(), workstations.WorkstationConfig()], next_page_token='abc'), workstations.ListWorkstationConfigsResponse(workstation_configs=[], next_page_token='def'), workstations.ListWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig()], next_page_token='ghi'), workstations.ListWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_workstation_configs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, workstations.WorkstationConfig) for i in results))

def test_list_workstation_configs_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_workstation_configs), '__call__') as call:
        call.side_effect = (workstations.ListWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig(), workstations.WorkstationConfig()], next_page_token='abc'), workstations.ListWorkstationConfigsResponse(workstation_configs=[], next_page_token='def'), workstations.ListWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig()], next_page_token='ghi'), workstations.ListWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig()]), RuntimeError)
        pages = list(client.list_workstation_configs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_workstation_configs_async_pager():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_workstation_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (workstations.ListWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig(), workstations.WorkstationConfig()], next_page_token='abc'), workstations.ListWorkstationConfigsResponse(workstation_configs=[], next_page_token='def'), workstations.ListWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig()], next_page_token='ghi'), workstations.ListWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig()]), RuntimeError)
        async_pager = await client.list_workstation_configs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, workstations.WorkstationConfig) for i in responses))

@pytest.mark.asyncio
async def test_list_workstation_configs_async_pages():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_workstation_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (workstations.ListWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig(), workstations.WorkstationConfig()], next_page_token='abc'), workstations.ListWorkstationConfigsResponse(workstation_configs=[], next_page_token='def'), workstations.ListWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig()], next_page_token='ghi'), workstations.ListWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_workstation_configs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [workstations.ListUsableWorkstationConfigsRequest, dict])
def test_list_usable_workstation_configs(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_usable_workstation_configs), '__call__') as call:
        call.return_value = workstations.ListUsableWorkstationConfigsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_usable_workstation_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.ListUsableWorkstationConfigsRequest()
    assert isinstance(response, pagers.ListUsableWorkstationConfigsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_usable_workstation_configs_empty_call():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_usable_workstation_configs), '__call__') as call:
        client.list_usable_workstation_configs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.ListUsableWorkstationConfigsRequest()

@pytest.mark.asyncio
async def test_list_usable_workstation_configs_async(transport: str='grpc_asyncio', request_type=workstations.ListUsableWorkstationConfigsRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_usable_workstation_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.ListUsableWorkstationConfigsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_usable_workstation_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.ListUsableWorkstationConfigsRequest()
    assert isinstance(response, pagers.ListUsableWorkstationConfigsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_usable_workstation_configs_async_from_dict():
    await test_list_usable_workstation_configs_async(request_type=dict)

def test_list_usable_workstation_configs_field_headers():
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.ListUsableWorkstationConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_usable_workstation_configs), '__call__') as call:
        call.return_value = workstations.ListUsableWorkstationConfigsResponse()
        client.list_usable_workstation_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_usable_workstation_configs_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.ListUsableWorkstationConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_usable_workstation_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.ListUsableWorkstationConfigsResponse())
        await client.list_usable_workstation_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_usable_workstation_configs_flattened():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_usable_workstation_configs), '__call__') as call:
        call.return_value = workstations.ListUsableWorkstationConfigsResponse()
        client.list_usable_workstation_configs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_usable_workstation_configs_flattened_error():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_usable_workstation_configs(workstations.ListUsableWorkstationConfigsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_usable_workstation_configs_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_usable_workstation_configs), '__call__') as call:
        call.return_value = workstations.ListUsableWorkstationConfigsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.ListUsableWorkstationConfigsResponse())
        response = await client.list_usable_workstation_configs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_usable_workstation_configs_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_usable_workstation_configs(workstations.ListUsableWorkstationConfigsRequest(), parent='parent_value')

def test_list_usable_workstation_configs_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_usable_workstation_configs), '__call__') as call:
        call.side_effect = (workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig(), workstations.WorkstationConfig()], next_page_token='abc'), workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[], next_page_token='def'), workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig()], next_page_token='ghi'), workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_usable_workstation_configs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, workstations.WorkstationConfig) for i in results))

def test_list_usable_workstation_configs_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_usable_workstation_configs), '__call__') as call:
        call.side_effect = (workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig(), workstations.WorkstationConfig()], next_page_token='abc'), workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[], next_page_token='def'), workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig()], next_page_token='ghi'), workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig()]), RuntimeError)
        pages = list(client.list_usable_workstation_configs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_usable_workstation_configs_async_pager():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_usable_workstation_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig(), workstations.WorkstationConfig()], next_page_token='abc'), workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[], next_page_token='def'), workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig()], next_page_token='ghi'), workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig()]), RuntimeError)
        async_pager = await client.list_usable_workstation_configs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, workstations.WorkstationConfig) for i in responses))

@pytest.mark.asyncio
async def test_list_usable_workstation_configs_async_pages():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_usable_workstation_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig(), workstations.WorkstationConfig()], next_page_token='abc'), workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[], next_page_token='def'), workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig()], next_page_token='ghi'), workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_usable_workstation_configs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [workstations.CreateWorkstationConfigRequest, dict])
def test_create_workstation_config(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_workstation_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_workstation_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.CreateWorkstationConfigRequest()
    assert isinstance(response, future.Future)

def test_create_workstation_config_empty_call():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_workstation_config), '__call__') as call:
        client.create_workstation_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.CreateWorkstationConfigRequest()

@pytest.mark.asyncio
async def test_create_workstation_config_async(transport: str='grpc_asyncio', request_type=workstations.CreateWorkstationConfigRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_workstation_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_workstation_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.CreateWorkstationConfigRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_workstation_config_async_from_dict():
    await test_create_workstation_config_async(request_type=dict)

def test_create_workstation_config_field_headers():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.CreateWorkstationConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_workstation_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_workstation_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_workstation_config_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.CreateWorkstationConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_workstation_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_workstation_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_workstation_config_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_workstation_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_workstation_config(parent='parent_value', workstation_config=workstations.WorkstationConfig(name='name_value'), workstation_config_id='workstation_config_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].workstation_config
        mock_val = workstations.WorkstationConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].workstation_config_id
        mock_val = 'workstation_config_id_value'
        assert arg == mock_val

def test_create_workstation_config_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_workstation_config(workstations.CreateWorkstationConfigRequest(), parent='parent_value', workstation_config=workstations.WorkstationConfig(name='name_value'), workstation_config_id='workstation_config_id_value')

@pytest.mark.asyncio
async def test_create_workstation_config_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_workstation_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_workstation_config(parent='parent_value', workstation_config=workstations.WorkstationConfig(name='name_value'), workstation_config_id='workstation_config_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].workstation_config
        mock_val = workstations.WorkstationConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].workstation_config_id
        mock_val = 'workstation_config_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_workstation_config_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_workstation_config(workstations.CreateWorkstationConfigRequest(), parent='parent_value', workstation_config=workstations.WorkstationConfig(name='name_value'), workstation_config_id='workstation_config_id_value')

@pytest.mark.parametrize('request_type', [workstations.UpdateWorkstationConfigRequest, dict])
def test_update_workstation_config(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_workstation_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_workstation_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.UpdateWorkstationConfigRequest()
    assert isinstance(response, future.Future)

def test_update_workstation_config_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_workstation_config), '__call__') as call:
        client.update_workstation_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.UpdateWorkstationConfigRequest()

@pytest.mark.asyncio
async def test_update_workstation_config_async(transport: str='grpc_asyncio', request_type=workstations.UpdateWorkstationConfigRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_workstation_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_workstation_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.UpdateWorkstationConfigRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_workstation_config_async_from_dict():
    await test_update_workstation_config_async(request_type=dict)

def test_update_workstation_config_field_headers():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.UpdateWorkstationConfigRequest()
    request.workstation_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_workstation_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_workstation_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'workstation_config.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_workstation_config_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.UpdateWorkstationConfigRequest()
    request.workstation_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_workstation_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_workstation_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'workstation_config.name=name_value') in kw['metadata']

def test_update_workstation_config_flattened():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_workstation_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_workstation_config(workstation_config=workstations.WorkstationConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].workstation_config
        mock_val = workstations.WorkstationConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_workstation_config_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_workstation_config(workstations.UpdateWorkstationConfigRequest(), workstation_config=workstations.WorkstationConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_workstation_config_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_workstation_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_workstation_config(workstation_config=workstations.WorkstationConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].workstation_config
        mock_val = workstations.WorkstationConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_workstation_config_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_workstation_config(workstations.UpdateWorkstationConfigRequest(), workstation_config=workstations.WorkstationConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [workstations.DeleteWorkstationConfigRequest, dict])
def test_delete_workstation_config(request_type, transport: str='grpc'):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_workstation_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_workstation_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.DeleteWorkstationConfigRequest()
    assert isinstance(response, future.Future)

def test_delete_workstation_config_empty_call():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_workstation_config), '__call__') as call:
        client.delete_workstation_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.DeleteWorkstationConfigRequest()

@pytest.mark.asyncio
async def test_delete_workstation_config_async(transport: str='grpc_asyncio', request_type=workstations.DeleteWorkstationConfigRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_workstation_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_workstation_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.DeleteWorkstationConfigRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_workstation_config_async_from_dict():
    await test_delete_workstation_config_async(request_type=dict)

def test_delete_workstation_config_field_headers():
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.DeleteWorkstationConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_workstation_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_workstation_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_workstation_config_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.DeleteWorkstationConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_workstation_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_workstation_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_workstation_config_flattened():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_workstation_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_workstation_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_workstation_config_flattened_error():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_workstation_config(workstations.DeleteWorkstationConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_workstation_config_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_workstation_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_workstation_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_workstation_config_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_workstation_config(workstations.DeleteWorkstationConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [workstations.GetWorkstationRequest, dict])
def test_get_workstation(request_type, transport: str='grpc'):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_workstation), '__call__') as call:
        call.return_value = workstations.Workstation(name='name_value', display_name='display_name_value', uid='uid_value', reconciling=True, etag='etag_value', state=workstations.Workstation.State.STATE_STARTING, host='host_value')
        response = client.get_workstation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.GetWorkstationRequest()
    assert isinstance(response, workstations.Workstation)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'
    assert response.state == workstations.Workstation.State.STATE_STARTING
    assert response.host == 'host_value'

def test_get_workstation_empty_call():
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_workstation), '__call__') as call:
        client.get_workstation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.GetWorkstationRequest()

@pytest.mark.asyncio
async def test_get_workstation_async(transport: str='grpc_asyncio', request_type=workstations.GetWorkstationRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_workstation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.Workstation(name='name_value', display_name='display_name_value', uid='uid_value', reconciling=True, etag='etag_value', state=workstations.Workstation.State.STATE_STARTING, host='host_value'))
        response = await client.get_workstation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.GetWorkstationRequest()
    assert isinstance(response, workstations.Workstation)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'
    assert response.state == workstations.Workstation.State.STATE_STARTING
    assert response.host == 'host_value'

@pytest.mark.asyncio
async def test_get_workstation_async_from_dict():
    await test_get_workstation_async(request_type=dict)

def test_get_workstation_field_headers():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.GetWorkstationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_workstation), '__call__') as call:
        call.return_value = workstations.Workstation()
        client.get_workstation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_workstation_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.GetWorkstationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_workstation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.Workstation())
        await client.get_workstation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_workstation_flattened():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_workstation), '__call__') as call:
        call.return_value = workstations.Workstation()
        client.get_workstation(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_workstation_flattened_error():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_workstation(workstations.GetWorkstationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_workstation_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_workstation), '__call__') as call:
        call.return_value = workstations.Workstation()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.Workstation())
        response = await client.get_workstation(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_workstation_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_workstation(workstations.GetWorkstationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [workstations.ListWorkstationsRequest, dict])
def test_list_workstations(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_workstations), '__call__') as call:
        call.return_value = workstations.ListWorkstationsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_workstations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.ListWorkstationsRequest()
    assert isinstance(response, pagers.ListWorkstationsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_workstations_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_workstations), '__call__') as call:
        client.list_workstations()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.ListWorkstationsRequest()

@pytest.mark.asyncio
async def test_list_workstations_async(transport: str='grpc_asyncio', request_type=workstations.ListWorkstationsRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_workstations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.ListWorkstationsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_workstations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.ListWorkstationsRequest()
    assert isinstance(response, pagers.ListWorkstationsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_workstations_async_from_dict():
    await test_list_workstations_async(request_type=dict)

def test_list_workstations_field_headers():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.ListWorkstationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_workstations), '__call__') as call:
        call.return_value = workstations.ListWorkstationsResponse()
        client.list_workstations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_workstations_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.ListWorkstationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_workstations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.ListWorkstationsResponse())
        await client.list_workstations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_workstations_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_workstations), '__call__') as call:
        call.return_value = workstations.ListWorkstationsResponse()
        client.list_workstations(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_workstations_flattened_error():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_workstations(workstations.ListWorkstationsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_workstations_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_workstations), '__call__') as call:
        call.return_value = workstations.ListWorkstationsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.ListWorkstationsResponse())
        response = await client.list_workstations(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_workstations_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_workstations(workstations.ListWorkstationsRequest(), parent='parent_value')

def test_list_workstations_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_workstations), '__call__') as call:
        call.side_effect = (workstations.ListWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation(), workstations.Workstation()], next_page_token='abc'), workstations.ListWorkstationsResponse(workstations=[], next_page_token='def'), workstations.ListWorkstationsResponse(workstations=[workstations.Workstation()], next_page_token='ghi'), workstations.ListWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_workstations(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, workstations.Workstation) for i in results))

def test_list_workstations_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_workstations), '__call__') as call:
        call.side_effect = (workstations.ListWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation(), workstations.Workstation()], next_page_token='abc'), workstations.ListWorkstationsResponse(workstations=[], next_page_token='def'), workstations.ListWorkstationsResponse(workstations=[workstations.Workstation()], next_page_token='ghi'), workstations.ListWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation()]), RuntimeError)
        pages = list(client.list_workstations(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_workstations_async_pager():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_workstations), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (workstations.ListWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation(), workstations.Workstation()], next_page_token='abc'), workstations.ListWorkstationsResponse(workstations=[], next_page_token='def'), workstations.ListWorkstationsResponse(workstations=[workstations.Workstation()], next_page_token='ghi'), workstations.ListWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation()]), RuntimeError)
        async_pager = await client.list_workstations(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, workstations.Workstation) for i in responses))

@pytest.mark.asyncio
async def test_list_workstations_async_pages():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_workstations), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (workstations.ListWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation(), workstations.Workstation()], next_page_token='abc'), workstations.ListWorkstationsResponse(workstations=[], next_page_token='def'), workstations.ListWorkstationsResponse(workstations=[workstations.Workstation()], next_page_token='ghi'), workstations.ListWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_workstations(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [workstations.ListUsableWorkstationsRequest, dict])
def test_list_usable_workstations(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_usable_workstations), '__call__') as call:
        call.return_value = workstations.ListUsableWorkstationsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_usable_workstations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.ListUsableWorkstationsRequest()
    assert isinstance(response, pagers.ListUsableWorkstationsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_usable_workstations_empty_call():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_usable_workstations), '__call__') as call:
        client.list_usable_workstations()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.ListUsableWorkstationsRequest()

@pytest.mark.asyncio
async def test_list_usable_workstations_async(transport: str='grpc_asyncio', request_type=workstations.ListUsableWorkstationsRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_usable_workstations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.ListUsableWorkstationsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_usable_workstations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.ListUsableWorkstationsRequest()
    assert isinstance(response, pagers.ListUsableWorkstationsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_usable_workstations_async_from_dict():
    await test_list_usable_workstations_async(request_type=dict)

def test_list_usable_workstations_field_headers():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.ListUsableWorkstationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_usable_workstations), '__call__') as call:
        call.return_value = workstations.ListUsableWorkstationsResponse()
        client.list_usable_workstations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_usable_workstations_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.ListUsableWorkstationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_usable_workstations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.ListUsableWorkstationsResponse())
        await client.list_usable_workstations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_usable_workstations_flattened():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_usable_workstations), '__call__') as call:
        call.return_value = workstations.ListUsableWorkstationsResponse()
        client.list_usable_workstations(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_usable_workstations_flattened_error():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_usable_workstations(workstations.ListUsableWorkstationsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_usable_workstations_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_usable_workstations), '__call__') as call:
        call.return_value = workstations.ListUsableWorkstationsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.ListUsableWorkstationsResponse())
        response = await client.list_usable_workstations(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_usable_workstations_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_usable_workstations(workstations.ListUsableWorkstationsRequest(), parent='parent_value')

def test_list_usable_workstations_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_usable_workstations), '__call__') as call:
        call.side_effect = (workstations.ListUsableWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation(), workstations.Workstation()], next_page_token='abc'), workstations.ListUsableWorkstationsResponse(workstations=[], next_page_token='def'), workstations.ListUsableWorkstationsResponse(workstations=[workstations.Workstation()], next_page_token='ghi'), workstations.ListUsableWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_usable_workstations(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, workstations.Workstation) for i in results))

def test_list_usable_workstations_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_usable_workstations), '__call__') as call:
        call.side_effect = (workstations.ListUsableWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation(), workstations.Workstation()], next_page_token='abc'), workstations.ListUsableWorkstationsResponse(workstations=[], next_page_token='def'), workstations.ListUsableWorkstationsResponse(workstations=[workstations.Workstation()], next_page_token='ghi'), workstations.ListUsableWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation()]), RuntimeError)
        pages = list(client.list_usable_workstations(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_usable_workstations_async_pager():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_usable_workstations), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (workstations.ListUsableWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation(), workstations.Workstation()], next_page_token='abc'), workstations.ListUsableWorkstationsResponse(workstations=[], next_page_token='def'), workstations.ListUsableWorkstationsResponse(workstations=[workstations.Workstation()], next_page_token='ghi'), workstations.ListUsableWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation()]), RuntimeError)
        async_pager = await client.list_usable_workstations(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, workstations.Workstation) for i in responses))

@pytest.mark.asyncio
async def test_list_usable_workstations_async_pages():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_usable_workstations), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (workstations.ListUsableWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation(), workstations.Workstation()], next_page_token='abc'), workstations.ListUsableWorkstationsResponse(workstations=[], next_page_token='def'), workstations.ListUsableWorkstationsResponse(workstations=[workstations.Workstation()], next_page_token='ghi'), workstations.ListUsableWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_usable_workstations(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [workstations.CreateWorkstationRequest, dict])
def test_create_workstation(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_workstation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.CreateWorkstationRequest()
    assert isinstance(response, future.Future)

def test_create_workstation_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_workstation), '__call__') as call:
        client.create_workstation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.CreateWorkstationRequest()

@pytest.mark.asyncio
async def test_create_workstation_async(transport: str='grpc_asyncio', request_type=workstations.CreateWorkstationRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_workstation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_workstation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.CreateWorkstationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_workstation_async_from_dict():
    await test_create_workstation_async(request_type=dict)

def test_create_workstation_field_headers():
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.CreateWorkstationRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_workstation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_workstation_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.CreateWorkstationRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_workstation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_workstation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_workstation_flattened():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_workstation(parent='parent_value', workstation=workstations.Workstation(name='name_value'), workstation_id='workstation_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].workstation
        mock_val = workstations.Workstation(name='name_value')
        assert arg == mock_val
        arg = args[0].workstation_id
        mock_val = 'workstation_id_value'
        assert arg == mock_val

def test_create_workstation_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_workstation(workstations.CreateWorkstationRequest(), parent='parent_value', workstation=workstations.Workstation(name='name_value'), workstation_id='workstation_id_value')

@pytest.mark.asyncio
async def test_create_workstation_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_workstation(parent='parent_value', workstation=workstations.Workstation(name='name_value'), workstation_id='workstation_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].workstation
        mock_val = workstations.Workstation(name='name_value')
        assert arg == mock_val
        arg = args[0].workstation_id
        mock_val = 'workstation_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_workstation_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_workstation(workstations.CreateWorkstationRequest(), parent='parent_value', workstation=workstations.Workstation(name='name_value'), workstation_id='workstation_id_value')

@pytest.mark.parametrize('request_type', [workstations.UpdateWorkstationRequest, dict])
def test_update_workstation(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_workstation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.UpdateWorkstationRequest()
    assert isinstance(response, future.Future)

def test_update_workstation_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_workstation), '__call__') as call:
        client.update_workstation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.UpdateWorkstationRequest()

@pytest.mark.asyncio
async def test_update_workstation_async(transport: str='grpc_asyncio', request_type=workstations.UpdateWorkstationRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_workstation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_workstation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.UpdateWorkstationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_workstation_async_from_dict():
    await test_update_workstation_async(request_type=dict)

def test_update_workstation_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.UpdateWorkstationRequest()
    request.workstation.name = 'name_value'
    with mock.patch.object(type(client.transport.update_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_workstation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'workstation.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_workstation_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.UpdateWorkstationRequest()
    request.workstation.name = 'name_value'
    with mock.patch.object(type(client.transport.update_workstation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_workstation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'workstation.name=name_value') in kw['metadata']

def test_update_workstation_flattened():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_workstation(workstation=workstations.Workstation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].workstation
        mock_val = workstations.Workstation(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_workstation_flattened_error():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_workstation(workstations.UpdateWorkstationRequest(), workstation=workstations.Workstation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_workstation_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_workstation(workstation=workstations.Workstation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].workstation
        mock_val = workstations.Workstation(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_workstation_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_workstation(workstations.UpdateWorkstationRequest(), workstation=workstations.Workstation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [workstations.DeleteWorkstationRequest, dict])
def test_delete_workstation(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_workstation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.DeleteWorkstationRequest()
    assert isinstance(response, future.Future)

def test_delete_workstation_empty_call():
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_workstation), '__call__') as call:
        client.delete_workstation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.DeleteWorkstationRequest()

@pytest.mark.asyncio
async def test_delete_workstation_async(transport: str='grpc_asyncio', request_type=workstations.DeleteWorkstationRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_workstation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_workstation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.DeleteWorkstationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_workstation_async_from_dict():
    await test_delete_workstation_async(request_type=dict)

def test_delete_workstation_field_headers():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.DeleteWorkstationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_workstation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_workstation_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.DeleteWorkstationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_workstation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_workstation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_workstation_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_workstation(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_workstation_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_workstation(workstations.DeleteWorkstationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_workstation_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_workstation(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_workstation_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_workstation(workstations.DeleteWorkstationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [workstations.StartWorkstationRequest, dict])
def test_start_workstation(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.start_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.start_workstation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.StartWorkstationRequest()
    assert isinstance(response, future.Future)

def test_start_workstation_empty_call():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.start_workstation), '__call__') as call:
        client.start_workstation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.StartWorkstationRequest()

@pytest.mark.asyncio
async def test_start_workstation_async(transport: str='grpc_asyncio', request_type=workstations.StartWorkstationRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.start_workstation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.start_workstation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.StartWorkstationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_start_workstation_async_from_dict():
    await test_start_workstation_async(request_type=dict)

def test_start_workstation_field_headers():
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.StartWorkstationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.start_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.start_workstation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_start_workstation_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.StartWorkstationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.start_workstation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.start_workstation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_start_workstation_flattened():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.start_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.start_workstation(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_start_workstation_flattened_error():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.start_workstation(workstations.StartWorkstationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_start_workstation_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.start_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.start_workstation(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_start_workstation_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.start_workstation(workstations.StartWorkstationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [workstations.StopWorkstationRequest, dict])
def test_stop_workstation(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.stop_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.stop_workstation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.StopWorkstationRequest()
    assert isinstance(response, future.Future)

def test_stop_workstation_empty_call():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.stop_workstation), '__call__') as call:
        client.stop_workstation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.StopWorkstationRequest()

@pytest.mark.asyncio
async def test_stop_workstation_async(transport: str='grpc_asyncio', request_type=workstations.StopWorkstationRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.stop_workstation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.stop_workstation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.StopWorkstationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_stop_workstation_async_from_dict():
    await test_stop_workstation_async(request_type=dict)

def test_stop_workstation_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.StopWorkstationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.stop_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.stop_workstation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_stop_workstation_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.StopWorkstationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.stop_workstation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.stop_workstation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_stop_workstation_flattened():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.stop_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.stop_workstation(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_stop_workstation_flattened_error():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.stop_workstation(workstations.StopWorkstationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_stop_workstation_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.stop_workstation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.stop_workstation(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_stop_workstation_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.stop_workstation(workstations.StopWorkstationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [workstations.GenerateAccessTokenRequest, dict])
def test_generate_access_token(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_access_token), '__call__') as call:
        call.return_value = workstations.GenerateAccessTokenResponse(access_token='access_token_value')
        response = client.generate_access_token(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.GenerateAccessTokenRequest()
    assert isinstance(response, workstations.GenerateAccessTokenResponse)
    assert response.access_token == 'access_token_value'

def test_generate_access_token_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.generate_access_token), '__call__') as call:
        client.generate_access_token()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.GenerateAccessTokenRequest()

@pytest.mark.asyncio
async def test_generate_access_token_async(transport: str='grpc_asyncio', request_type=workstations.GenerateAccessTokenRequest):
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_access_token), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.GenerateAccessTokenResponse(access_token='access_token_value'))
        response = await client.generate_access_token(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workstations.GenerateAccessTokenRequest()
    assert isinstance(response, workstations.GenerateAccessTokenResponse)
    assert response.access_token == 'access_token_value'

@pytest.mark.asyncio
async def test_generate_access_token_async_from_dict():
    await test_generate_access_token_async(request_type=dict)

def test_generate_access_token_field_headers():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.GenerateAccessTokenRequest()
    request.workstation = 'workstation_value'
    with mock.patch.object(type(client.transport.generate_access_token), '__call__') as call:
        call.return_value = workstations.GenerateAccessTokenResponse()
        client.generate_access_token(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'workstation=workstation_value') in kw['metadata']

@pytest.mark.asyncio
async def test_generate_access_token_field_headers_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workstations.GenerateAccessTokenRequest()
    request.workstation = 'workstation_value'
    with mock.patch.object(type(client.transport.generate_access_token), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.GenerateAccessTokenResponse())
        await client.generate_access_token(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'workstation=workstation_value') in kw['metadata']

def test_generate_access_token_flattened():
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.generate_access_token), '__call__') as call:
        call.return_value = workstations.GenerateAccessTokenResponse()
        client.generate_access_token(workstation='workstation_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].workstation
        mock_val = 'workstation_value'
        assert arg == mock_val

def test_generate_access_token_flattened_error():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.generate_access_token(workstations.GenerateAccessTokenRequest(), workstation='workstation_value')

@pytest.mark.asyncio
async def test_generate_access_token_flattened_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.generate_access_token), '__call__') as call:
        call.return_value = workstations.GenerateAccessTokenResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workstations.GenerateAccessTokenResponse())
        response = await client.generate_access_token(workstation='workstation_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].workstation
        mock_val = 'workstation_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_generate_access_token_flattened_error_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.generate_access_token(workstations.GenerateAccessTokenRequest(), workstation='workstation_value')

@pytest.mark.parametrize('request_type', [workstations.GetWorkstationClusterRequest, dict])
def test_get_workstation_cluster_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workstations.WorkstationCluster(name='name_value', display_name='display_name_value', uid='uid_value', reconciling=True, etag='etag_value', network='network_value', subnetwork='subnetwork_value', control_plane_ip='control_plane_ip_value', degraded=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = workstations.WorkstationCluster.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_workstation_cluster(request)
    assert isinstance(response, workstations.WorkstationCluster)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'
    assert response.network == 'network_value'
    assert response.subnetwork == 'subnetwork_value'
    assert response.control_plane_ip == 'control_plane_ip_value'
    assert response.degraded is True

def test_get_workstation_cluster_rest_required_fields(request_type=workstations.GetWorkstationClusterRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_workstation_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_workstation_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = workstations.WorkstationCluster()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = workstations.WorkstationCluster.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_workstation_cluster(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_workstation_cluster_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_workstation_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_workstation_cluster_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WorkstationsRestInterceptor, 'post_get_workstation_cluster') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_get_workstation_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.GetWorkstationClusterRequest.pb(workstations.GetWorkstationClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = workstations.WorkstationCluster.to_json(workstations.WorkstationCluster())
        request = workstations.GetWorkstationClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = workstations.WorkstationCluster()
        client.get_workstation_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_workstation_cluster_rest_bad_request(transport: str='rest', request_type=workstations.GetWorkstationClusterRequest):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_workstation_cluster(request)

def test_get_workstation_cluster_rest_flattened():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workstations.WorkstationCluster()
        sample_request = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = workstations.WorkstationCluster.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_workstation_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/workstationClusters/*}' % client.transport._host, args[1])

def test_get_workstation_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_workstation_cluster(workstations.GetWorkstationClusterRequest(), name='name_value')

def test_get_workstation_cluster_rest_error():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workstations.ListWorkstationClustersRequest, dict])
def test_list_workstation_clusters_rest(request_type):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workstations.ListWorkstationClustersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = workstations.ListWorkstationClustersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_workstation_clusters(request)
    assert isinstance(response, pagers.ListWorkstationClustersPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_workstation_clusters_rest_required_fields(request_type=workstations.ListWorkstationClustersRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_workstation_clusters._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_workstation_clusters._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = workstations.ListWorkstationClustersResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = workstations.ListWorkstationClustersResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_workstation_clusters(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_workstation_clusters_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_workstation_clusters._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_workstation_clusters_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WorkstationsRestInterceptor, 'post_list_workstation_clusters') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_list_workstation_clusters') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.ListWorkstationClustersRequest.pb(workstations.ListWorkstationClustersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = workstations.ListWorkstationClustersResponse.to_json(workstations.ListWorkstationClustersResponse())
        request = workstations.ListWorkstationClustersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = workstations.ListWorkstationClustersResponse()
        client.list_workstation_clusters(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_workstation_clusters_rest_bad_request(transport: str='rest', request_type=workstations.ListWorkstationClustersRequest):
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_workstation_clusters(request)

def test_list_workstation_clusters_rest_flattened():
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workstations.ListWorkstationClustersResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = workstations.ListWorkstationClustersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_workstation_clusters(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*}/workstationClusters' % client.transport._host, args[1])

def test_list_workstation_clusters_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_workstation_clusters(workstations.ListWorkstationClustersRequest(), parent='parent_value')

def test_list_workstation_clusters_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (workstations.ListWorkstationClustersResponse(workstation_clusters=[workstations.WorkstationCluster(), workstations.WorkstationCluster(), workstations.WorkstationCluster()], next_page_token='abc'), workstations.ListWorkstationClustersResponse(workstation_clusters=[], next_page_token='def'), workstations.ListWorkstationClustersResponse(workstation_clusters=[workstations.WorkstationCluster()], next_page_token='ghi'), workstations.ListWorkstationClustersResponse(workstation_clusters=[workstations.WorkstationCluster(), workstations.WorkstationCluster()]))
        response = response + response
        response = tuple((workstations.ListWorkstationClustersResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_workstation_clusters(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, workstations.WorkstationCluster) for i in results))
        pages = list(client.list_workstation_clusters(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [workstations.CreateWorkstationClusterRequest, dict])
def test_create_workstation_cluster_rest(request_type):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['workstation_cluster'] = {'name': 'name_value', 'display_name': 'display_name_value', 'uid': 'uid_value', 'reconciling': True, 'annotations': {}, 'labels': {}, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'etag': 'etag_value', 'network': 'network_value', 'subnetwork': 'subnetwork_value', 'control_plane_ip': 'control_plane_ip_value', 'private_cluster_config': {'enable_private_endpoint': True, 'cluster_hostname': 'cluster_hostname_value', 'service_attachment_uri': 'service_attachment_uri_value', 'allowed_projects': ['allowed_projects_value1', 'allowed_projects_value2']}, 'degraded': True, 'conditions': [{'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}]}
    test_field = workstations.CreateWorkstationClusterRequest.meta.fields['workstation_cluster']

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
    for (field, value) in request_init['workstation_cluster'].items():
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
                for i in range(0, len(request_init['workstation_cluster'][field])):
                    del request_init['workstation_cluster'][field][i][subfield]
            else:
                del request_init['workstation_cluster'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_workstation_cluster(request)
    assert response.operation.name == 'operations/spam'

def test_create_workstation_cluster_rest_required_fields(request_type=workstations.CreateWorkstationClusterRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['workstation_cluster_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'workstationClusterId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_workstation_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'workstationClusterId' in jsonified_request
    assert jsonified_request['workstationClusterId'] == request_init['workstation_cluster_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['workstationClusterId'] = 'workstation_cluster_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_workstation_cluster._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('validate_only', 'workstation_cluster_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'workstationClusterId' in jsonified_request
    assert jsonified_request['workstationClusterId'] == 'workstation_cluster_id_value'
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_workstation_cluster(request)
            expected_params = [('workstationClusterId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_workstation_cluster_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_workstation_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(('validateOnly', 'workstationClusterId')) & set(('parent', 'workstationClusterId', 'workstationCluster'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_workstation_cluster_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.WorkstationsRestInterceptor, 'post_create_workstation_cluster') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_create_workstation_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.CreateWorkstationClusterRequest.pb(workstations.CreateWorkstationClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = workstations.CreateWorkstationClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_workstation_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_workstation_cluster_rest_bad_request(transport: str='rest', request_type=workstations.CreateWorkstationClusterRequest):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_workstation_cluster(request)

def test_create_workstation_cluster_rest_flattened():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', workstation_cluster=workstations.WorkstationCluster(name='name_value'), workstation_cluster_id='workstation_cluster_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_workstation_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*}/workstationClusters' % client.transport._host, args[1])

def test_create_workstation_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_workstation_cluster(workstations.CreateWorkstationClusterRequest(), parent='parent_value', workstation_cluster=workstations.WorkstationCluster(name='name_value'), workstation_cluster_id='workstation_cluster_id_value')

def test_create_workstation_cluster_rest_error():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workstations.UpdateWorkstationClusterRequest, dict])
def test_update_workstation_cluster_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'workstation_cluster': {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3'}}
    request_init['workstation_cluster'] = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3', 'display_name': 'display_name_value', 'uid': 'uid_value', 'reconciling': True, 'annotations': {}, 'labels': {}, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'etag': 'etag_value', 'network': 'network_value', 'subnetwork': 'subnetwork_value', 'control_plane_ip': 'control_plane_ip_value', 'private_cluster_config': {'enable_private_endpoint': True, 'cluster_hostname': 'cluster_hostname_value', 'service_attachment_uri': 'service_attachment_uri_value', 'allowed_projects': ['allowed_projects_value1', 'allowed_projects_value2']}, 'degraded': True, 'conditions': [{'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}]}
    test_field = workstations.UpdateWorkstationClusterRequest.meta.fields['workstation_cluster']

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
    for (field, value) in request_init['workstation_cluster'].items():
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
                for i in range(0, len(request_init['workstation_cluster'][field])):
                    del request_init['workstation_cluster'][field][i][subfield]
            else:
                del request_init['workstation_cluster'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_workstation_cluster(request)
    assert response.operation.name == 'operations/spam'

def test_update_workstation_cluster_rest_required_fields(request_type=workstations.UpdateWorkstationClusterRequest):
    if False:
        return 10
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_workstation_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_workstation_cluster._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_workstation_cluster(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_workstation_cluster_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_workstation_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'updateMask', 'validateOnly')) & set(('workstationCluster', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_workstation_cluster_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.WorkstationsRestInterceptor, 'post_update_workstation_cluster') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_update_workstation_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.UpdateWorkstationClusterRequest.pb(workstations.UpdateWorkstationClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = workstations.UpdateWorkstationClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_workstation_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_workstation_cluster_rest_bad_request(transport: str='rest', request_type=workstations.UpdateWorkstationClusterRequest):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'workstation_cluster': {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_workstation_cluster(request)

def test_update_workstation_cluster_rest_flattened():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'workstation_cluster': {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3'}}
        mock_args = dict(workstation_cluster=workstations.WorkstationCluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_workstation_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{workstation_cluster.name=projects/*/locations/*/workstationClusters/*}' % client.transport._host, args[1])

def test_update_workstation_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_workstation_cluster(workstations.UpdateWorkstationClusterRequest(), workstation_cluster=workstations.WorkstationCluster(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_workstation_cluster_rest_error():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workstations.DeleteWorkstationClusterRequest, dict])
def test_delete_workstation_cluster_rest(request_type):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_workstation_cluster(request)
    assert response.operation.name == 'operations/spam'

def test_delete_workstation_cluster_rest_required_fields(request_type=workstations.DeleteWorkstationClusterRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_workstation_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_workstation_cluster._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('etag', 'force', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_workstation_cluster(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_workstation_cluster_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_workstation_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(('etag', 'force', 'validateOnly')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_workstation_cluster_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.WorkstationsRestInterceptor, 'post_delete_workstation_cluster') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_delete_workstation_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.DeleteWorkstationClusterRequest.pb(workstations.DeleteWorkstationClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = workstations.DeleteWorkstationClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_workstation_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_workstation_cluster_rest_bad_request(transport: str='rest', request_type=workstations.DeleteWorkstationClusterRequest):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_workstation_cluster(request)

def test_delete_workstation_cluster_rest_flattened():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_workstation_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/workstationClusters/*}' % client.transport._host, args[1])

def test_delete_workstation_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_workstation_cluster(workstations.DeleteWorkstationClusterRequest(), name='name_value')

def test_delete_workstation_cluster_rest_error():
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workstations.GetWorkstationConfigRequest, dict])
def test_get_workstation_config_rest(request_type):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workstations.WorkstationConfig(name='name_value', display_name='display_name_value', uid='uid_value', reconciling=True, etag='etag_value', replica_zones=['replica_zones_value'], degraded=True, enable_audit_agent=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = workstations.WorkstationConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_workstation_config(request)
    assert isinstance(response, workstations.WorkstationConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'
    assert response.replica_zones == ['replica_zones_value']
    assert response.degraded is True
    assert response.enable_audit_agent is True

def test_get_workstation_config_rest_required_fields(request_type=workstations.GetWorkstationConfigRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_workstation_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_workstation_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = workstations.WorkstationConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = workstations.WorkstationConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_workstation_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_workstation_config_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_workstation_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_workstation_config_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WorkstationsRestInterceptor, 'post_get_workstation_config') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_get_workstation_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.GetWorkstationConfigRequest.pb(workstations.GetWorkstationConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = workstations.WorkstationConfig.to_json(workstations.WorkstationConfig())
        request = workstations.GetWorkstationConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = workstations.WorkstationConfig()
        client.get_workstation_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_workstation_config_rest_bad_request(transport: str='rest', request_type=workstations.GetWorkstationConfigRequest):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_workstation_config(request)

def test_get_workstation_config_rest_flattened():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workstations.WorkstationConfig()
        sample_request = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = workstations.WorkstationConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_workstation_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/workstationClusters/*/workstationConfigs/*}' % client.transport._host, args[1])

def test_get_workstation_config_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_workstation_config(workstations.GetWorkstationConfigRequest(), name='name_value')

def test_get_workstation_config_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workstations.ListWorkstationConfigsRequest, dict])
def test_list_workstation_configs_rest(request_type):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workstations.ListWorkstationConfigsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = workstations.ListWorkstationConfigsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_workstation_configs(request)
    assert isinstance(response, pagers.ListWorkstationConfigsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_workstation_configs_rest_required_fields(request_type=workstations.ListWorkstationConfigsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_workstation_configs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_workstation_configs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = workstations.ListWorkstationConfigsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = workstations.ListWorkstationConfigsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_workstation_configs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_workstation_configs_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_workstation_configs._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_workstation_configs_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WorkstationsRestInterceptor, 'post_list_workstation_configs') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_list_workstation_configs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.ListWorkstationConfigsRequest.pb(workstations.ListWorkstationConfigsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = workstations.ListWorkstationConfigsResponse.to_json(workstations.ListWorkstationConfigsResponse())
        request = workstations.ListWorkstationConfigsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = workstations.ListWorkstationConfigsResponse()
        client.list_workstation_configs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_workstation_configs_rest_bad_request(transport: str='rest', request_type=workstations.ListWorkstationConfigsRequest):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_workstation_configs(request)

def test_list_workstation_configs_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workstations.ListWorkstationConfigsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = workstations.ListWorkstationConfigsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_workstation_configs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*/workstationClusters/*}/workstationConfigs' % client.transport._host, args[1])

def test_list_workstation_configs_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_workstation_configs(workstations.ListWorkstationConfigsRequest(), parent='parent_value')

def test_list_workstation_configs_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (workstations.ListWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig(), workstations.WorkstationConfig()], next_page_token='abc'), workstations.ListWorkstationConfigsResponse(workstation_configs=[], next_page_token='def'), workstations.ListWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig()], next_page_token='ghi'), workstations.ListWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig()]))
        response = response + response
        response = tuple((workstations.ListWorkstationConfigsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3'}
        pager = client.list_workstation_configs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, workstations.WorkstationConfig) for i in results))
        pages = list(client.list_workstation_configs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [workstations.ListUsableWorkstationConfigsRequest, dict])
def test_list_usable_workstation_configs_rest(request_type):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workstations.ListUsableWorkstationConfigsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = workstations.ListUsableWorkstationConfigsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_usable_workstation_configs(request)
    assert isinstance(response, pagers.ListUsableWorkstationConfigsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_usable_workstation_configs_rest_required_fields(request_type=workstations.ListUsableWorkstationConfigsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_usable_workstation_configs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_usable_workstation_configs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = workstations.ListUsableWorkstationConfigsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = workstations.ListUsableWorkstationConfigsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_usable_workstation_configs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_usable_workstation_configs_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_usable_workstation_configs._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_usable_workstation_configs_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WorkstationsRestInterceptor, 'post_list_usable_workstation_configs') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_list_usable_workstation_configs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.ListUsableWorkstationConfigsRequest.pb(workstations.ListUsableWorkstationConfigsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = workstations.ListUsableWorkstationConfigsResponse.to_json(workstations.ListUsableWorkstationConfigsResponse())
        request = workstations.ListUsableWorkstationConfigsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = workstations.ListUsableWorkstationConfigsResponse()
        client.list_usable_workstation_configs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_usable_workstation_configs_rest_bad_request(transport: str='rest', request_type=workstations.ListUsableWorkstationConfigsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_usable_workstation_configs(request)

def test_list_usable_workstation_configs_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workstations.ListUsableWorkstationConfigsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = workstations.ListUsableWorkstationConfigsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_usable_workstation_configs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*/workstationClusters/*}/workstationConfigs:listUsable' % client.transport._host, args[1])

def test_list_usable_workstation_configs_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_usable_workstation_configs(workstations.ListUsableWorkstationConfigsRequest(), parent='parent_value')

def test_list_usable_workstation_configs_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig(), workstations.WorkstationConfig()], next_page_token='abc'), workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[], next_page_token='def'), workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig()], next_page_token='ghi'), workstations.ListUsableWorkstationConfigsResponse(workstation_configs=[workstations.WorkstationConfig(), workstations.WorkstationConfig()]))
        response = response + response
        response = tuple((workstations.ListUsableWorkstationConfigsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3'}
        pager = client.list_usable_workstation_configs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, workstations.WorkstationConfig) for i in results))
        pages = list(client.list_usable_workstation_configs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [workstations.CreateWorkstationConfigRequest, dict])
def test_create_workstation_config_rest(request_type):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3'}
    request_init['workstation_config'] = {'name': 'name_value', 'display_name': 'display_name_value', 'uid': 'uid_value', 'reconciling': True, 'annotations': {}, 'labels': {}, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'etag': 'etag_value', 'idle_timeout': {'seconds': 751, 'nanos': 543}, 'running_timeout': {}, 'host': {'gce_instance': {'machine_type': 'machine_type_value', 'service_account': 'service_account_value', 'service_account_scopes': ['service_account_scopes_value1', 'service_account_scopes_value2'], 'tags': ['tags_value1', 'tags_value2'], 'pool_size': 980, 'pooled_instances': 1706, 'disable_public_ip_addresses': True, 'enable_nested_virtualization': True, 'shielded_instance_config': {'enable_secure_boot': True, 'enable_vtpm': True, 'enable_integrity_monitoring': True}, 'confidential_instance_config': {'enable_confidential_compute': True}, 'boot_disk_size_gb': 1792, 'accelerators': [{'type_': 'type__value', 'count': 553}]}}, 'persistent_directories': [{'gce_pd': {'size_gb': 739, 'fs_type': 'fs_type_value', 'disk_type': 'disk_type_value', 'source_snapshot': 'source_snapshot_value', 'reclaim_policy': 1}, 'mount_path': 'mount_path_value'}], 'ephemeral_directories': [{'gce_pd': {'disk_type': 'disk_type_value', 'source_snapshot': 'source_snapshot_value', 'source_image': 'source_image_value', 'read_only': True}, 'mount_path': 'mount_path_value'}], 'container': {'image': 'image_value', 'command': ['command_value1', 'command_value2'], 'args': ['args_value1', 'args_value2'], 'env': {}, 'working_dir': 'working_dir_value', 'run_as_user': 1190}, 'encryption_key': {'kms_key': 'kms_key_value', 'kms_key_service_account': 'kms_key_service_account_value'}, 'readiness_checks': [{'path': 'path_value', 'port': 453}], 'replica_zones': ['replica_zones_value1', 'replica_zones_value2'], 'degraded': True, 'conditions': [{'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}], 'enable_audit_agent': True}
    test_field = workstations.CreateWorkstationConfigRequest.meta.fields['workstation_config']

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
    for (field, value) in request_init['workstation_config'].items():
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
                for i in range(0, len(request_init['workstation_config'][field])):
                    del request_init['workstation_config'][field][i][subfield]
            else:
                del request_init['workstation_config'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_workstation_config(request)
    assert response.operation.name == 'operations/spam'

def test_create_workstation_config_rest_required_fields(request_type=workstations.CreateWorkstationConfigRequest):
    if False:
        return 10
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['workstation_config_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'workstationConfigId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_workstation_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'workstationConfigId' in jsonified_request
    assert jsonified_request['workstationConfigId'] == request_init['workstation_config_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['workstationConfigId'] = 'workstation_config_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_workstation_config._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('validate_only', 'workstation_config_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'workstationConfigId' in jsonified_request
    assert jsonified_request['workstationConfigId'] == 'workstation_config_id_value'
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_workstation_config(request)
            expected_params = [('workstationConfigId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_workstation_config_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_workstation_config._get_unset_required_fields({})
    assert set(unset_fields) == set(('validateOnly', 'workstationConfigId')) & set(('parent', 'workstationConfigId', 'workstationConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_workstation_config_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.WorkstationsRestInterceptor, 'post_create_workstation_config') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_create_workstation_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.CreateWorkstationConfigRequest.pb(workstations.CreateWorkstationConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = workstations.CreateWorkstationConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_workstation_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_workstation_config_rest_bad_request(transport: str='rest', request_type=workstations.CreateWorkstationConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_workstation_config(request)

def test_create_workstation_config_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3'}
        mock_args = dict(parent='parent_value', workstation_config=workstations.WorkstationConfig(name='name_value'), workstation_config_id='workstation_config_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_workstation_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*/workstationClusters/*}/workstationConfigs' % client.transport._host, args[1])

def test_create_workstation_config_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_workstation_config(workstations.CreateWorkstationConfigRequest(), parent='parent_value', workstation_config=workstations.WorkstationConfig(name='name_value'), workstation_config_id='workstation_config_id_value')

def test_create_workstation_config_rest_error():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workstations.UpdateWorkstationConfigRequest, dict])
def test_update_workstation_config_rest(request_type):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'workstation_config': {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}}
    request_init['workstation_config'] = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4', 'display_name': 'display_name_value', 'uid': 'uid_value', 'reconciling': True, 'annotations': {}, 'labels': {}, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'etag': 'etag_value', 'idle_timeout': {'seconds': 751, 'nanos': 543}, 'running_timeout': {}, 'host': {'gce_instance': {'machine_type': 'machine_type_value', 'service_account': 'service_account_value', 'service_account_scopes': ['service_account_scopes_value1', 'service_account_scopes_value2'], 'tags': ['tags_value1', 'tags_value2'], 'pool_size': 980, 'pooled_instances': 1706, 'disable_public_ip_addresses': True, 'enable_nested_virtualization': True, 'shielded_instance_config': {'enable_secure_boot': True, 'enable_vtpm': True, 'enable_integrity_monitoring': True}, 'confidential_instance_config': {'enable_confidential_compute': True}, 'boot_disk_size_gb': 1792, 'accelerators': [{'type_': 'type__value', 'count': 553}]}}, 'persistent_directories': [{'gce_pd': {'size_gb': 739, 'fs_type': 'fs_type_value', 'disk_type': 'disk_type_value', 'source_snapshot': 'source_snapshot_value', 'reclaim_policy': 1}, 'mount_path': 'mount_path_value'}], 'ephemeral_directories': [{'gce_pd': {'disk_type': 'disk_type_value', 'source_snapshot': 'source_snapshot_value', 'source_image': 'source_image_value', 'read_only': True}, 'mount_path': 'mount_path_value'}], 'container': {'image': 'image_value', 'command': ['command_value1', 'command_value2'], 'args': ['args_value1', 'args_value2'], 'env': {}, 'working_dir': 'working_dir_value', 'run_as_user': 1190}, 'encryption_key': {'kms_key': 'kms_key_value', 'kms_key_service_account': 'kms_key_service_account_value'}, 'readiness_checks': [{'path': 'path_value', 'port': 453}], 'replica_zones': ['replica_zones_value1', 'replica_zones_value2'], 'degraded': True, 'conditions': [{'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}], 'enable_audit_agent': True}
    test_field = workstations.UpdateWorkstationConfigRequest.meta.fields['workstation_config']

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
    for (field, value) in request_init['workstation_config'].items():
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
                for i in range(0, len(request_init['workstation_config'][field])):
                    del request_init['workstation_config'][field][i][subfield]
            else:
                del request_init['workstation_config'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_workstation_config(request)
    assert response.operation.name == 'operations/spam'

def test_update_workstation_config_rest_required_fields(request_type=workstations.UpdateWorkstationConfigRequest):
    if False:
        print('Hello World!')
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_workstation_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_workstation_config._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_workstation_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_workstation_config_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_workstation_config._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'updateMask', 'validateOnly')) & set(('workstationConfig', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_workstation_config_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.WorkstationsRestInterceptor, 'post_update_workstation_config') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_update_workstation_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.UpdateWorkstationConfigRequest.pb(workstations.UpdateWorkstationConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = workstations.UpdateWorkstationConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_workstation_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_workstation_config_rest_bad_request(transport: str='rest', request_type=workstations.UpdateWorkstationConfigRequest):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'workstation_config': {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_workstation_config(request)

def test_update_workstation_config_rest_flattened():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'workstation_config': {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}}
        mock_args = dict(workstation_config=workstations.WorkstationConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_workstation_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{workstation_config.name=projects/*/locations/*/workstationClusters/*/workstationConfigs/*}' % client.transport._host, args[1])

def test_update_workstation_config_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_workstation_config(workstations.UpdateWorkstationConfigRequest(), workstation_config=workstations.WorkstationConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_workstation_config_rest_error():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workstations.DeleteWorkstationConfigRequest, dict])
def test_delete_workstation_config_rest(request_type):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_workstation_config(request)
    assert response.operation.name == 'operations/spam'

def test_delete_workstation_config_rest_required_fields(request_type=workstations.DeleteWorkstationConfigRequest):
    if False:
        return 10
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_workstation_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_workstation_config._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('etag', 'force', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_workstation_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_workstation_config_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_workstation_config._get_unset_required_fields({})
    assert set(unset_fields) == set(('etag', 'force', 'validateOnly')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_workstation_config_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.WorkstationsRestInterceptor, 'post_delete_workstation_config') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_delete_workstation_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.DeleteWorkstationConfigRequest.pb(workstations.DeleteWorkstationConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = workstations.DeleteWorkstationConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_workstation_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_workstation_config_rest_bad_request(transport: str='rest', request_type=workstations.DeleteWorkstationConfigRequest):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_workstation_config(request)

def test_delete_workstation_config_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_workstation_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/workstationClusters/*/workstationConfigs/*}' % client.transport._host, args[1])

def test_delete_workstation_config_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_workstation_config(workstations.DeleteWorkstationConfigRequest(), name='name_value')

def test_delete_workstation_config_rest_error():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workstations.GetWorkstationRequest, dict])
def test_get_workstation_rest(request_type):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workstations.Workstation(name='name_value', display_name='display_name_value', uid='uid_value', reconciling=True, etag='etag_value', state=workstations.Workstation.State.STATE_STARTING, host='host_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = workstations.Workstation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_workstation(request)
    assert isinstance(response, workstations.Workstation)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'
    assert response.state == workstations.Workstation.State.STATE_STARTING
    assert response.host == 'host_value'

def test_get_workstation_rest_required_fields(request_type=workstations.GetWorkstationRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_workstation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_workstation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = workstations.Workstation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = workstations.Workstation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_workstation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_workstation_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_workstation._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_workstation_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WorkstationsRestInterceptor, 'post_get_workstation') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_get_workstation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.GetWorkstationRequest.pb(workstations.GetWorkstationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = workstations.Workstation.to_json(workstations.Workstation())
        request = workstations.GetWorkstationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = workstations.Workstation()
        client.get_workstation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_workstation_rest_bad_request(transport: str='rest', request_type=workstations.GetWorkstationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_workstation(request)

def test_get_workstation_rest_flattened():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workstations.Workstation()
        sample_request = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = workstations.Workstation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_workstation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/workstationClusters/*/workstationConfigs/*/workstations/*}' % client.transport._host, args[1])

def test_get_workstation_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_workstation(workstations.GetWorkstationRequest(), name='name_value')

def test_get_workstation_rest_error():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workstations.ListWorkstationsRequest, dict])
def test_list_workstations_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workstations.ListWorkstationsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = workstations.ListWorkstationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_workstations(request)
    assert isinstance(response, pagers.ListWorkstationsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_workstations_rest_required_fields(request_type=workstations.ListWorkstationsRequest):
    if False:
        return 10
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_workstations._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_workstations._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = workstations.ListWorkstationsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = workstations.ListWorkstationsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_workstations(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_workstations_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_workstations._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_workstations_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WorkstationsRestInterceptor, 'post_list_workstations') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_list_workstations') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.ListWorkstationsRequest.pb(workstations.ListWorkstationsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = workstations.ListWorkstationsResponse.to_json(workstations.ListWorkstationsResponse())
        request = workstations.ListWorkstationsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = workstations.ListWorkstationsResponse()
        client.list_workstations(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_workstations_rest_bad_request(transport: str='rest', request_type=workstations.ListWorkstationsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_workstations(request)

def test_list_workstations_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workstations.ListWorkstationsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = workstations.ListWorkstationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_workstations(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*/workstationClusters/*/workstationConfigs/*}/workstations' % client.transport._host, args[1])

def test_list_workstations_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_workstations(workstations.ListWorkstationsRequest(), parent='parent_value')

def test_list_workstations_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (workstations.ListWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation(), workstations.Workstation()], next_page_token='abc'), workstations.ListWorkstationsResponse(workstations=[], next_page_token='def'), workstations.ListWorkstationsResponse(workstations=[workstations.Workstation()], next_page_token='ghi'), workstations.ListWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation()]))
        response = response + response
        response = tuple((workstations.ListWorkstationsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
        pager = client.list_workstations(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, workstations.Workstation) for i in results))
        pages = list(client.list_workstations(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [workstations.ListUsableWorkstationsRequest, dict])
def test_list_usable_workstations_rest(request_type):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workstations.ListUsableWorkstationsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = workstations.ListUsableWorkstationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_usable_workstations(request)
    assert isinstance(response, pagers.ListUsableWorkstationsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_usable_workstations_rest_required_fields(request_type=workstations.ListUsableWorkstationsRequest):
    if False:
        return 10
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_usable_workstations._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_usable_workstations._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = workstations.ListUsableWorkstationsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = workstations.ListUsableWorkstationsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_usable_workstations(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_usable_workstations_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_usable_workstations._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_usable_workstations_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WorkstationsRestInterceptor, 'post_list_usable_workstations') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_list_usable_workstations') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.ListUsableWorkstationsRequest.pb(workstations.ListUsableWorkstationsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = workstations.ListUsableWorkstationsResponse.to_json(workstations.ListUsableWorkstationsResponse())
        request = workstations.ListUsableWorkstationsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = workstations.ListUsableWorkstationsResponse()
        client.list_usable_workstations(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_usable_workstations_rest_bad_request(transport: str='rest', request_type=workstations.ListUsableWorkstationsRequest):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_usable_workstations(request)

def test_list_usable_workstations_rest_flattened():
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workstations.ListUsableWorkstationsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = workstations.ListUsableWorkstationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_usable_workstations(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*/workstationClusters/*/workstationConfigs/*}/workstations:listUsable' % client.transport._host, args[1])

def test_list_usable_workstations_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_usable_workstations(workstations.ListUsableWorkstationsRequest(), parent='parent_value')

def test_list_usable_workstations_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (workstations.ListUsableWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation(), workstations.Workstation()], next_page_token='abc'), workstations.ListUsableWorkstationsResponse(workstations=[], next_page_token='def'), workstations.ListUsableWorkstationsResponse(workstations=[workstations.Workstation()], next_page_token='ghi'), workstations.ListUsableWorkstationsResponse(workstations=[workstations.Workstation(), workstations.Workstation()]))
        response = response + response
        response = tuple((workstations.ListUsableWorkstationsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
        pager = client.list_usable_workstations(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, workstations.Workstation) for i in results))
        pages = list(client.list_usable_workstations(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [workstations.CreateWorkstationRequest, dict])
def test_create_workstation_rest(request_type):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
    request_init['workstation'] = {'name': 'name_value', 'display_name': 'display_name_value', 'uid': 'uid_value', 'reconciling': True, 'annotations': {}, 'labels': {}, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'start_time': {}, 'delete_time': {}, 'etag': 'etag_value', 'state': 1, 'host': 'host_value', 'env': {}}
    test_field = workstations.CreateWorkstationRequest.meta.fields['workstation']

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
    for (field, value) in request_init['workstation'].items():
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
                for i in range(0, len(request_init['workstation'][field])):
                    del request_init['workstation'][field][i][subfield]
            else:
                del request_init['workstation'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_workstation(request)
    assert response.operation.name == 'operations/spam'

def test_create_workstation_rest_required_fields(request_type=workstations.CreateWorkstationRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['workstation_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'workstationId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_workstation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'workstationId' in jsonified_request
    assert jsonified_request['workstationId'] == request_init['workstation_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['workstationId'] = 'workstation_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_workstation._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('validate_only', 'workstation_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'workstationId' in jsonified_request
    assert jsonified_request['workstationId'] == 'workstation_id_value'
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_workstation(request)
            expected_params = [('workstationId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_workstation_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_workstation._get_unset_required_fields({})
    assert set(unset_fields) == set(('validateOnly', 'workstationId')) & set(('parent', 'workstationId', 'workstation'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_workstation_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.WorkstationsRestInterceptor, 'post_create_workstation') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_create_workstation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.CreateWorkstationRequest.pb(workstations.CreateWorkstationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = workstations.CreateWorkstationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_workstation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_workstation_rest_bad_request(transport: str='rest', request_type=workstations.CreateWorkstationRequest):
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_workstation(request)

def test_create_workstation_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
        mock_args = dict(parent='parent_value', workstation=workstations.Workstation(name='name_value'), workstation_id='workstation_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_workstation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*/workstationClusters/*/workstationConfigs/*}/workstations' % client.transport._host, args[1])

def test_create_workstation_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_workstation(workstations.CreateWorkstationRequest(), parent='parent_value', workstation=workstations.Workstation(name='name_value'), workstation_id='workstation_id_value')

def test_create_workstation_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workstations.UpdateWorkstationRequest, dict])
def test_update_workstation_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'workstation': {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5'}}
    request_init['workstation'] = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5', 'display_name': 'display_name_value', 'uid': 'uid_value', 'reconciling': True, 'annotations': {}, 'labels': {}, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'start_time': {}, 'delete_time': {}, 'etag': 'etag_value', 'state': 1, 'host': 'host_value', 'env': {}}
    test_field = workstations.UpdateWorkstationRequest.meta.fields['workstation']

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
    for (field, value) in request_init['workstation'].items():
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
                for i in range(0, len(request_init['workstation'][field])):
                    del request_init['workstation'][field][i][subfield]
            else:
                del request_init['workstation'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_workstation(request)
    assert response.operation.name == 'operations/spam'

def test_update_workstation_rest_required_fields(request_type=workstations.UpdateWorkstationRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_workstation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_workstation._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_workstation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_workstation_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_workstation._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'updateMask', 'validateOnly')) & set(('workstation', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_workstation_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.WorkstationsRestInterceptor, 'post_update_workstation') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_update_workstation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.UpdateWorkstationRequest.pb(workstations.UpdateWorkstationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = workstations.UpdateWorkstationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_workstation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_workstation_rest_bad_request(transport: str='rest', request_type=workstations.UpdateWorkstationRequest):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'workstation': {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_workstation(request)

def test_update_workstation_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'workstation': {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5'}}
        mock_args = dict(workstation=workstations.Workstation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_workstation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{workstation.name=projects/*/locations/*/workstationClusters/*/workstationConfigs/*/workstations/*}' % client.transport._host, args[1])

def test_update_workstation_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_workstation(workstations.UpdateWorkstationRequest(), workstation=workstations.Workstation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_workstation_rest_error():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workstations.DeleteWorkstationRequest, dict])
def test_delete_workstation_rest(request_type):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_workstation(request)
    assert response.operation.name == 'operations/spam'

def test_delete_workstation_rest_required_fields(request_type=workstations.DeleteWorkstationRequest):
    if False:
        print('Hello World!')
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_workstation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_workstation._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('etag', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_workstation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_workstation_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_workstation._get_unset_required_fields({})
    assert set(unset_fields) == set(('etag', 'validateOnly')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_workstation_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.WorkstationsRestInterceptor, 'post_delete_workstation') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_delete_workstation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.DeleteWorkstationRequest.pb(workstations.DeleteWorkstationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = workstations.DeleteWorkstationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_workstation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_workstation_rest_bad_request(transport: str='rest', request_type=workstations.DeleteWorkstationRequest):
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_workstation(request)

def test_delete_workstation_rest_flattened():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_workstation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/workstationClusters/*/workstationConfigs/*/workstations/*}' % client.transport._host, args[1])

def test_delete_workstation_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_workstation(workstations.DeleteWorkstationRequest(), name='name_value')

def test_delete_workstation_rest_error():
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workstations.StartWorkstationRequest, dict])
def test_start_workstation_rest(request_type):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.start_workstation(request)
    assert response.operation.name == 'operations/spam'

def test_start_workstation_rest_required_fields(request_type=workstations.StartWorkstationRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).start_workstation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).start_workstation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.start_workstation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_start_workstation_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.start_workstation._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_start_workstation_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.WorkstationsRestInterceptor, 'post_start_workstation') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_start_workstation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.StartWorkstationRequest.pb(workstations.StartWorkstationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = workstations.StartWorkstationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.start_workstation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_start_workstation_rest_bad_request(transport: str='rest', request_type=workstations.StartWorkstationRequest):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.start_workstation(request)

def test_start_workstation_rest_flattened():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.start_workstation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/workstationClusters/*/workstationConfigs/*/workstations/*}:start' % client.transport._host, args[1])

def test_start_workstation_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.start_workstation(workstations.StartWorkstationRequest(), name='name_value')

def test_start_workstation_rest_error():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workstations.StopWorkstationRequest, dict])
def test_stop_workstation_rest(request_type):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.stop_workstation(request)
    assert response.operation.name == 'operations/spam'

def test_stop_workstation_rest_required_fields(request_type=workstations.StopWorkstationRequest):
    if False:
        return 10
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).stop_workstation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).stop_workstation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.stop_workstation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_stop_workstation_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.stop_workstation._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_stop_workstation_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.WorkstationsRestInterceptor, 'post_stop_workstation') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_stop_workstation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.StopWorkstationRequest.pb(workstations.StopWorkstationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = workstations.StopWorkstationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.stop_workstation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_stop_workstation_rest_bad_request(transport: str='rest', request_type=workstations.StopWorkstationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.stop_workstation(request)

def test_stop_workstation_rest_flattened():
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.stop_workstation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/workstationClusters/*/workstationConfigs/*/workstations/*}:stop' % client.transport._host, args[1])

def test_stop_workstation_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.stop_workstation(workstations.StopWorkstationRequest(), name='name_value')

def test_stop_workstation_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workstations.GenerateAccessTokenRequest, dict])
def test_generate_access_token_rest(request_type):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'workstation': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workstations.GenerateAccessTokenResponse(access_token='access_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = workstations.GenerateAccessTokenResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.generate_access_token(request)
    assert isinstance(response, workstations.GenerateAccessTokenResponse)
    assert response.access_token == 'access_token_value'

def test_generate_access_token_rest_required_fields(request_type=workstations.GenerateAccessTokenRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.WorkstationsRestTransport
    request_init = {}
    request_init['workstation'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_access_token._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['workstation'] = 'workstation_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_access_token._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'workstation' in jsonified_request
    assert jsonified_request['workstation'] == 'workstation_value'
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = workstations.GenerateAccessTokenResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = workstations.GenerateAccessTokenResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.generate_access_token(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_generate_access_token_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.generate_access_token._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('workstation',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_generate_access_token_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.WorkstationsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkstationsRestInterceptor())
    client = WorkstationsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WorkstationsRestInterceptor, 'post_generate_access_token') as post, mock.patch.object(transports.WorkstationsRestInterceptor, 'pre_generate_access_token') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workstations.GenerateAccessTokenRequest.pb(workstations.GenerateAccessTokenRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = workstations.GenerateAccessTokenResponse.to_json(workstations.GenerateAccessTokenResponse())
        request = workstations.GenerateAccessTokenRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = workstations.GenerateAccessTokenResponse()
        client.generate_access_token(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_generate_access_token_rest_bad_request(transport: str='rest', request_type=workstations.GenerateAccessTokenRequest):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'workstation': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.generate_access_token(request)

def test_generate_access_token_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workstations.GenerateAccessTokenResponse()
        sample_request = {'workstation': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4/workstations/sample5'}
        mock_args = dict(workstation='workstation_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = workstations.GenerateAccessTokenResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.generate_access_token(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{workstation=projects/*/locations/*/workstationClusters/*/workstationConfigs/*/workstations/*}:generateAccessToken' % client.transport._host, args[1])

def test_generate_access_token_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.generate_access_token(workstations.GenerateAccessTokenRequest(), workstation='workstation_value')

def test_generate_access_token_rest_error():
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        print('Hello World!')
    transport = transports.WorkstationsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.WorkstationsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = WorkstationsClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.WorkstationsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = WorkstationsClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = WorkstationsClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.WorkstationsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = WorkstationsClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.WorkstationsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = WorkstationsClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.WorkstationsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.WorkstationsGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.WorkstationsGrpcTransport, transports.WorkstationsGrpcAsyncIOTransport, transports.WorkstationsRestTransport])
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
        while True:
            i = 10
    transport = WorkstationsClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.WorkstationsGrpcTransport)

def test_workstations_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.WorkstationsTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_workstations_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.workstations_v1beta.services.workstations.transports.WorkstationsTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.WorkstationsTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('get_workstation_cluster', 'list_workstation_clusters', 'create_workstation_cluster', 'update_workstation_cluster', 'delete_workstation_cluster', 'get_workstation_config', 'list_workstation_configs', 'list_usable_workstation_configs', 'create_workstation_config', 'update_workstation_config', 'delete_workstation_config', 'get_workstation', 'list_workstations', 'list_usable_workstations', 'create_workstation', 'update_workstation', 'delete_workstation', 'start_workstation', 'stop_workstation', 'generate_access_token', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_workstations_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.workstations_v1beta.services.workstations.transports.WorkstationsTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.WorkstationsTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_workstations_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.workstations_v1beta.services.workstations.transports.WorkstationsTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.WorkstationsTransport()
        adc.assert_called_once()

def test_workstations_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        WorkstationsClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.WorkstationsGrpcTransport, transports.WorkstationsGrpcAsyncIOTransport])
def test_workstations_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.WorkstationsGrpcTransport, transports.WorkstationsGrpcAsyncIOTransport, transports.WorkstationsRestTransport])
def test_workstations_transport_auth_gdch_credentials(transport_class):
    if False:
        print('Hello World!')
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.WorkstationsGrpcTransport, grpc_helpers), (transports.WorkstationsGrpcAsyncIOTransport, grpc_helpers_async)])
def test_workstations_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('workstations.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='workstations.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.WorkstationsGrpcTransport, transports.WorkstationsGrpcAsyncIOTransport])
def test_workstations_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_workstations_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.WorkstationsRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_workstations_rest_lro_client():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_workstations_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='workstations.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('workstations.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://workstations.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_workstations_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='workstations.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('workstations.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://workstations.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_workstations_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = WorkstationsClient(credentials=creds1, transport=transport_name)
    client2 = WorkstationsClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.get_workstation_cluster._session
    session2 = client2.transport.get_workstation_cluster._session
    assert session1 != session2
    session1 = client1.transport.list_workstation_clusters._session
    session2 = client2.transport.list_workstation_clusters._session
    assert session1 != session2
    session1 = client1.transport.create_workstation_cluster._session
    session2 = client2.transport.create_workstation_cluster._session
    assert session1 != session2
    session1 = client1.transport.update_workstation_cluster._session
    session2 = client2.transport.update_workstation_cluster._session
    assert session1 != session2
    session1 = client1.transport.delete_workstation_cluster._session
    session2 = client2.transport.delete_workstation_cluster._session
    assert session1 != session2
    session1 = client1.transport.get_workstation_config._session
    session2 = client2.transport.get_workstation_config._session
    assert session1 != session2
    session1 = client1.transport.list_workstation_configs._session
    session2 = client2.transport.list_workstation_configs._session
    assert session1 != session2
    session1 = client1.transport.list_usable_workstation_configs._session
    session2 = client2.transport.list_usable_workstation_configs._session
    assert session1 != session2
    session1 = client1.transport.create_workstation_config._session
    session2 = client2.transport.create_workstation_config._session
    assert session1 != session2
    session1 = client1.transport.update_workstation_config._session
    session2 = client2.transport.update_workstation_config._session
    assert session1 != session2
    session1 = client1.transport.delete_workstation_config._session
    session2 = client2.transport.delete_workstation_config._session
    assert session1 != session2
    session1 = client1.transport.get_workstation._session
    session2 = client2.transport.get_workstation._session
    assert session1 != session2
    session1 = client1.transport.list_workstations._session
    session2 = client2.transport.list_workstations._session
    assert session1 != session2
    session1 = client1.transport.list_usable_workstations._session
    session2 = client2.transport.list_usable_workstations._session
    assert session1 != session2
    session1 = client1.transport.create_workstation._session
    session2 = client2.transport.create_workstation._session
    assert session1 != session2
    session1 = client1.transport.update_workstation._session
    session2 = client2.transport.update_workstation._session
    assert session1 != session2
    session1 = client1.transport.delete_workstation._session
    session2 = client2.transport.delete_workstation._session
    assert session1 != session2
    session1 = client1.transport.start_workstation._session
    session2 = client2.transport.start_workstation._session
    assert session1 != session2
    session1 = client1.transport.stop_workstation._session
    session2 = client2.transport.stop_workstation._session
    assert session1 != session2
    session1 = client1.transport.generate_access_token._session
    session2 = client2.transport.generate_access_token._session
    assert session1 != session2

def test_workstations_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.WorkstationsGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_workstations_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.WorkstationsGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.WorkstationsGrpcTransport, transports.WorkstationsGrpcAsyncIOTransport])
def test_workstations_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.WorkstationsGrpcTransport, transports.WorkstationsGrpcAsyncIOTransport])
def test_workstations_transport_channel_mtls_with_adc(transport_class):
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

def test_workstations_grpc_lro_client():
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_workstations_grpc_lro_async_client():
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_workstation_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    workstation_cluster = 'whelk'
    workstation_config = 'octopus'
    workstation = 'oyster'
    expected = 'projects/{project}/locations/{location}/workstationClusters/{workstation_cluster}/workstationConfigs/{workstation_config}/workstations/{workstation}'.format(project=project, location=location, workstation_cluster=workstation_cluster, workstation_config=workstation_config, workstation=workstation)
    actual = WorkstationsClient.workstation_path(project, location, workstation_cluster, workstation_config, workstation)
    assert expected == actual

def test_parse_workstation_path():
    if False:
        return 10
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'workstation_cluster': 'mussel', 'workstation_config': 'winkle', 'workstation': 'nautilus'}
    path = WorkstationsClient.workstation_path(**expected)
    actual = WorkstationsClient.parse_workstation_path(path)
    assert expected == actual

def test_workstation_cluster_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'scallop'
    location = 'abalone'
    workstation_cluster = 'squid'
    expected = 'projects/{project}/locations/{location}/workstationClusters/{workstation_cluster}'.format(project=project, location=location, workstation_cluster=workstation_cluster)
    actual = WorkstationsClient.workstation_cluster_path(project, location, workstation_cluster)
    assert expected == actual

def test_parse_workstation_cluster_path():
    if False:
        print('Hello World!')
    expected = {'project': 'clam', 'location': 'whelk', 'workstation_cluster': 'octopus'}
    path = WorkstationsClient.workstation_cluster_path(**expected)
    actual = WorkstationsClient.parse_workstation_cluster_path(path)
    assert expected == actual

def test_workstation_config_path():
    if False:
        print('Hello World!')
    project = 'oyster'
    location = 'nudibranch'
    workstation_cluster = 'cuttlefish'
    workstation_config = 'mussel'
    expected = 'projects/{project}/locations/{location}/workstationClusters/{workstation_cluster}/workstationConfigs/{workstation_config}'.format(project=project, location=location, workstation_cluster=workstation_cluster, workstation_config=workstation_config)
    actual = WorkstationsClient.workstation_config_path(project, location, workstation_cluster, workstation_config)
    assert expected == actual

def test_parse_workstation_config_path():
    if False:
        return 10
    expected = {'project': 'winkle', 'location': 'nautilus', 'workstation_cluster': 'scallop', 'workstation_config': 'abalone'}
    path = WorkstationsClient.workstation_config_path(**expected)
    actual = WorkstationsClient.parse_workstation_config_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = WorkstationsClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'clam'}
    path = WorkstationsClient.common_billing_account_path(**expected)
    actual = WorkstationsClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = WorkstationsClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'octopus'}
    path = WorkstationsClient.common_folder_path(**expected)
    actual = WorkstationsClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = WorkstationsClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'nudibranch'}
    path = WorkstationsClient.common_organization_path(**expected)
    actual = WorkstationsClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = WorkstationsClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'mussel'}
    path = WorkstationsClient.common_project_path(**expected)
    actual = WorkstationsClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = WorkstationsClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = WorkstationsClient.common_location_path(**expected)
    actual = WorkstationsClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.WorkstationsTransport, '_prep_wrapped_messages') as prep:
        client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.WorkstationsTransport, '_prep_wrapped_messages') as prep:
        transport_class = WorkstationsClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
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
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}, request)
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
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
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
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/workstationClusters/sample3/workstationConfigs/sample4'}
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
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = WorkstationsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        return 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = WorkstationsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(WorkstationsClient, transports.WorkstationsGrpcTransport), (WorkstationsAsyncClient, transports.WorkstationsGrpcAsyncIOTransport)])
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