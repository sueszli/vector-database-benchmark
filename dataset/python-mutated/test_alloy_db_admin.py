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
from google.protobuf import duration_pb2
from google.protobuf import empty_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.type import dayofweek_pb2
from google.type import timeofday_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.alloydb_v1beta.services.alloy_db_admin import AlloyDBAdminAsyncClient, AlloyDBAdminClient, pagers, transports
from google.cloud.alloydb_v1beta.types import resources, service

def client_cert_source_callback():
    if False:
        print('Hello World!')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        i = 10
        return i + 15
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
    assert AlloyDBAdminClient._get_default_mtls_endpoint(None) is None
    assert AlloyDBAdminClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert AlloyDBAdminClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert AlloyDBAdminClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert AlloyDBAdminClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert AlloyDBAdminClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(AlloyDBAdminClient, 'grpc'), (AlloyDBAdminAsyncClient, 'grpc_asyncio'), (AlloyDBAdminClient, 'rest')])
def test_alloy_db_admin_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('alloydb.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://alloydb.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.AlloyDBAdminGrpcTransport, 'grpc'), (transports.AlloyDBAdminGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.AlloyDBAdminRestTransport, 'rest')])
def test_alloy_db_admin_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(AlloyDBAdminClient, 'grpc'), (AlloyDBAdminAsyncClient, 'grpc_asyncio'), (AlloyDBAdminClient, 'rest')])
def test_alloy_db_admin_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('alloydb.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://alloydb.googleapis.com')

def test_alloy_db_admin_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = AlloyDBAdminClient.get_transport_class()
    available_transports = [transports.AlloyDBAdminGrpcTransport, transports.AlloyDBAdminRestTransport]
    assert transport in available_transports
    transport = AlloyDBAdminClient.get_transport_class('grpc')
    assert transport == transports.AlloyDBAdminGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AlloyDBAdminClient, transports.AlloyDBAdminGrpcTransport, 'grpc'), (AlloyDBAdminAsyncClient, transports.AlloyDBAdminGrpcAsyncIOTransport, 'grpc_asyncio'), (AlloyDBAdminClient, transports.AlloyDBAdminRestTransport, 'rest')])
@mock.patch.object(AlloyDBAdminClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AlloyDBAdminClient))
@mock.patch.object(AlloyDBAdminAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AlloyDBAdminAsyncClient))
def test_alloy_db_admin_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(AlloyDBAdminClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(AlloyDBAdminClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(AlloyDBAdminClient, transports.AlloyDBAdminGrpcTransport, 'grpc', 'true'), (AlloyDBAdminAsyncClient, transports.AlloyDBAdminGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (AlloyDBAdminClient, transports.AlloyDBAdminGrpcTransport, 'grpc', 'false'), (AlloyDBAdminAsyncClient, transports.AlloyDBAdminGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (AlloyDBAdminClient, transports.AlloyDBAdminRestTransport, 'rest', 'true'), (AlloyDBAdminClient, transports.AlloyDBAdminRestTransport, 'rest', 'false')])
@mock.patch.object(AlloyDBAdminClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AlloyDBAdminClient))
@mock.patch.object(AlloyDBAdminAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AlloyDBAdminAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_alloy_db_admin_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [AlloyDBAdminClient, AlloyDBAdminAsyncClient])
@mock.patch.object(AlloyDBAdminClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AlloyDBAdminClient))
@mock.patch.object(AlloyDBAdminAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AlloyDBAdminAsyncClient))
def test_alloy_db_admin_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AlloyDBAdminClient, transports.AlloyDBAdminGrpcTransport, 'grpc'), (AlloyDBAdminAsyncClient, transports.AlloyDBAdminGrpcAsyncIOTransport, 'grpc_asyncio'), (AlloyDBAdminClient, transports.AlloyDBAdminRestTransport, 'rest')])
def test_alloy_db_admin_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AlloyDBAdminClient, transports.AlloyDBAdminGrpcTransport, 'grpc', grpc_helpers), (AlloyDBAdminAsyncClient, transports.AlloyDBAdminGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (AlloyDBAdminClient, transports.AlloyDBAdminRestTransport, 'rest', None)])
def test_alloy_db_admin_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_alloy_db_admin_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.alloydb_v1beta.services.alloy_db_admin.transports.AlloyDBAdminGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = AlloyDBAdminClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AlloyDBAdminClient, transports.AlloyDBAdminGrpcTransport, 'grpc', grpc_helpers), (AlloyDBAdminAsyncClient, transports.AlloyDBAdminGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_alloy_db_admin_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('alloydb.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='alloydb.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [service.ListClustersRequest, dict])
def test_list_clusters(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_clusters), '__call__') as call:
        call.return_value = service.ListClustersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_clusters(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListClustersRequest()
    assert isinstance(response, pagers.ListClustersPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_clusters_empty_call():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_clusters), '__call__') as call:
        client.list_clusters()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListClustersRequest()

@pytest.mark.asyncio
async def test_list_clusters_async(transport: str='grpc_asyncio', request_type=service.ListClustersRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_clusters), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListClustersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_clusters(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListClustersRequest()
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListClustersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_clusters), '__call__') as call:
        call.return_value = service.ListClustersResponse()
        client.list_clusters(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_clusters_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListClustersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_clusters), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListClustersResponse())
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_clusters), '__call__') as call:
        call.return_value = service.ListClustersResponse()
        client.list_clusters(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_clusters_flattened_error():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_clusters(service.ListClustersRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_clusters_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_clusters), '__call__') as call:
        call.return_value = service.ListClustersResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListClustersResponse())
        response = await client.list_clusters(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_clusters_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_clusters(service.ListClustersRequest(), parent='parent_value')

def test_list_clusters_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_clusters), '__call__') as call:
        call.side_effect = (service.ListClustersResponse(clusters=[resources.Cluster(), resources.Cluster(), resources.Cluster()], next_page_token='abc'), service.ListClustersResponse(clusters=[], next_page_token='def'), service.ListClustersResponse(clusters=[resources.Cluster()], next_page_token='ghi'), service.ListClustersResponse(clusters=[resources.Cluster(), resources.Cluster()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_clusters(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Cluster) for i in results))

def test_list_clusters_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_clusters), '__call__') as call:
        call.side_effect = (service.ListClustersResponse(clusters=[resources.Cluster(), resources.Cluster(), resources.Cluster()], next_page_token='abc'), service.ListClustersResponse(clusters=[], next_page_token='def'), service.ListClustersResponse(clusters=[resources.Cluster()], next_page_token='ghi'), service.ListClustersResponse(clusters=[resources.Cluster(), resources.Cluster()]), RuntimeError)
        pages = list(client.list_clusters(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_clusters_async_pager():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_clusters), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListClustersResponse(clusters=[resources.Cluster(), resources.Cluster(), resources.Cluster()], next_page_token='abc'), service.ListClustersResponse(clusters=[], next_page_token='def'), service.ListClustersResponse(clusters=[resources.Cluster()], next_page_token='ghi'), service.ListClustersResponse(clusters=[resources.Cluster(), resources.Cluster()]), RuntimeError)
        async_pager = await client.list_clusters(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Cluster) for i in responses))

@pytest.mark.asyncio
async def test_list_clusters_async_pages():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_clusters), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListClustersResponse(clusters=[resources.Cluster(), resources.Cluster(), resources.Cluster()], next_page_token='abc'), service.ListClustersResponse(clusters=[], next_page_token='def'), service.ListClustersResponse(clusters=[resources.Cluster()], next_page_token='ghi'), service.ListClustersResponse(clusters=[resources.Cluster(), resources.Cluster()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_clusters(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetClusterRequest, dict])
def test_get_cluster(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_cluster), '__call__') as call:
        call.return_value = resources.Cluster(name='name_value', display_name='display_name_value', uid='uid_value', state=resources.Cluster.State.READY, cluster_type=resources.Cluster.ClusterType.PRIMARY, database_version=resources.DatabaseVersion.POSTGRES_13, network='network_value', etag='etag_value', reconciling=True)
        response = client.get_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetClusterRequest()
    assert isinstance(response, resources.Cluster)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.state == resources.Cluster.State.READY
    assert response.cluster_type == resources.Cluster.ClusterType.PRIMARY
    assert response.database_version == resources.DatabaseVersion.POSTGRES_13
    assert response.network == 'network_value'
    assert response.etag == 'etag_value'
    assert response.reconciling is True

def test_get_cluster_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_cluster), '__call__') as call:
        client.get_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetClusterRequest()

@pytest.mark.asyncio
async def test_get_cluster_async(transport: str='grpc_asyncio', request_type=service.GetClusterRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Cluster(name='name_value', display_name='display_name_value', uid='uid_value', state=resources.Cluster.State.READY, cluster_type=resources.Cluster.ClusterType.PRIMARY, database_version=resources.DatabaseVersion.POSTGRES_13, network='network_value', etag='etag_value', reconciling=True))
        response = await client.get_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetClusterRequest()
    assert isinstance(response, resources.Cluster)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.state == resources.Cluster.State.READY
    assert response.cluster_type == resources.Cluster.ClusterType.PRIMARY
    assert response.database_version == resources.DatabaseVersion.POSTGRES_13
    assert response.network == 'network_value'
    assert response.etag == 'etag_value'
    assert response.reconciling is True

@pytest.mark.asyncio
async def test_get_cluster_async_from_dict():
    await test_get_cluster_async(request_type=dict)

def test_get_cluster_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_cluster), '__call__') as call:
        call.return_value = resources.Cluster()
        client.get_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_cluster_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Cluster())
        await client.get_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_cluster_flattened():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_cluster), '__call__') as call:
        call.return_value = resources.Cluster()
        client.get_cluster(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_cluster_flattened_error():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_cluster(service.GetClusterRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_cluster_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_cluster), '__call__') as call:
        call.return_value = resources.Cluster()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Cluster())
        response = await client.get_cluster(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_cluster_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_cluster(service.GetClusterRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.CreateClusterRequest, dict])
def test_create_cluster(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateClusterRequest()
    assert isinstance(response, future.Future)

def test_create_cluster_empty_call():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_cluster), '__call__') as call:
        client.create_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateClusterRequest()

@pytest.mark.asyncio
async def test_create_cluster_async(transport: str='grpc_asyncio', request_type=service.CreateClusterRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_cluster_async_from_dict():
    await test_create_cluster_async(request_type=dict)

def test_create_cluster_field_headers():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateClusterRequest()
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
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateClusterRequest()
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
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_cluster(parent='parent_value', cluster=resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value')), cluster_id='cluster_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].cluster
        mock_val = resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value'))
        assert arg == mock_val
        arg = args[0].cluster_id
        mock_val = 'cluster_id_value'
        assert arg == mock_val

def test_create_cluster_flattened_error():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_cluster(service.CreateClusterRequest(), parent='parent_value', cluster=resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value')), cluster_id='cluster_id_value')

@pytest.mark.asyncio
async def test_create_cluster_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_cluster(parent='parent_value', cluster=resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value')), cluster_id='cluster_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].cluster
        mock_val = resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value'))
        assert arg == mock_val
        arg = args[0].cluster_id
        mock_val = 'cluster_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_cluster_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_cluster(service.CreateClusterRequest(), parent='parent_value', cluster=resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value')), cluster_id='cluster_id_value')

@pytest.mark.parametrize('request_type', [service.UpdateClusterRequest, dict])
def test_update_cluster(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateClusterRequest()
    assert isinstance(response, future.Future)

def test_update_cluster_empty_call():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_cluster), '__call__') as call:
        client.update_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateClusterRequest()

@pytest.mark.asyncio
async def test_update_cluster_async(transport: str='grpc_asyncio', request_type=service.UpdateClusterRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_cluster_async_from_dict():
    await test_update_cluster_async(request_type=dict)

def test_update_cluster_field_headers():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateClusterRequest()
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
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateClusterRequest()
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
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_cluster(cluster=resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].cluster
        mock_val = resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value'))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_cluster_flattened_error():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_cluster(service.UpdateClusterRequest(), cluster=resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_cluster_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_cluster(cluster=resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].cluster
        mock_val = resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value'))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_cluster_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_cluster(service.UpdateClusterRequest(), cluster=resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [service.DeleteClusterRequest, dict])
def test_delete_cluster(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteClusterRequest()
    assert isinstance(response, future.Future)

def test_delete_cluster_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_cluster), '__call__') as call:
        client.delete_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteClusterRequest()

@pytest.mark.asyncio
async def test_delete_cluster_async(transport: str='grpc_asyncio', request_type=service.DeleteClusterRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_cluster_async_from_dict():
    await test_delete_cluster_async(request_type=dict)

def test_delete_cluster_field_headers():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteClusterRequest()
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
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteClusterRequest()
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
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_cluster(service.DeleteClusterRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_cluster_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_cluster(service.DeleteClusterRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.PromoteClusterRequest, dict])
def test_promote_cluster(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.promote_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.promote_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.PromoteClusterRequest()
    assert isinstance(response, future.Future)

def test_promote_cluster_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.promote_cluster), '__call__') as call:
        client.promote_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.PromoteClusterRequest()

@pytest.mark.asyncio
async def test_promote_cluster_async(transport: str='grpc_asyncio', request_type=service.PromoteClusterRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.promote_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.promote_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.PromoteClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_promote_cluster_async_from_dict():
    await test_promote_cluster_async(request_type=dict)

def test_promote_cluster_field_headers():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.PromoteClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.promote_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.promote_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_promote_cluster_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.PromoteClusterRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.promote_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.promote_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_promote_cluster_flattened():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.promote_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.promote_cluster(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_promote_cluster_flattened_error():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.promote_cluster(service.PromoteClusterRequest(), name='name_value')

@pytest.mark.asyncio
async def test_promote_cluster_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.promote_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.promote_cluster(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_promote_cluster_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.promote_cluster(service.PromoteClusterRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.RestoreClusterRequest, dict])
def test_restore_cluster(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.restore_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.restore_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RestoreClusterRequest()
    assert isinstance(response, future.Future)

def test_restore_cluster_empty_call():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.restore_cluster), '__call__') as call:
        client.restore_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RestoreClusterRequest()

@pytest.mark.asyncio
async def test_restore_cluster_async(transport: str='grpc_asyncio', request_type=service.RestoreClusterRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.restore_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.restore_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RestoreClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_restore_cluster_async_from_dict():
    await test_restore_cluster_async(request_type=dict)

def test_restore_cluster_field_headers():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.RestoreClusterRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.restore_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.restore_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_restore_cluster_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.RestoreClusterRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.restore_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.restore_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.CreateSecondaryClusterRequest, dict])
def test_create_secondary_cluster(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_secondary_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_secondary_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateSecondaryClusterRequest()
    assert isinstance(response, future.Future)

def test_create_secondary_cluster_empty_call():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_secondary_cluster), '__call__') as call:
        client.create_secondary_cluster()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateSecondaryClusterRequest()

@pytest.mark.asyncio
async def test_create_secondary_cluster_async(transport: str='grpc_asyncio', request_type=service.CreateSecondaryClusterRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_secondary_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_secondary_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateSecondaryClusterRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_secondary_cluster_async_from_dict():
    await test_create_secondary_cluster_async(request_type=dict)

def test_create_secondary_cluster_field_headers():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateSecondaryClusterRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_secondary_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_secondary_cluster(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_secondary_cluster_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateSecondaryClusterRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_secondary_cluster), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_secondary_cluster(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_secondary_cluster_flattened():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_secondary_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_secondary_cluster(parent='parent_value', cluster=resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value')), cluster_id='cluster_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].cluster
        mock_val = resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value'))
        assert arg == mock_val
        arg = args[0].cluster_id
        mock_val = 'cluster_id_value'
        assert arg == mock_val

def test_create_secondary_cluster_flattened_error():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_secondary_cluster(service.CreateSecondaryClusterRequest(), parent='parent_value', cluster=resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value')), cluster_id='cluster_id_value')

@pytest.mark.asyncio
async def test_create_secondary_cluster_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_secondary_cluster), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_secondary_cluster(parent='parent_value', cluster=resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value')), cluster_id='cluster_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].cluster
        mock_val = resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value'))
        assert arg == mock_val
        arg = args[0].cluster_id
        mock_val = 'cluster_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_secondary_cluster_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_secondary_cluster(service.CreateSecondaryClusterRequest(), parent='parent_value', cluster=resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value')), cluster_id='cluster_id_value')

@pytest.mark.parametrize('request_type', [service.ListInstancesRequest, dict])
def test_list_instances(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = service.ListInstancesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_instances(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListInstancesRequest()
    assert isinstance(response, pagers.ListInstancesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_instances_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        client.list_instances()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListInstancesRequest()

@pytest.mark.asyncio
async def test_list_instances_async(transport: str='grpc_asyncio', request_type=service.ListInstancesRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListInstancesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_instances(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListInstancesRequest()
    assert isinstance(response, pagers.ListInstancesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_instances_async_from_dict():
    await test_list_instances_async(request_type=dict)

def test_list_instances_field_headers():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListInstancesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = service.ListInstancesResponse()
        client.list_instances(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_instances_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListInstancesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListInstancesResponse())
        await client.list_instances(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_instances_flattened():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = service.ListInstancesResponse()
        client.list_instances(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_instances_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_instances(service.ListInstancesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_instances_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.return_value = service.ListInstancesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListInstancesResponse())
        response = await client.list_instances(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_instances_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_instances(service.ListInstancesRequest(), parent='parent_value')

def test_list_instances_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.side_effect = (service.ListInstancesResponse(instances=[resources.Instance(), resources.Instance(), resources.Instance()], next_page_token='abc'), service.ListInstancesResponse(instances=[], next_page_token='def'), service.ListInstancesResponse(instances=[resources.Instance()], next_page_token='ghi'), service.ListInstancesResponse(instances=[resources.Instance(), resources.Instance()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_instances(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Instance) for i in results))

def test_list_instances_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_instances), '__call__') as call:
        call.side_effect = (service.ListInstancesResponse(instances=[resources.Instance(), resources.Instance(), resources.Instance()], next_page_token='abc'), service.ListInstancesResponse(instances=[], next_page_token='def'), service.ListInstancesResponse(instances=[resources.Instance()], next_page_token='ghi'), service.ListInstancesResponse(instances=[resources.Instance(), resources.Instance()]), RuntimeError)
        pages = list(client.list_instances(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_instances_async_pager():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_instances), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListInstancesResponse(instances=[resources.Instance(), resources.Instance(), resources.Instance()], next_page_token='abc'), service.ListInstancesResponse(instances=[], next_page_token='def'), service.ListInstancesResponse(instances=[resources.Instance()], next_page_token='ghi'), service.ListInstancesResponse(instances=[resources.Instance(), resources.Instance()]), RuntimeError)
        async_pager = await client.list_instances(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Instance) for i in responses))

@pytest.mark.asyncio
async def test_list_instances_async_pages():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_instances), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListInstancesResponse(instances=[resources.Instance(), resources.Instance(), resources.Instance()], next_page_token='abc'), service.ListInstancesResponse(instances=[], next_page_token='def'), service.ListInstancesResponse(instances=[resources.Instance()], next_page_token='ghi'), service.ListInstancesResponse(instances=[resources.Instance(), resources.Instance()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_instances(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetInstanceRequest, dict])
def test_get_instance(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = resources.Instance(name='name_value', display_name='display_name_value', uid='uid_value', state=resources.Instance.State.READY, instance_type=resources.Instance.InstanceType.PRIMARY, availability_type=resources.Instance.AvailabilityType.ZONAL, gce_zone='gce_zone_value', ip_address='ip_address_value', reconciling=True, etag='etag_value')
        response = client.get_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetInstanceRequest()
    assert isinstance(response, resources.Instance)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.state == resources.Instance.State.READY
    assert response.instance_type == resources.Instance.InstanceType.PRIMARY
    assert response.availability_type == resources.Instance.AvailabilityType.ZONAL
    assert response.gce_zone == 'gce_zone_value'
    assert response.ip_address == 'ip_address_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'

def test_get_instance_empty_call():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        client.get_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetInstanceRequest()

@pytest.mark.asyncio
async def test_get_instance_async(transport: str='grpc_asyncio', request_type=service.GetInstanceRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Instance(name='name_value', display_name='display_name_value', uid='uid_value', state=resources.Instance.State.READY, instance_type=resources.Instance.InstanceType.PRIMARY, availability_type=resources.Instance.AvailabilityType.ZONAL, gce_zone='gce_zone_value', ip_address='ip_address_value', reconciling=True, etag='etag_value'))
        response = await client.get_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetInstanceRequest()
    assert isinstance(response, resources.Instance)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.state == resources.Instance.State.READY
    assert response.instance_type == resources.Instance.InstanceType.PRIMARY
    assert response.availability_type == resources.Instance.AvailabilityType.ZONAL
    assert response.gce_zone == 'gce_zone_value'
    assert response.ip_address == 'ip_address_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_get_instance_async_from_dict():
    await test_get_instance_async(request_type=dict)

def test_get_instance_field_headers():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = resources.Instance()
        client.get_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_instance_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Instance())
        await client.get_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_instance_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = resources.Instance()
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_instance(service.GetInstanceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_instance_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_instance), '__call__') as call:
        call.return_value = resources.Instance()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Instance())
        response = await client.get_instance(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_instance_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_instance(service.GetInstanceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.CreateInstanceRequest, dict])
def test_create_instance(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateInstanceRequest()
    assert isinstance(response, future.Future)

def test_create_instance_empty_call():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        client.create_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateInstanceRequest()

@pytest.mark.asyncio
async def test_create_instance_async(transport: str='grpc_asyncio', request_type=service.CreateInstanceRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_instance_async_from_dict():
    await test_create_instance_async(request_type=dict)

def test_create_instance_field_headers():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateInstanceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_instance_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateInstanceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_instance_flattened():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_instance(parent='parent_value', instance=resources.Instance(name='name_value'), instance_id='instance_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].instance
        mock_val = resources.Instance(name='name_value')
        assert arg == mock_val
        arg = args[0].instance_id
        mock_val = 'instance_id_value'
        assert arg == mock_val

def test_create_instance_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_instance(service.CreateInstanceRequest(), parent='parent_value', instance=resources.Instance(name='name_value'), instance_id='instance_id_value')

@pytest.mark.asyncio
async def test_create_instance_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_instance(parent='parent_value', instance=resources.Instance(name='name_value'), instance_id='instance_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].instance
        mock_val = resources.Instance(name='name_value')
        assert arg == mock_val
        arg = args[0].instance_id
        mock_val = 'instance_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_instance_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_instance(service.CreateInstanceRequest(), parent='parent_value', instance=resources.Instance(name='name_value'), instance_id='instance_id_value')

@pytest.mark.parametrize('request_type', [service.CreateSecondaryInstanceRequest, dict])
def test_create_secondary_instance(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_secondary_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_secondary_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateSecondaryInstanceRequest()
    assert isinstance(response, future.Future)

def test_create_secondary_instance_empty_call():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_secondary_instance), '__call__') as call:
        client.create_secondary_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateSecondaryInstanceRequest()

@pytest.mark.asyncio
async def test_create_secondary_instance_async(transport: str='grpc_asyncio', request_type=service.CreateSecondaryInstanceRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_secondary_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_secondary_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateSecondaryInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_secondary_instance_async_from_dict():
    await test_create_secondary_instance_async(request_type=dict)

def test_create_secondary_instance_field_headers():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateSecondaryInstanceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_secondary_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_secondary_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_secondary_instance_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateSecondaryInstanceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_secondary_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_secondary_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_secondary_instance_flattened():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_secondary_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_secondary_instance(parent='parent_value', instance=resources.Instance(name='name_value'), instance_id='instance_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].instance
        mock_val = resources.Instance(name='name_value')
        assert arg == mock_val
        arg = args[0].instance_id
        mock_val = 'instance_id_value'
        assert arg == mock_val

def test_create_secondary_instance_flattened_error():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_secondary_instance(service.CreateSecondaryInstanceRequest(), parent='parent_value', instance=resources.Instance(name='name_value'), instance_id='instance_id_value')

@pytest.mark.asyncio
async def test_create_secondary_instance_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_secondary_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_secondary_instance(parent='parent_value', instance=resources.Instance(name='name_value'), instance_id='instance_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].instance
        mock_val = resources.Instance(name='name_value')
        assert arg == mock_val
        arg = args[0].instance_id
        mock_val = 'instance_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_secondary_instance_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_secondary_instance(service.CreateSecondaryInstanceRequest(), parent='parent_value', instance=resources.Instance(name='name_value'), instance_id='instance_id_value')

@pytest.mark.parametrize('request_type', [service.BatchCreateInstancesRequest, dict])
def test_batch_create_instances(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_create_instances), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.batch_create_instances(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.BatchCreateInstancesRequest()
    assert isinstance(response, future.Future)

def test_batch_create_instances_empty_call():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_create_instances), '__call__') as call:
        client.batch_create_instances()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.BatchCreateInstancesRequest()

@pytest.mark.asyncio
async def test_batch_create_instances_async(transport: str='grpc_asyncio', request_type=service.BatchCreateInstancesRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_create_instances), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_create_instances(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.BatchCreateInstancesRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_batch_create_instances_async_from_dict():
    await test_batch_create_instances_async(request_type=dict)

def test_batch_create_instances_field_headers():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.BatchCreateInstancesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_create_instances), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_create_instances(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_create_instances_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.BatchCreateInstancesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_create_instances), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.batch_create_instances(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [service.UpdateInstanceRequest, dict])
def test_update_instance(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateInstanceRequest()
    assert isinstance(response, future.Future)

def test_update_instance_empty_call():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        client.update_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateInstanceRequest()

@pytest.mark.asyncio
async def test_update_instance_async(transport: str='grpc_asyncio', request_type=service.UpdateInstanceRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_instance_async_from_dict():
    await test_update_instance_async(request_type=dict)

def test_update_instance_field_headers():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateInstanceRequest()
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
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateInstanceRequest()
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
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_instance(instance=resources.Instance(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].instance
        mock_val = resources.Instance(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_instance_flattened_error():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_instance(service.UpdateInstanceRequest(), instance=resources.Instance(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_instance_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_instance(instance=resources.Instance(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].instance
        mock_val = resources.Instance(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_instance_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_instance(service.UpdateInstanceRequest(), instance=resources.Instance(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [service.DeleteInstanceRequest, dict])
def test_delete_instance(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteInstanceRequest()
    assert isinstance(response, future.Future)

def test_delete_instance_empty_call():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        client.delete_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteInstanceRequest()

@pytest.mark.asyncio
async def test_delete_instance_async(transport: str='grpc_asyncio', request_type=service.DeleteInstanceRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_instance_async_from_dict():
    await test_delete_instance_async(request_type=dict)

def test_delete_instance_field_headers():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_instance_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_instance_flattened():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_instance(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_instance_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_instance(service.DeleteInstanceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_instance_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_instance(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_instance_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_instance(service.DeleteInstanceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.FailoverInstanceRequest, dict])
def test_failover_instance(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.failover_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.failover_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.FailoverInstanceRequest()
    assert isinstance(response, future.Future)

def test_failover_instance_empty_call():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.failover_instance), '__call__') as call:
        client.failover_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.FailoverInstanceRequest()

@pytest.mark.asyncio
async def test_failover_instance_async(transport: str='grpc_asyncio', request_type=service.FailoverInstanceRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.failover_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.failover_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.FailoverInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_failover_instance_async_from_dict():
    await test_failover_instance_async(request_type=dict)

def test_failover_instance_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.FailoverInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.failover_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.failover_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_failover_instance_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.FailoverInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.failover_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.failover_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_failover_instance_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.failover_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.failover_instance(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_failover_instance_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.failover_instance(service.FailoverInstanceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_failover_instance_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.failover_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.failover_instance(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_failover_instance_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.failover_instance(service.FailoverInstanceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.InjectFaultRequest, dict])
def test_inject_fault(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.inject_fault), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.inject_fault(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.InjectFaultRequest()
    assert isinstance(response, future.Future)

def test_inject_fault_empty_call():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.inject_fault), '__call__') as call:
        client.inject_fault()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.InjectFaultRequest()

@pytest.mark.asyncio
async def test_inject_fault_async(transport: str='grpc_asyncio', request_type=service.InjectFaultRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.inject_fault), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.inject_fault(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.InjectFaultRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_inject_fault_async_from_dict():
    await test_inject_fault_async(request_type=dict)

def test_inject_fault_field_headers():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.InjectFaultRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.inject_fault), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.inject_fault(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_inject_fault_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.InjectFaultRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.inject_fault), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.inject_fault(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_inject_fault_flattened():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.inject_fault), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.inject_fault(fault_type=service.InjectFaultRequest.FaultType.STOP_VM, name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].fault_type
        mock_val = service.InjectFaultRequest.FaultType.STOP_VM
        assert arg == mock_val
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_inject_fault_flattened_error():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.inject_fault(service.InjectFaultRequest(), fault_type=service.InjectFaultRequest.FaultType.STOP_VM, name='name_value')

@pytest.mark.asyncio
async def test_inject_fault_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.inject_fault), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.inject_fault(fault_type=service.InjectFaultRequest.FaultType.STOP_VM, name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].fault_type
        mock_val = service.InjectFaultRequest.FaultType.STOP_VM
        assert arg == mock_val
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_inject_fault_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.inject_fault(service.InjectFaultRequest(), fault_type=service.InjectFaultRequest.FaultType.STOP_VM, name='name_value')

@pytest.mark.parametrize('request_type', [service.RestartInstanceRequest, dict])
def test_restart_instance(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.restart_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.restart_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RestartInstanceRequest()
    assert isinstance(response, future.Future)

def test_restart_instance_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.restart_instance), '__call__') as call:
        client.restart_instance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RestartInstanceRequest()

@pytest.mark.asyncio
async def test_restart_instance_async(transport: str='grpc_asyncio', request_type=service.RestartInstanceRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.restart_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.restart_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.RestartInstanceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_restart_instance_async_from_dict():
    await test_restart_instance_async(request_type=dict)

def test_restart_instance_field_headers():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.RestartInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.restart_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.restart_instance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_restart_instance_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.RestartInstanceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.restart_instance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.restart_instance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_restart_instance_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.restart_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.restart_instance(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_restart_instance_flattened_error():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.restart_instance(service.RestartInstanceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_restart_instance_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.restart_instance), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.restart_instance(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_restart_instance_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.restart_instance(service.RestartInstanceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListBackupsRequest, dict])
def test_list_backups(request_type, transport: str='grpc'):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_backups), '__call__') as call:
        call.return_value = service.ListBackupsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_backups(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListBackupsRequest()
    assert isinstance(response, pagers.ListBackupsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_backups_empty_call():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_backups), '__call__') as call:
        client.list_backups()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListBackupsRequest()

@pytest.mark.asyncio
async def test_list_backups_async(transport: str='grpc_asyncio', request_type=service.ListBackupsRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_backups), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListBackupsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_backups(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListBackupsRequest()
    assert isinstance(response, pagers.ListBackupsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_backups_async_from_dict():
    await test_list_backups_async(request_type=dict)

def test_list_backups_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListBackupsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_backups), '__call__') as call:
        call.return_value = service.ListBackupsResponse()
        client.list_backups(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_backups_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListBackupsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_backups), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListBackupsResponse())
        await client.list_backups(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_backups_flattened():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_backups), '__call__') as call:
        call.return_value = service.ListBackupsResponse()
        client.list_backups(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_backups_flattened_error():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_backups(service.ListBackupsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_backups_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_backups), '__call__') as call:
        call.return_value = service.ListBackupsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListBackupsResponse())
        response = await client.list_backups(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_backups_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_backups(service.ListBackupsRequest(), parent='parent_value')

def test_list_backups_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_backups), '__call__') as call:
        call.side_effect = (service.ListBackupsResponse(backups=[resources.Backup(), resources.Backup(), resources.Backup()], next_page_token='abc'), service.ListBackupsResponse(backups=[], next_page_token='def'), service.ListBackupsResponse(backups=[resources.Backup()], next_page_token='ghi'), service.ListBackupsResponse(backups=[resources.Backup(), resources.Backup()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_backups(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Backup) for i in results))

def test_list_backups_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_backups), '__call__') as call:
        call.side_effect = (service.ListBackupsResponse(backups=[resources.Backup(), resources.Backup(), resources.Backup()], next_page_token='abc'), service.ListBackupsResponse(backups=[], next_page_token='def'), service.ListBackupsResponse(backups=[resources.Backup()], next_page_token='ghi'), service.ListBackupsResponse(backups=[resources.Backup(), resources.Backup()]), RuntimeError)
        pages = list(client.list_backups(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_backups_async_pager():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_backups), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListBackupsResponse(backups=[resources.Backup(), resources.Backup(), resources.Backup()], next_page_token='abc'), service.ListBackupsResponse(backups=[], next_page_token='def'), service.ListBackupsResponse(backups=[resources.Backup()], next_page_token='ghi'), service.ListBackupsResponse(backups=[resources.Backup(), resources.Backup()]), RuntimeError)
        async_pager = await client.list_backups(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Backup) for i in responses))

@pytest.mark.asyncio
async def test_list_backups_async_pages():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_backups), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListBackupsResponse(backups=[resources.Backup(), resources.Backup(), resources.Backup()], next_page_token='abc'), service.ListBackupsResponse(backups=[], next_page_token='def'), service.ListBackupsResponse(backups=[resources.Backup()], next_page_token='ghi'), service.ListBackupsResponse(backups=[resources.Backup(), resources.Backup()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_backups(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetBackupRequest, dict])
def test_get_backup(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_backup), '__call__') as call:
        call.return_value = resources.Backup(name='name_value', display_name='display_name_value', uid='uid_value', state=resources.Backup.State.READY, type_=resources.Backup.Type.ON_DEMAND, description='description_value', cluster_uid='cluster_uid_value', cluster_name='cluster_name_value', reconciling=True, etag='etag_value', size_bytes=1089, database_version=resources.DatabaseVersion.POSTGRES_13)
        response = client.get_backup(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetBackupRequest()
    assert isinstance(response, resources.Backup)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.state == resources.Backup.State.READY
    assert response.type_ == resources.Backup.Type.ON_DEMAND
    assert response.description == 'description_value'
    assert response.cluster_uid == 'cluster_uid_value'
    assert response.cluster_name == 'cluster_name_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'
    assert response.size_bytes == 1089
    assert response.database_version == resources.DatabaseVersion.POSTGRES_13

def test_get_backup_empty_call():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_backup), '__call__') as call:
        client.get_backup()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetBackupRequest()

@pytest.mark.asyncio
async def test_get_backup_async(transport: str='grpc_asyncio', request_type=service.GetBackupRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_backup), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Backup(name='name_value', display_name='display_name_value', uid='uid_value', state=resources.Backup.State.READY, type_=resources.Backup.Type.ON_DEMAND, description='description_value', cluster_uid='cluster_uid_value', cluster_name='cluster_name_value', reconciling=True, etag='etag_value', size_bytes=1089, database_version=resources.DatabaseVersion.POSTGRES_13))
        response = await client.get_backup(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetBackupRequest()
    assert isinstance(response, resources.Backup)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.state == resources.Backup.State.READY
    assert response.type_ == resources.Backup.Type.ON_DEMAND
    assert response.description == 'description_value'
    assert response.cluster_uid == 'cluster_uid_value'
    assert response.cluster_name == 'cluster_name_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'
    assert response.size_bytes == 1089
    assert response.database_version == resources.DatabaseVersion.POSTGRES_13

@pytest.mark.asyncio
async def test_get_backup_async_from_dict():
    await test_get_backup_async(request_type=dict)

def test_get_backup_field_headers():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetBackupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_backup), '__call__') as call:
        call.return_value = resources.Backup()
        client.get_backup(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_backup_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetBackupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_backup), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Backup())
        await client.get_backup(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_backup_flattened():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_backup), '__call__') as call:
        call.return_value = resources.Backup()
        client.get_backup(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_backup_flattened_error():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_backup(service.GetBackupRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_backup_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_backup), '__call__') as call:
        call.return_value = resources.Backup()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Backup())
        response = await client.get_backup(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_backup_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_backup(service.GetBackupRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.CreateBackupRequest, dict])
def test_create_backup(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_backup), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_backup(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateBackupRequest()
    assert isinstance(response, future.Future)

def test_create_backup_empty_call():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_backup), '__call__') as call:
        client.create_backup()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateBackupRequest()

@pytest.mark.asyncio
async def test_create_backup_async(transport: str='grpc_asyncio', request_type=service.CreateBackupRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_backup), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_backup(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateBackupRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_backup_async_from_dict():
    await test_create_backup_async(request_type=dict)

def test_create_backup_field_headers():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateBackupRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_backup), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_backup(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_backup_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateBackupRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_backup), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_backup(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_backup_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_backup), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_backup(parent='parent_value', backup=resources.Backup(name='name_value'), backup_id='backup_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].backup
        mock_val = resources.Backup(name='name_value')
        assert arg == mock_val
        arg = args[0].backup_id
        mock_val = 'backup_id_value'
        assert arg == mock_val

def test_create_backup_flattened_error():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_backup(service.CreateBackupRequest(), parent='parent_value', backup=resources.Backup(name='name_value'), backup_id='backup_id_value')

@pytest.mark.asyncio
async def test_create_backup_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_backup), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_backup(parent='parent_value', backup=resources.Backup(name='name_value'), backup_id='backup_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].backup
        mock_val = resources.Backup(name='name_value')
        assert arg == mock_val
        arg = args[0].backup_id
        mock_val = 'backup_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_backup_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_backup(service.CreateBackupRequest(), parent='parent_value', backup=resources.Backup(name='name_value'), backup_id='backup_id_value')

@pytest.mark.parametrize('request_type', [service.UpdateBackupRequest, dict])
def test_update_backup(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_backup), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_backup(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateBackupRequest()
    assert isinstance(response, future.Future)

def test_update_backup_empty_call():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_backup), '__call__') as call:
        client.update_backup()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateBackupRequest()

@pytest.mark.asyncio
async def test_update_backup_async(transport: str='grpc_asyncio', request_type=service.UpdateBackupRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_backup), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_backup(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateBackupRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_backup_async_from_dict():
    await test_update_backup_async(request_type=dict)

def test_update_backup_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateBackupRequest()
    request.backup.name = 'name_value'
    with mock.patch.object(type(client.transport.update_backup), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_backup(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'backup.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_backup_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateBackupRequest()
    request.backup.name = 'name_value'
    with mock.patch.object(type(client.transport.update_backup), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_backup(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'backup.name=name_value') in kw['metadata']

def test_update_backup_flattened():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_backup), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_backup(backup=resources.Backup(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].backup
        mock_val = resources.Backup(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_backup_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_backup(service.UpdateBackupRequest(), backup=resources.Backup(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_backup_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_backup), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_backup(backup=resources.Backup(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].backup
        mock_val = resources.Backup(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_backup_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_backup(service.UpdateBackupRequest(), backup=resources.Backup(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [service.DeleteBackupRequest, dict])
def test_delete_backup(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_backup), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_backup(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteBackupRequest()
    assert isinstance(response, future.Future)

def test_delete_backup_empty_call():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_backup), '__call__') as call:
        client.delete_backup()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteBackupRequest()

@pytest.mark.asyncio
async def test_delete_backup_async(transport: str='grpc_asyncio', request_type=service.DeleteBackupRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_backup), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_backup(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteBackupRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_backup_async_from_dict():
    await test_delete_backup_async(request_type=dict)

def test_delete_backup_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteBackupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_backup), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_backup(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_backup_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteBackupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_backup), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_backup(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_backup_flattened():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_backup), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_backup(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_backup_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_backup(service.DeleteBackupRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_backup_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_backup), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_backup(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_backup_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_backup(service.DeleteBackupRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListSupportedDatabaseFlagsRequest, dict])
def test_list_supported_database_flags(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_supported_database_flags), '__call__') as call:
        call.return_value = service.ListSupportedDatabaseFlagsResponse(next_page_token='next_page_token_value')
        response = client.list_supported_database_flags(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSupportedDatabaseFlagsRequest()
    assert isinstance(response, pagers.ListSupportedDatabaseFlagsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_supported_database_flags_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_supported_database_flags), '__call__') as call:
        client.list_supported_database_flags()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSupportedDatabaseFlagsRequest()

@pytest.mark.asyncio
async def test_list_supported_database_flags_async(transport: str='grpc_asyncio', request_type=service.ListSupportedDatabaseFlagsRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_supported_database_flags), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSupportedDatabaseFlagsResponse(next_page_token='next_page_token_value'))
        response = await client.list_supported_database_flags(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListSupportedDatabaseFlagsRequest()
    assert isinstance(response, pagers.ListSupportedDatabaseFlagsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_supported_database_flags_async_from_dict():
    await test_list_supported_database_flags_async(request_type=dict)

def test_list_supported_database_flags_field_headers():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListSupportedDatabaseFlagsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_supported_database_flags), '__call__') as call:
        call.return_value = service.ListSupportedDatabaseFlagsResponse()
        client.list_supported_database_flags(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_supported_database_flags_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListSupportedDatabaseFlagsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_supported_database_flags), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSupportedDatabaseFlagsResponse())
        await client.list_supported_database_flags(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_supported_database_flags_flattened():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_supported_database_flags), '__call__') as call:
        call.return_value = service.ListSupportedDatabaseFlagsResponse()
        client.list_supported_database_flags(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_supported_database_flags_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_supported_database_flags(service.ListSupportedDatabaseFlagsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_supported_database_flags_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_supported_database_flags), '__call__') as call:
        call.return_value = service.ListSupportedDatabaseFlagsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListSupportedDatabaseFlagsResponse())
        response = await client.list_supported_database_flags(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_supported_database_flags_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_supported_database_flags(service.ListSupportedDatabaseFlagsRequest(), parent='parent_value')

def test_list_supported_database_flags_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_supported_database_flags), '__call__') as call:
        call.side_effect = (service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[resources.SupportedDatabaseFlag(), resources.SupportedDatabaseFlag(), resources.SupportedDatabaseFlag()], next_page_token='abc'), service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[], next_page_token='def'), service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[resources.SupportedDatabaseFlag()], next_page_token='ghi'), service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[resources.SupportedDatabaseFlag(), resources.SupportedDatabaseFlag()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_supported_database_flags(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.SupportedDatabaseFlag) for i in results))

def test_list_supported_database_flags_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_supported_database_flags), '__call__') as call:
        call.side_effect = (service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[resources.SupportedDatabaseFlag(), resources.SupportedDatabaseFlag(), resources.SupportedDatabaseFlag()], next_page_token='abc'), service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[], next_page_token='def'), service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[resources.SupportedDatabaseFlag()], next_page_token='ghi'), service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[resources.SupportedDatabaseFlag(), resources.SupportedDatabaseFlag()]), RuntimeError)
        pages = list(client.list_supported_database_flags(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_supported_database_flags_async_pager():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_supported_database_flags), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[resources.SupportedDatabaseFlag(), resources.SupportedDatabaseFlag(), resources.SupportedDatabaseFlag()], next_page_token='abc'), service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[], next_page_token='def'), service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[resources.SupportedDatabaseFlag()], next_page_token='ghi'), service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[resources.SupportedDatabaseFlag(), resources.SupportedDatabaseFlag()]), RuntimeError)
        async_pager = await client.list_supported_database_flags(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.SupportedDatabaseFlag) for i in responses))

@pytest.mark.asyncio
async def test_list_supported_database_flags_async_pages():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_supported_database_flags), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[resources.SupportedDatabaseFlag(), resources.SupportedDatabaseFlag(), resources.SupportedDatabaseFlag()], next_page_token='abc'), service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[], next_page_token='def'), service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[resources.SupportedDatabaseFlag()], next_page_token='ghi'), service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[resources.SupportedDatabaseFlag(), resources.SupportedDatabaseFlag()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_supported_database_flags(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GenerateClientCertificateRequest, dict])
def test_generate_client_certificate(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_client_certificate), '__call__') as call:
        call.return_value = service.GenerateClientCertificateResponse(pem_certificate='pem_certificate_value', pem_certificate_chain=['pem_certificate_chain_value'], ca_cert='ca_cert_value')
        response = client.generate_client_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GenerateClientCertificateRequest()
    assert isinstance(response, service.GenerateClientCertificateResponse)
    assert response.pem_certificate == 'pem_certificate_value'
    assert response.pem_certificate_chain == ['pem_certificate_chain_value']
    assert response.ca_cert == 'ca_cert_value'

def test_generate_client_certificate_empty_call():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.generate_client_certificate), '__call__') as call:
        client.generate_client_certificate()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GenerateClientCertificateRequest()

@pytest.mark.asyncio
async def test_generate_client_certificate_async(transport: str='grpc_asyncio', request_type=service.GenerateClientCertificateRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_client_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.GenerateClientCertificateResponse(pem_certificate='pem_certificate_value', pem_certificate_chain=['pem_certificate_chain_value'], ca_cert='ca_cert_value'))
        response = await client.generate_client_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GenerateClientCertificateRequest()
    assert isinstance(response, service.GenerateClientCertificateResponse)
    assert response.pem_certificate == 'pem_certificate_value'
    assert response.pem_certificate_chain == ['pem_certificate_chain_value']
    assert response.ca_cert == 'ca_cert_value'

@pytest.mark.asyncio
async def test_generate_client_certificate_async_from_dict():
    await test_generate_client_certificate_async(request_type=dict)

def test_generate_client_certificate_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GenerateClientCertificateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.generate_client_certificate), '__call__') as call:
        call.return_value = service.GenerateClientCertificateResponse()
        client.generate_client_certificate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_generate_client_certificate_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GenerateClientCertificateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.generate_client_certificate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.GenerateClientCertificateResponse())
        await client.generate_client_certificate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_generate_client_certificate_flattened():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.generate_client_certificate), '__call__') as call:
        call.return_value = service.GenerateClientCertificateResponse()
        client.generate_client_certificate(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_generate_client_certificate_flattened_error():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.generate_client_certificate(service.GenerateClientCertificateRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_generate_client_certificate_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.generate_client_certificate), '__call__') as call:
        call.return_value = service.GenerateClientCertificateResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.GenerateClientCertificateResponse())
        response = await client.generate_client_certificate(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_generate_client_certificate_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.generate_client_certificate(service.GenerateClientCertificateRequest(), parent='parent_value')

@pytest.mark.parametrize('request_type', [service.GetConnectionInfoRequest, dict])
def test_get_connection_info(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_connection_info), '__call__') as call:
        call.return_value = resources.ConnectionInfo(name='name_value', ip_address='ip_address_value', pem_certificate_chain=['pem_certificate_chain_value'], instance_uid='instance_uid_value')
        response = client.get_connection_info(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetConnectionInfoRequest()
    assert isinstance(response, resources.ConnectionInfo)
    assert response.name == 'name_value'
    assert response.ip_address == 'ip_address_value'
    assert response.pem_certificate_chain == ['pem_certificate_chain_value']
    assert response.instance_uid == 'instance_uid_value'

def test_get_connection_info_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_connection_info), '__call__') as call:
        client.get_connection_info()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetConnectionInfoRequest()

@pytest.mark.asyncio
async def test_get_connection_info_async(transport: str='grpc_asyncio', request_type=service.GetConnectionInfoRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_connection_info), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.ConnectionInfo(name='name_value', ip_address='ip_address_value', pem_certificate_chain=['pem_certificate_chain_value'], instance_uid='instance_uid_value'))
        response = await client.get_connection_info(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetConnectionInfoRequest()
    assert isinstance(response, resources.ConnectionInfo)
    assert response.name == 'name_value'
    assert response.ip_address == 'ip_address_value'
    assert response.pem_certificate_chain == ['pem_certificate_chain_value']
    assert response.instance_uid == 'instance_uid_value'

@pytest.mark.asyncio
async def test_get_connection_info_async_from_dict():
    await test_get_connection_info_async(request_type=dict)

def test_get_connection_info_field_headers():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetConnectionInfoRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.get_connection_info), '__call__') as call:
        call.return_value = resources.ConnectionInfo()
        client.get_connection_info(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_connection_info_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetConnectionInfoRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.get_connection_info), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.ConnectionInfo())
        await client.get_connection_info(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_get_connection_info_flattened():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_connection_info), '__call__') as call:
        call.return_value = resources.ConnectionInfo()
        client.get_connection_info(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_get_connection_info_flattened_error():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_connection_info(service.GetConnectionInfoRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_get_connection_info_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_connection_info), '__call__') as call:
        call.return_value = resources.ConnectionInfo()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.ConnectionInfo())
        response = await client.get_connection_info(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_connection_info_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_connection_info(service.GetConnectionInfoRequest(), parent='parent_value')

@pytest.mark.parametrize('request_type', [service.ListUsersRequest, dict])
def test_list_users(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_users), '__call__') as call:
        call.return_value = service.ListUsersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_users(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListUsersRequest()
    assert isinstance(response, pagers.ListUsersPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_users_empty_call():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_users), '__call__') as call:
        client.list_users()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListUsersRequest()

@pytest.mark.asyncio
async def test_list_users_async(transport: str='grpc_asyncio', request_type=service.ListUsersRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_users), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListUsersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_users(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListUsersRequest()
    assert isinstance(response, pagers.ListUsersAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_users_async_from_dict():
    await test_list_users_async(request_type=dict)

def test_list_users_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListUsersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_users), '__call__') as call:
        call.return_value = service.ListUsersResponse()
        client.list_users(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_users_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListUsersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_users), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListUsersResponse())
        await client.list_users(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_users_flattened():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_users), '__call__') as call:
        call.return_value = service.ListUsersResponse()
        client.list_users(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_users_flattened_error():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_users(service.ListUsersRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_users_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_users), '__call__') as call:
        call.return_value = service.ListUsersResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListUsersResponse())
        response = await client.list_users(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_users_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_users(service.ListUsersRequest(), parent='parent_value')

def test_list_users_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_users), '__call__') as call:
        call.side_effect = (service.ListUsersResponse(users=[resources.User(), resources.User(), resources.User()], next_page_token='abc'), service.ListUsersResponse(users=[], next_page_token='def'), service.ListUsersResponse(users=[resources.User()], next_page_token='ghi'), service.ListUsersResponse(users=[resources.User(), resources.User()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_users(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.User) for i in results))

def test_list_users_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_users), '__call__') as call:
        call.side_effect = (service.ListUsersResponse(users=[resources.User(), resources.User(), resources.User()], next_page_token='abc'), service.ListUsersResponse(users=[], next_page_token='def'), service.ListUsersResponse(users=[resources.User()], next_page_token='ghi'), service.ListUsersResponse(users=[resources.User(), resources.User()]), RuntimeError)
        pages = list(client.list_users(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_users_async_pager():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_users), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListUsersResponse(users=[resources.User(), resources.User(), resources.User()], next_page_token='abc'), service.ListUsersResponse(users=[], next_page_token='def'), service.ListUsersResponse(users=[resources.User()], next_page_token='ghi'), service.ListUsersResponse(users=[resources.User(), resources.User()]), RuntimeError)
        async_pager = await client.list_users(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.User) for i in responses))

@pytest.mark.asyncio
async def test_list_users_async_pages():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_users), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListUsersResponse(users=[resources.User(), resources.User(), resources.User()], next_page_token='abc'), service.ListUsersResponse(users=[], next_page_token='def'), service.ListUsersResponse(users=[resources.User()], next_page_token='ghi'), service.ListUsersResponse(users=[resources.User(), resources.User()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_users(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetUserRequest, dict])
def test_get_user(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_user), '__call__') as call:
        call.return_value = resources.User(name='name_value', password='password_value', database_roles=['database_roles_value'], user_type=resources.User.UserType.ALLOYDB_BUILT_IN)
        response = client.get_user(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetUserRequest()
    assert isinstance(response, resources.User)
    assert response.name == 'name_value'
    assert response.password == 'password_value'
    assert response.database_roles == ['database_roles_value']
    assert response.user_type == resources.User.UserType.ALLOYDB_BUILT_IN

def test_get_user_empty_call():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_user), '__call__') as call:
        client.get_user()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetUserRequest()

@pytest.mark.asyncio
async def test_get_user_async(transport: str='grpc_asyncio', request_type=service.GetUserRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_user), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.User(name='name_value', password='password_value', database_roles=['database_roles_value'], user_type=resources.User.UserType.ALLOYDB_BUILT_IN))
        response = await client.get_user(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetUserRequest()
    assert isinstance(response, resources.User)
    assert response.name == 'name_value'
    assert response.password == 'password_value'
    assert response.database_roles == ['database_roles_value']
    assert response.user_type == resources.User.UserType.ALLOYDB_BUILT_IN

@pytest.mark.asyncio
async def test_get_user_async_from_dict():
    await test_get_user_async(request_type=dict)

def test_get_user_field_headers():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetUserRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_user), '__call__') as call:
        call.return_value = resources.User()
        client.get_user(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_user_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetUserRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_user), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.User())
        await client.get_user(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_user_flattened():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_user), '__call__') as call:
        call.return_value = resources.User()
        client.get_user(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_user_flattened_error():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_user(service.GetUserRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_user_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_user), '__call__') as call:
        call.return_value = resources.User()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.User())
        response = await client.get_user(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_user_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_user(service.GetUserRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.CreateUserRequest, dict])
def test_create_user(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_user), '__call__') as call:
        call.return_value = resources.User(name='name_value', password='password_value', database_roles=['database_roles_value'], user_type=resources.User.UserType.ALLOYDB_BUILT_IN)
        response = client.create_user(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateUserRequest()
    assert isinstance(response, resources.User)
    assert response.name == 'name_value'
    assert response.password == 'password_value'
    assert response.database_roles == ['database_roles_value']
    assert response.user_type == resources.User.UserType.ALLOYDB_BUILT_IN

def test_create_user_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_user), '__call__') as call:
        client.create_user()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateUserRequest()

@pytest.mark.asyncio
async def test_create_user_async(transport: str='grpc_asyncio', request_type=service.CreateUserRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_user), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.User(name='name_value', password='password_value', database_roles=['database_roles_value'], user_type=resources.User.UserType.ALLOYDB_BUILT_IN))
        response = await client.create_user(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateUserRequest()
    assert isinstance(response, resources.User)
    assert response.name == 'name_value'
    assert response.password == 'password_value'
    assert response.database_roles == ['database_roles_value']
    assert response.user_type == resources.User.UserType.ALLOYDB_BUILT_IN

@pytest.mark.asyncio
async def test_create_user_async_from_dict():
    await test_create_user_async(request_type=dict)

def test_create_user_field_headers():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateUserRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_user), '__call__') as call:
        call.return_value = resources.User()
        client.create_user(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_user_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateUserRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_user), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.User())
        await client.create_user(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_user_flattened():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_user), '__call__') as call:
        call.return_value = resources.User()
        client.create_user(parent='parent_value', user=resources.User(name='name_value'), user_id='user_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].user
        mock_val = resources.User(name='name_value')
        assert arg == mock_val
        arg = args[0].user_id
        mock_val = 'user_id_value'
        assert arg == mock_val

def test_create_user_flattened_error():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_user(service.CreateUserRequest(), parent='parent_value', user=resources.User(name='name_value'), user_id='user_id_value')

@pytest.mark.asyncio
async def test_create_user_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_user), '__call__') as call:
        call.return_value = resources.User()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.User())
        response = await client.create_user(parent='parent_value', user=resources.User(name='name_value'), user_id='user_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].user
        mock_val = resources.User(name='name_value')
        assert arg == mock_val
        arg = args[0].user_id
        mock_val = 'user_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_user_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_user(service.CreateUserRequest(), parent='parent_value', user=resources.User(name='name_value'), user_id='user_id_value')

@pytest.mark.parametrize('request_type', [service.UpdateUserRequest, dict])
def test_update_user(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_user), '__call__') as call:
        call.return_value = resources.User(name='name_value', password='password_value', database_roles=['database_roles_value'], user_type=resources.User.UserType.ALLOYDB_BUILT_IN)
        response = client.update_user(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateUserRequest()
    assert isinstance(response, resources.User)
    assert response.name == 'name_value'
    assert response.password == 'password_value'
    assert response.database_roles == ['database_roles_value']
    assert response.user_type == resources.User.UserType.ALLOYDB_BUILT_IN

def test_update_user_empty_call():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_user), '__call__') as call:
        client.update_user()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateUserRequest()

@pytest.mark.asyncio
async def test_update_user_async(transport: str='grpc_asyncio', request_type=service.UpdateUserRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_user), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.User(name='name_value', password='password_value', database_roles=['database_roles_value'], user_type=resources.User.UserType.ALLOYDB_BUILT_IN))
        response = await client.update_user(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateUserRequest()
    assert isinstance(response, resources.User)
    assert response.name == 'name_value'
    assert response.password == 'password_value'
    assert response.database_roles == ['database_roles_value']
    assert response.user_type == resources.User.UserType.ALLOYDB_BUILT_IN

@pytest.mark.asyncio
async def test_update_user_async_from_dict():
    await test_update_user_async(request_type=dict)

def test_update_user_field_headers():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateUserRequest()
    request.user.name = 'name_value'
    with mock.patch.object(type(client.transport.update_user), '__call__') as call:
        call.return_value = resources.User()
        client.update_user(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'user.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_user_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateUserRequest()
    request.user.name = 'name_value'
    with mock.patch.object(type(client.transport.update_user), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.User())
        await client.update_user(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'user.name=name_value') in kw['metadata']

def test_update_user_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_user), '__call__') as call:
        call.return_value = resources.User()
        client.update_user(user=resources.User(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].user
        mock_val = resources.User(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_user_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_user(service.UpdateUserRequest(), user=resources.User(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_user_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_user), '__call__') as call:
        call.return_value = resources.User()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.User())
        response = await client.update_user(user=resources.User(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].user
        mock_val = resources.User(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_user_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_user(service.UpdateUserRequest(), user=resources.User(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [service.DeleteUserRequest, dict])
def test_delete_user(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_user), '__call__') as call:
        call.return_value = None
        response = client.delete_user(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteUserRequest()
    assert response is None

def test_delete_user_empty_call():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_user), '__call__') as call:
        client.delete_user()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteUserRequest()

@pytest.mark.asyncio
async def test_delete_user_async(transport: str='grpc_asyncio', request_type=service.DeleteUserRequest):
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_user), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_user(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteUserRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_user_async_from_dict():
    await test_delete_user_async(request_type=dict)

def test_delete_user_field_headers():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteUserRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_user), '__call__') as call:
        call.return_value = None
        client.delete_user(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_user_field_headers_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteUserRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_user), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_user(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_user_flattened():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_user), '__call__') as call:
        call.return_value = None
        client.delete_user(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_user_flattened_error():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_user(service.DeleteUserRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_user_flattened_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_user), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_user(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_user_flattened_error_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_user(service.DeleteUserRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListClustersRequest, dict])
def test_list_clusters_rest(request_type):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListClustersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListClustersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_clusters(request)
    assert isinstance(response, pagers.ListClustersPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_clusters_rest_required_fields(request_type=service.ListClustersRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AlloyDBAdminRestTransport
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListClustersResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListClustersResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_clusters(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_clusters_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_clusters._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_clusters_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_list_clusters') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_list_clusters') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListClustersRequest.pb(service.ListClustersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListClustersResponse.to_json(service.ListClustersResponse())
        request = service.ListClustersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListClustersResponse()
        client.list_clusters(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_clusters_rest_bad_request(transport: str='rest', request_type=service.ListClustersRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_clusters(request)

def test_list_clusters_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListClustersResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListClustersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_clusters(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*}/clusters' % client.transport._host, args[1])

def test_list_clusters_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_clusters(service.ListClustersRequest(), parent='parent_value')

def test_list_clusters_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListClustersResponse(clusters=[resources.Cluster(), resources.Cluster(), resources.Cluster()], next_page_token='abc'), service.ListClustersResponse(clusters=[], next_page_token='def'), service.ListClustersResponse(clusters=[resources.Cluster()], next_page_token='ghi'), service.ListClustersResponse(clusters=[resources.Cluster(), resources.Cluster()]))
        response = response + response
        response = tuple((service.ListClustersResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_clusters(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Cluster) for i in results))
        pages = list(client.list_clusters(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetClusterRequest, dict])
def test_get_cluster_rest(request_type):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Cluster(name='name_value', display_name='display_name_value', uid='uid_value', state=resources.Cluster.State.READY, cluster_type=resources.Cluster.ClusterType.PRIMARY, database_version=resources.DatabaseVersion.POSTGRES_13, network='network_value', etag='etag_value', reconciling=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Cluster.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_cluster(request)
    assert isinstance(response, resources.Cluster)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.state == resources.Cluster.State.READY
    assert response.cluster_type == resources.Cluster.ClusterType.PRIMARY
    assert response.database_version == resources.DatabaseVersion.POSTGRES_13
    assert response.network == 'network_value'
    assert response.etag == 'etag_value'
    assert response.reconciling is True

def test_get_cluster_rest_required_fields(request_type=service.GetClusterRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_cluster._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('view',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Cluster()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Cluster.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_cluster(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_cluster_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(('view',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_cluster_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_get_cluster') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_get_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetClusterRequest.pb(service.GetClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Cluster.to_json(resources.Cluster())
        request = service.GetClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Cluster()
        client.get_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_cluster_rest_bad_request(transport: str='rest', request_type=service.GetClusterRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_cluster(request)

def test_get_cluster_rest_flattened():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Cluster()
        sample_request = {'name': 'projects/sample1/locations/sample2/clusters/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Cluster.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/clusters/*}' % client.transport._host, args[1])

def test_get_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_cluster(service.GetClusterRequest(), name='name_value')

def test_get_cluster_rest_error():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateClusterRequest, dict])
def test_create_cluster_rest(request_type):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['cluster'] = {'backup_source': {'backup_uid': 'backup_uid_value', 'backup_name': 'backup_name_value'}, 'migration_source': {'host_port': 'host_port_value', 'reference_id': 'reference_id_value', 'source_type': 1}, 'name': 'name_value', 'display_name': 'display_name_value', 'uid': 'uid_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'labels': {}, 'state': 1, 'cluster_type': 1, 'database_version': 1, 'network_config': {'network': 'network_value', 'allocated_ip_range': 'allocated_ip_range_value'}, 'network': 'network_value', 'etag': 'etag_value', 'annotations': {}, 'reconciling': True, 'initial_user': {'user': 'user_value', 'password': 'password_value'}, 'automated_backup_policy': {'weekly_schedule': {'start_times': [{'hours': 561, 'minutes': 773, 'seconds': 751, 'nanos': 543}], 'days_of_week': [1]}, 'time_based_retention': {'retention_period': {'seconds': 751, 'nanos': 543}}, 'quantity_based_retention': {'count': 553}, 'enabled': True, 'backup_window': {}, 'encryption_config': {'kms_key_name': 'kms_key_name_value'}, 'location': 'location_value', 'labels': {}}, 'ssl_config': {'ssl_mode': 1, 'ca_source': 1}, 'encryption_config': {}, 'encryption_info': {'encryption_type': 1, 'kms_key_versions': ['kms_key_versions_value1', 'kms_key_versions_value2']}, 'continuous_backup_config': {'enabled': True, 'recovery_window_days': 2166, 'encryption_config': {}}, 'continuous_backup_info': {'encryption_info': {}, 'enabled_time': {}, 'schedule': [1], 'earliest_restorable_time': {}}, 'secondary_config': {'primary_cluster_name': 'primary_cluster_name_value'}, 'primary_config': {'secondary_cluster_names': ['secondary_cluster_names_value1', 'secondary_cluster_names_value2']}}
    test_field = service.CreateClusterRequest.meta.fields['cluster']

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

def test_create_cluster_rest_required_fields(request_type=service.CreateClusterRequest):
    if False:
        print('Hello World!')
    transport_class = transports.AlloyDBAdminRestTransport
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(('clusterId', 'requestId', 'validateOnly')) & set(('parent', 'clusterId', 'cluster'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_cluster_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_create_cluster') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_create_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateClusterRequest.pb(service.CreateClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.CreateClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_cluster_rest_bad_request(transport: str='rest', request_type=service.CreateClusterRequest):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_cluster(request)

def test_create_cluster_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', cluster=resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value')), cluster_id='cluster_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*}/clusters' % client.transport._host, args[1])

def test_create_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_cluster(service.CreateClusterRequest(), parent='parent_value', cluster=resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value')), cluster_id='cluster_id_value')

def test_create_cluster_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateClusterRequest, dict])
def test_update_cluster_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'cluster': {'name': 'projects/sample1/locations/sample2/clusters/sample3'}}
    request_init['cluster'] = {'backup_source': {'backup_uid': 'backup_uid_value', 'backup_name': 'backup_name_value'}, 'migration_source': {'host_port': 'host_port_value', 'reference_id': 'reference_id_value', 'source_type': 1}, 'name': 'projects/sample1/locations/sample2/clusters/sample3', 'display_name': 'display_name_value', 'uid': 'uid_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'labels': {}, 'state': 1, 'cluster_type': 1, 'database_version': 1, 'network_config': {'network': 'network_value', 'allocated_ip_range': 'allocated_ip_range_value'}, 'network': 'network_value', 'etag': 'etag_value', 'annotations': {}, 'reconciling': True, 'initial_user': {'user': 'user_value', 'password': 'password_value'}, 'automated_backup_policy': {'weekly_schedule': {'start_times': [{'hours': 561, 'minutes': 773, 'seconds': 751, 'nanos': 543}], 'days_of_week': [1]}, 'time_based_retention': {'retention_period': {'seconds': 751, 'nanos': 543}}, 'quantity_based_retention': {'count': 553}, 'enabled': True, 'backup_window': {}, 'encryption_config': {'kms_key_name': 'kms_key_name_value'}, 'location': 'location_value', 'labels': {}}, 'ssl_config': {'ssl_mode': 1, 'ca_source': 1}, 'encryption_config': {}, 'encryption_info': {'encryption_type': 1, 'kms_key_versions': ['kms_key_versions_value1', 'kms_key_versions_value2']}, 'continuous_backup_config': {'enabled': True, 'recovery_window_days': 2166, 'encryption_config': {}}, 'continuous_backup_info': {'encryption_info': {}, 'enabled_time': {}, 'schedule': [1], 'earliest_restorable_time': {}}, 'secondary_config': {'primary_cluster_name': 'primary_cluster_name_value'}, 'primary_config': {'secondary_cluster_names': ['secondary_cluster_names_value1', 'secondary_cluster_names_value2']}}
    test_field = service.UpdateClusterRequest.meta.fields['cluster']

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

def test_update_cluster_rest_required_fields(request_type=service.UpdateClusterRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_cluster._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'request_id', 'update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'requestId', 'updateMask', 'validateOnly')) & set(('cluster',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_cluster_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_update_cluster') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_update_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateClusterRequest.pb(service.UpdateClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.UpdateClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_cluster_rest_bad_request(transport: str='rest', request_type=service.UpdateClusterRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'cluster': {'name': 'projects/sample1/locations/sample2/clusters/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_cluster(request)

def test_update_cluster_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'cluster': {'name': 'projects/sample1/locations/sample2/clusters/sample3'}}
        mock_args = dict(cluster=resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{cluster.name=projects/*/locations/*/clusters/*}' % client.transport._host, args[1])

def test_update_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_cluster(service.UpdateClusterRequest(), cluster=resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_cluster_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DeleteClusterRequest, dict])
def test_delete_cluster_rest(request_type):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3'}
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

def test_delete_cluster_rest_required_fields(request_type=service.DeleteClusterRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_cluster._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('etag', 'force', 'request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(('etag', 'force', 'requestId', 'validateOnly')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_cluster_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_delete_cluster') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_delete_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DeleteClusterRequest.pb(service.DeleteClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.DeleteClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_cluster_rest_bad_request(transport: str='rest', request_type=service.DeleteClusterRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_cluster(request)

def test_delete_cluster_rest_flattened():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/clusters/sample3'}
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
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/clusters/*}' % client.transport._host, args[1])

def test_delete_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_cluster(service.DeleteClusterRequest(), name='name_value')

def test_delete_cluster_rest_error():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.PromoteClusterRequest, dict])
def test_promote_cluster_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.promote_cluster(request)
    assert response.operation.name == 'operations/spam'

def test_promote_cluster_rest_required_fields(request_type=service.PromoteClusterRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).promote_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).promote_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.promote_cluster(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_promote_cluster_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.promote_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_promote_cluster_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_promote_cluster') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_promote_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.PromoteClusterRequest.pb(service.PromoteClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.PromoteClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.promote_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_promote_cluster_rest_bad_request(transport: str='rest', request_type=service.PromoteClusterRequest):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.promote_cluster(request)

def test_promote_cluster_rest_flattened():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/clusters/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.promote_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/clusters/*}:promote' % client.transport._host, args[1])

def test_promote_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.promote_cluster(service.PromoteClusterRequest(), name='name_value')

def test_promote_cluster_rest_error():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.RestoreClusterRequest, dict])
def test_restore_cluster_rest(request_type):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.restore_cluster(request)
    assert response.operation.name == 'operations/spam'

def test_restore_cluster_rest_required_fields(request_type=service.RestoreClusterRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['cluster_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).restore_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['clusterId'] = 'cluster_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).restore_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'clusterId' in jsonified_request
    assert jsonified_request['clusterId'] == 'cluster_id_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.restore_cluster(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_restore_cluster_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.restore_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'clusterId', 'cluster'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_restore_cluster_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_restore_cluster') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_restore_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.RestoreClusterRequest.pb(service.RestoreClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.RestoreClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.restore_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_restore_cluster_rest_bad_request(transport: str='rest', request_type=service.RestoreClusterRequest):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.restore_cluster(request)

def test_restore_cluster_rest_error():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateSecondaryClusterRequest, dict])
def test_create_secondary_cluster_rest(request_type):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['cluster'] = {'backup_source': {'backup_uid': 'backup_uid_value', 'backup_name': 'backup_name_value'}, 'migration_source': {'host_port': 'host_port_value', 'reference_id': 'reference_id_value', 'source_type': 1}, 'name': 'name_value', 'display_name': 'display_name_value', 'uid': 'uid_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'labels': {}, 'state': 1, 'cluster_type': 1, 'database_version': 1, 'network_config': {'network': 'network_value', 'allocated_ip_range': 'allocated_ip_range_value'}, 'network': 'network_value', 'etag': 'etag_value', 'annotations': {}, 'reconciling': True, 'initial_user': {'user': 'user_value', 'password': 'password_value'}, 'automated_backup_policy': {'weekly_schedule': {'start_times': [{'hours': 561, 'minutes': 773, 'seconds': 751, 'nanos': 543}], 'days_of_week': [1]}, 'time_based_retention': {'retention_period': {'seconds': 751, 'nanos': 543}}, 'quantity_based_retention': {'count': 553}, 'enabled': True, 'backup_window': {}, 'encryption_config': {'kms_key_name': 'kms_key_name_value'}, 'location': 'location_value', 'labels': {}}, 'ssl_config': {'ssl_mode': 1, 'ca_source': 1}, 'encryption_config': {}, 'encryption_info': {'encryption_type': 1, 'kms_key_versions': ['kms_key_versions_value1', 'kms_key_versions_value2']}, 'continuous_backup_config': {'enabled': True, 'recovery_window_days': 2166, 'encryption_config': {}}, 'continuous_backup_info': {'encryption_info': {}, 'enabled_time': {}, 'schedule': [1], 'earliest_restorable_time': {}}, 'secondary_config': {'primary_cluster_name': 'primary_cluster_name_value'}, 'primary_config': {'secondary_cluster_names': ['secondary_cluster_names_value1', 'secondary_cluster_names_value2']}}
    test_field = service.CreateSecondaryClusterRequest.meta.fields['cluster']

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
        response = client.create_secondary_cluster(request)
    assert response.operation.name == 'operations/spam'

def test_create_secondary_cluster_rest_required_fields(request_type=service.CreateSecondaryClusterRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['cluster_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'clusterId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_secondary_cluster._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'clusterId' in jsonified_request
    assert jsonified_request['clusterId'] == request_init['cluster_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['clusterId'] = 'cluster_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_secondary_cluster._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('cluster_id', 'request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'clusterId' in jsonified_request
    assert jsonified_request['clusterId'] == 'cluster_id_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_secondary_cluster(request)
            expected_params = [('clusterId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_secondary_cluster_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_secondary_cluster._get_unset_required_fields({})
    assert set(unset_fields) == set(('clusterId', 'requestId', 'validateOnly')) & set(('parent', 'clusterId', 'cluster'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_secondary_cluster_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_create_secondary_cluster') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_create_secondary_cluster') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateSecondaryClusterRequest.pb(service.CreateSecondaryClusterRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.CreateSecondaryClusterRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_secondary_cluster(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_secondary_cluster_rest_bad_request(transport: str='rest', request_type=service.CreateSecondaryClusterRequest):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_secondary_cluster(request)

def test_create_secondary_cluster_rest_flattened():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', cluster=resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value')), cluster_id='cluster_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_secondary_cluster(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*}/clusters:createsecondary' % client.transport._host, args[1])

def test_create_secondary_cluster_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_secondary_cluster(service.CreateSecondaryClusterRequest(), parent='parent_value', cluster=resources.Cluster(backup_source=resources.BackupSource(backup_uid='backup_uid_value')), cluster_id='cluster_id_value')

def test_create_secondary_cluster_rest_error():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListInstancesRequest, dict])
def test_list_instances_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListInstancesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListInstancesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_instances(request)
    assert isinstance(response, pagers.ListInstancesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_instances_rest_required_fields(request_type=service.ListInstancesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_instances._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListInstancesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListInstancesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_instances(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_instances_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_instances._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_instances_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_list_instances') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_list_instances') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListInstancesRequest.pb(service.ListInstancesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListInstancesResponse.to_json(service.ListInstancesResponse())
        request = service.ListInstancesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListInstancesResponse()
        client.list_instances(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_instances_rest_bad_request(transport: str='rest', request_type=service.ListInstancesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_instances(request)

def test_list_instances_rest_flattened():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListInstancesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListInstancesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_instances(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*/clusters/*}/instances' % client.transport._host, args[1])

def test_list_instances_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_instances(service.ListInstancesRequest(), parent='parent_value')

def test_list_instances_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListInstancesResponse(instances=[resources.Instance(), resources.Instance(), resources.Instance()], next_page_token='abc'), service.ListInstancesResponse(instances=[], next_page_token='def'), service.ListInstancesResponse(instances=[resources.Instance()], next_page_token='ghi'), service.ListInstancesResponse(instances=[resources.Instance(), resources.Instance()]))
        response = response + response
        response = tuple((service.ListInstancesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
        pager = client.list_instances(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Instance) for i in results))
        pages = list(client.list_instances(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetInstanceRequest, dict])
def test_get_instance_rest(request_type):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Instance(name='name_value', display_name='display_name_value', uid='uid_value', state=resources.Instance.State.READY, instance_type=resources.Instance.InstanceType.PRIMARY, availability_type=resources.Instance.AvailabilityType.ZONAL, gce_zone='gce_zone_value', ip_address='ip_address_value', reconciling=True, etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Instance.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_instance(request)
    assert isinstance(response, resources.Instance)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.state == resources.Instance.State.READY
    assert response.instance_type == resources.Instance.InstanceType.PRIMARY
    assert response.availability_type == resources.Instance.AvailabilityType.ZONAL
    assert response.gce_zone == 'gce_zone_value'
    assert response.ip_address == 'ip_address_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'

def test_get_instance_rest_required_fields(request_type=service.GetInstanceRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_instance._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('view',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Instance()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Instance.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_instance_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(('view',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_instance_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_get_instance') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_get_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetInstanceRequest.pb(service.GetInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Instance.to_json(resources.Instance())
        request = service.GetInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Instance()
        client.get_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_instance_rest_bad_request(transport: str='rest', request_type=service.GetInstanceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_instance(request)

def test_get_instance_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Instance()
        sample_request = {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Instance.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/clusters/*/instances/*}' % client.transport._host, args[1])

def test_get_instance_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_instance(service.GetInstanceRequest(), name='name_value')

def test_get_instance_rest_error():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateInstanceRequest, dict])
def test_create_instance_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
    request_init['instance'] = {'name': 'name_value', 'display_name': 'display_name_value', 'uid': 'uid_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'labels': {}, 'state': 1, 'instance_type': 1, 'machine_config': {'cpu_count': 976}, 'availability_type': 1, 'gce_zone': 'gce_zone_value', 'database_flags': {}, 'writable_node': {'zone_id': 'zone_id_value', 'id': 'id_value', 'ip': 'ip_value', 'state': 'state_value'}, 'nodes': {}, 'query_insights_config': {'record_application_tags': True, 'record_client_address': True, 'query_string_length': 2061, 'query_plans_per_minute': 2378}, 'read_pool_config': {'node_count': 1070}, 'ip_address': 'ip_address_value', 'reconciling': True, 'etag': 'etag_value', 'annotations': {}, 'update_policy': {'mode': 1}, 'client_connection_config': {'require_connectors': True, 'ssl_config': {'ssl_mode': 1, 'ca_source': 1}}}
    test_field = service.CreateInstanceRequest.meta.fields['instance']

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
        response = client.create_instance(request)
    assert response.operation.name == 'operations/spam'

def test_create_instance_rest_required_fields(request_type=service.CreateInstanceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['instance_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'instanceId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'instanceId' in jsonified_request
    assert jsonified_request['instanceId'] == request_init['instance_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['instanceId'] = 'instance_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_instance._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('instance_id', 'request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'instanceId' in jsonified_request
    assert jsonified_request['instanceId'] == 'instance_id_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_instance(request)
            expected_params = [('instanceId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_instance_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(('instanceId', 'requestId', 'validateOnly')) & set(('parent', 'instanceId', 'instance'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_instance_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_create_instance') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_create_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateInstanceRequest.pb(service.CreateInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.CreateInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_instance_rest_bad_request(transport: str='rest', request_type=service.CreateInstanceRequest):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_instance(request)

def test_create_instance_rest_flattened():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
        mock_args = dict(parent='parent_value', instance=resources.Instance(name='name_value'), instance_id='instance_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*/clusters/*}/instances' % client.transport._host, args[1])

def test_create_instance_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_instance(service.CreateInstanceRequest(), parent='parent_value', instance=resources.Instance(name='name_value'), instance_id='instance_id_value')

def test_create_instance_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateSecondaryInstanceRequest, dict])
def test_create_secondary_instance_rest(request_type):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
    request_init['instance'] = {'name': 'name_value', 'display_name': 'display_name_value', 'uid': 'uid_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'labels': {}, 'state': 1, 'instance_type': 1, 'machine_config': {'cpu_count': 976}, 'availability_type': 1, 'gce_zone': 'gce_zone_value', 'database_flags': {}, 'writable_node': {'zone_id': 'zone_id_value', 'id': 'id_value', 'ip': 'ip_value', 'state': 'state_value'}, 'nodes': {}, 'query_insights_config': {'record_application_tags': True, 'record_client_address': True, 'query_string_length': 2061, 'query_plans_per_minute': 2378}, 'read_pool_config': {'node_count': 1070}, 'ip_address': 'ip_address_value', 'reconciling': True, 'etag': 'etag_value', 'annotations': {}, 'update_policy': {'mode': 1}, 'client_connection_config': {'require_connectors': True, 'ssl_config': {'ssl_mode': 1, 'ca_source': 1}}}
    test_field = service.CreateSecondaryInstanceRequest.meta.fields['instance']

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
        response = client.create_secondary_instance(request)
    assert response.operation.name == 'operations/spam'

def test_create_secondary_instance_rest_required_fields(request_type=service.CreateSecondaryInstanceRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['instance_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'instanceId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_secondary_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'instanceId' in jsonified_request
    assert jsonified_request['instanceId'] == request_init['instance_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['instanceId'] = 'instance_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_secondary_instance._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('instance_id', 'request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'instanceId' in jsonified_request
    assert jsonified_request['instanceId'] == 'instance_id_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_secondary_instance(request)
            expected_params = [('instanceId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_secondary_instance_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_secondary_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(('instanceId', 'requestId', 'validateOnly')) & set(('parent', 'instanceId', 'instance'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_secondary_instance_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_create_secondary_instance') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_create_secondary_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateSecondaryInstanceRequest.pb(service.CreateSecondaryInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.CreateSecondaryInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_secondary_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_secondary_instance_rest_bad_request(transport: str='rest', request_type=service.CreateSecondaryInstanceRequest):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_secondary_instance(request)

def test_create_secondary_instance_rest_flattened():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
        mock_args = dict(parent='parent_value', instance=resources.Instance(name='name_value'), instance_id='instance_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_secondary_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*/clusters/*}/instances:createsecondary' % client.transport._host, args[1])

def test_create_secondary_instance_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_secondary_instance(service.CreateSecondaryInstanceRequest(), parent='parent_value', instance=resources.Instance(name='name_value'), instance_id='instance_id_value')

def test_create_secondary_instance_rest_error():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.BatchCreateInstancesRequest, dict])
def test_batch_create_instances_rest(request_type):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
    request_init['requests'] = {'create_instance_requests': [{'parent': 'parent_value', 'instance_id': 'instance_id_value', 'instance': {'name': 'name_value', 'display_name': 'display_name_value', 'uid': 'uid_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'labels': {}, 'state': 1, 'instance_type': 1, 'machine_config': {'cpu_count': 976}, 'availability_type': 1, 'gce_zone': 'gce_zone_value', 'database_flags': {}, 'writable_node': {'zone_id': 'zone_id_value', 'id': 'id_value', 'ip': 'ip_value', 'state': 'state_value'}, 'nodes': {}, 'query_insights_config': {'record_application_tags': True, 'record_client_address': True, 'query_string_length': 2061, 'query_plans_per_minute': 2378}, 'read_pool_config': {'node_count': 1070}, 'ip_address': 'ip_address_value', 'reconciling': True, 'etag': 'etag_value', 'annotations': {}, 'update_policy': {'mode': 1}, 'client_connection_config': {'require_connectors': True, 'ssl_config': {'ssl_mode': 1, 'ca_source': 1}}}, 'request_id': 'request_id_value', 'validate_only': True}]}
    test_field = service.BatchCreateInstancesRequest.meta.fields['requests']

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
    for (field, value) in request_init['requests'].items():
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
                for i in range(0, len(request_init['requests'][field])):
                    del request_init['requests'][field][i][subfield]
            else:
                del request_init['requests'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_create_instances(request)
    assert response.operation.name == 'operations/spam'

def test_batch_create_instances_rest_required_fields(request_type=service.BatchCreateInstancesRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_create_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_create_instances._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.batch_create_instances(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_create_instances_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_create_instances._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('parent', 'requests'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_create_instances_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_batch_create_instances') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_batch_create_instances') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.BatchCreateInstancesRequest.pb(service.BatchCreateInstancesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.BatchCreateInstancesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.batch_create_instances(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_create_instances_rest_bad_request(transport: str='rest', request_type=service.BatchCreateInstancesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_create_instances(request)

def test_batch_create_instances_rest_error():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateInstanceRequest, dict])
def test_update_instance_rest(request_type):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'instance': {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}}
    request_init['instance'] = {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4', 'display_name': 'display_name_value', 'uid': 'uid_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'labels': {}, 'state': 1, 'instance_type': 1, 'machine_config': {'cpu_count': 976}, 'availability_type': 1, 'gce_zone': 'gce_zone_value', 'database_flags': {}, 'writable_node': {'zone_id': 'zone_id_value', 'id': 'id_value', 'ip': 'ip_value', 'state': 'state_value'}, 'nodes': {}, 'query_insights_config': {'record_application_tags': True, 'record_client_address': True, 'query_string_length': 2061, 'query_plans_per_minute': 2378}, 'read_pool_config': {'node_count': 1070}, 'ip_address': 'ip_address_value', 'reconciling': True, 'etag': 'etag_value', 'annotations': {}, 'update_policy': {'mode': 1}, 'client_connection_config': {'require_connectors': True, 'ssl_config': {'ssl_mode': 1, 'ca_source': 1}}}
    test_field = service.UpdateInstanceRequest.meta.fields['instance']

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

def test_update_instance_rest_required_fields(request_type=service.UpdateInstanceRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_instance._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'request_id', 'update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'requestId', 'updateMask', 'validateOnly')) & set(('instance',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_instance_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_update_instance') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_update_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateInstanceRequest.pb(service.UpdateInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.UpdateInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_instance_rest_bad_request(transport: str='rest', request_type=service.UpdateInstanceRequest):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'instance': {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}}
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'instance': {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}}
        mock_args = dict(instance=resources.Instance(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{instance.name=projects/*/locations/*/clusters/*/instances/*}' % client.transport._host, args[1])

def test_update_instance_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_instance(service.UpdateInstanceRequest(), instance=resources.Instance(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_instance_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DeleteInstanceRequest, dict])
def test_delete_instance_rest(request_type):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_instance(request)
    assert response.operation.name == 'operations/spam'

def test_delete_instance_rest_required_fields(request_type=service.DeleteInstanceRequest):
    if False:
        print('Hello World!')
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_instance._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('etag', 'request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_instance_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(('etag', 'requestId', 'validateOnly')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_instance_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_delete_instance') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_delete_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DeleteInstanceRequest.pb(service.DeleteInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.DeleteInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_instance_rest_bad_request(transport: str='rest', request_type=service.DeleteInstanceRequest):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_instance(request)

def test_delete_instance_rest_flattened():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/clusters/*/instances/*}' % client.transport._host, args[1])

def test_delete_instance_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_instance(service.DeleteInstanceRequest(), name='name_value')

def test_delete_instance_rest_error():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.FailoverInstanceRequest, dict])
def test_failover_instance_rest(request_type):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.failover_instance(request)
    assert response.operation.name == 'operations/spam'

def test_failover_instance_rest_required_fields(request_type=service.FailoverInstanceRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).failover_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).failover_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.failover_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_failover_instance_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.failover_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_failover_instance_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_failover_instance') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_failover_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.FailoverInstanceRequest.pb(service.FailoverInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.FailoverInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.failover_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_failover_instance_rest_bad_request(transport: str='rest', request_type=service.FailoverInstanceRequest):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.failover_instance(request)

def test_failover_instance_rest_flattened():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.failover_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/clusters/*/instances/*}:failover' % client.transport._host, args[1])

def test_failover_instance_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.failover_instance(service.FailoverInstanceRequest(), name='name_value')

def test_failover_instance_rest_error():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.InjectFaultRequest, dict])
def test_inject_fault_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.inject_fault(request)
    assert response.operation.name == 'operations/spam'

def test_inject_fault_rest_required_fields(request_type=service.InjectFaultRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).inject_fault._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).inject_fault._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.inject_fault(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_inject_fault_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.inject_fault._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('faultType', 'name'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_inject_fault_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_inject_fault') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_inject_fault') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.InjectFaultRequest.pb(service.InjectFaultRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.InjectFaultRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.inject_fault(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_inject_fault_rest_bad_request(transport: str='rest', request_type=service.InjectFaultRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.inject_fault(request)

def test_inject_fault_rest_flattened():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}
        mock_args = dict(fault_type=service.InjectFaultRequest.FaultType.STOP_VM, name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.inject_fault(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/clusters/*/instances/*}:injectFault' % client.transport._host, args[1])

def test_inject_fault_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.inject_fault(service.InjectFaultRequest(), fault_type=service.InjectFaultRequest.FaultType.STOP_VM, name='name_value')

def test_inject_fault_rest_error():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.RestartInstanceRequest, dict])
def test_restart_instance_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.restart_instance(request)
    assert response.operation.name == 'operations/spam'

def test_restart_instance_rest_required_fields(request_type=service.RestartInstanceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).restart_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).restart_instance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.restart_instance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_restart_instance_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.restart_instance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_restart_instance_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_restart_instance') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_restart_instance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.RestartInstanceRequest.pb(service.RestartInstanceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.RestartInstanceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.restart_instance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_restart_instance_rest_bad_request(transport: str='rest', request_type=service.RestartInstanceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.restart_instance(request)

def test_restart_instance_rest_flattened():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.restart_instance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/clusters/*/instances/*}:restart' % client.transport._host, args[1])

def test_restart_instance_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.restart_instance(service.RestartInstanceRequest(), name='name_value')

def test_restart_instance_rest_error():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListBackupsRequest, dict])
def test_list_backups_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListBackupsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListBackupsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_backups(request)
    assert isinstance(response, pagers.ListBackupsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_backups_rest_required_fields(request_type=service.ListBackupsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_backups._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_backups._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListBackupsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListBackupsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_backups(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_backups_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_backups._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_backups_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_list_backups') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_list_backups') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListBackupsRequest.pb(service.ListBackupsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListBackupsResponse.to_json(service.ListBackupsResponse())
        request = service.ListBackupsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListBackupsResponse()
        client.list_backups(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_backups_rest_bad_request(transport: str='rest', request_type=service.ListBackupsRequest):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_backups(request)

def test_list_backups_rest_flattened():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListBackupsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListBackupsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_backups(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*}/backups' % client.transport._host, args[1])

def test_list_backups_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_backups(service.ListBackupsRequest(), parent='parent_value')

def test_list_backups_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListBackupsResponse(backups=[resources.Backup(), resources.Backup(), resources.Backup()], next_page_token='abc'), service.ListBackupsResponse(backups=[], next_page_token='def'), service.ListBackupsResponse(backups=[resources.Backup()], next_page_token='ghi'), service.ListBackupsResponse(backups=[resources.Backup(), resources.Backup()]))
        response = response + response
        response = tuple((service.ListBackupsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_backups(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Backup) for i in results))
        pages = list(client.list_backups(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetBackupRequest, dict])
def test_get_backup_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/backups/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Backup(name='name_value', display_name='display_name_value', uid='uid_value', state=resources.Backup.State.READY, type_=resources.Backup.Type.ON_DEMAND, description='description_value', cluster_uid='cluster_uid_value', cluster_name='cluster_name_value', reconciling=True, etag='etag_value', size_bytes=1089, database_version=resources.DatabaseVersion.POSTGRES_13)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Backup.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_backup(request)
    assert isinstance(response, resources.Backup)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.uid == 'uid_value'
    assert response.state == resources.Backup.State.READY
    assert response.type_ == resources.Backup.Type.ON_DEMAND
    assert response.description == 'description_value'
    assert response.cluster_uid == 'cluster_uid_value'
    assert response.cluster_name == 'cluster_name_value'
    assert response.reconciling is True
    assert response.etag == 'etag_value'
    assert response.size_bytes == 1089
    assert response.database_version == resources.DatabaseVersion.POSTGRES_13

def test_get_backup_rest_required_fields(request_type=service.GetBackupRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_backup._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_backup._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Backup()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Backup.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_backup(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_backup_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_backup._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_backup_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_get_backup') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_get_backup') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetBackupRequest.pb(service.GetBackupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Backup.to_json(resources.Backup())
        request = service.GetBackupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Backup()
        client.get_backup(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_backup_rest_bad_request(transport: str='rest', request_type=service.GetBackupRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/backups/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_backup(request)

def test_get_backup_rest_flattened():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Backup()
        sample_request = {'name': 'projects/sample1/locations/sample2/backups/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Backup.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_backup(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/backups/*}' % client.transport._host, args[1])

def test_get_backup_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_backup(service.GetBackupRequest(), name='name_value')

def test_get_backup_rest_error():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateBackupRequest, dict])
def test_create_backup_rest(request_type):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['backup'] = {'name': 'name_value', 'display_name': 'display_name_value', 'uid': 'uid_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'labels': {}, 'state': 1, 'type_': 1, 'description': 'description_value', 'cluster_uid': 'cluster_uid_value', 'cluster_name': 'cluster_name_value', 'reconciling': True, 'encryption_config': {'kms_key_name': 'kms_key_name_value'}, 'encryption_info': {'encryption_type': 1, 'kms_key_versions': ['kms_key_versions_value1', 'kms_key_versions_value2']}, 'etag': 'etag_value', 'annotations': {}, 'size_bytes': 1089, 'expiry_time': {}, 'expiry_quantity': {'retention_count': 1632, 'total_retention_count': 2275}, 'database_version': 1}
    test_field = service.CreateBackupRequest.meta.fields['backup']

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
    for (field, value) in request_init['backup'].items():
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
                for i in range(0, len(request_init['backup'][field])):
                    del request_init['backup'][field][i][subfield]
            else:
                del request_init['backup'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_backup(request)
    assert response.operation.name == 'operations/spam'

def test_create_backup_rest_required_fields(request_type=service.CreateBackupRequest):
    if False:
        print('Hello World!')
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['backup_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'backupId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_backup._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'backupId' in jsonified_request
    assert jsonified_request['backupId'] == request_init['backup_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['backupId'] = 'backup_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_backup._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('backup_id', 'request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'backupId' in jsonified_request
    assert jsonified_request['backupId'] == 'backup_id_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_backup(request)
            expected_params = [('backupId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_backup_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_backup._get_unset_required_fields({})
    assert set(unset_fields) == set(('backupId', 'requestId', 'validateOnly')) & set(('parent', 'backupId', 'backup'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_backup_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_create_backup') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_create_backup') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateBackupRequest.pb(service.CreateBackupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.CreateBackupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_backup(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_backup_rest_bad_request(transport: str='rest', request_type=service.CreateBackupRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_backup(request)

def test_create_backup_rest_flattened():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', backup=resources.Backup(name='name_value'), backup_id='backup_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_backup(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*}/backups' % client.transport._host, args[1])

def test_create_backup_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_backup(service.CreateBackupRequest(), parent='parent_value', backup=resources.Backup(name='name_value'), backup_id='backup_id_value')

def test_create_backup_rest_error():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateBackupRequest, dict])
def test_update_backup_rest(request_type):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'backup': {'name': 'projects/sample1/locations/sample2/backups/sample3'}}
    request_init['backup'] = {'name': 'projects/sample1/locations/sample2/backups/sample3', 'display_name': 'display_name_value', 'uid': 'uid_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'delete_time': {}, 'labels': {}, 'state': 1, 'type_': 1, 'description': 'description_value', 'cluster_uid': 'cluster_uid_value', 'cluster_name': 'cluster_name_value', 'reconciling': True, 'encryption_config': {'kms_key_name': 'kms_key_name_value'}, 'encryption_info': {'encryption_type': 1, 'kms_key_versions': ['kms_key_versions_value1', 'kms_key_versions_value2']}, 'etag': 'etag_value', 'annotations': {}, 'size_bytes': 1089, 'expiry_time': {}, 'expiry_quantity': {'retention_count': 1632, 'total_retention_count': 2275}, 'database_version': 1}
    test_field = service.UpdateBackupRequest.meta.fields['backup']

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
    for (field, value) in request_init['backup'].items():
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
                for i in range(0, len(request_init['backup'][field])):
                    del request_init['backup'][field][i][subfield]
            else:
                del request_init['backup'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_backup(request)
    assert response.operation.name == 'operations/spam'

def test_update_backup_rest_required_fields(request_type=service.UpdateBackupRequest):
    if False:
        return 10
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_backup._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_backup._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'request_id', 'update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_backup(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_backup_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_backup._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'requestId', 'updateMask', 'validateOnly')) & set(('backup',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_backup_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_update_backup') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_update_backup') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateBackupRequest.pb(service.UpdateBackupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.UpdateBackupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_backup(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_backup_rest_bad_request(transport: str='rest', request_type=service.UpdateBackupRequest):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'backup': {'name': 'projects/sample1/locations/sample2/backups/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_backup(request)

def test_update_backup_rest_flattened():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'backup': {'name': 'projects/sample1/locations/sample2/backups/sample3'}}
        mock_args = dict(backup=resources.Backup(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_backup(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{backup.name=projects/*/locations/*/backups/*}' % client.transport._host, args[1])

def test_update_backup_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_backup(service.UpdateBackupRequest(), backup=resources.Backup(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_backup_rest_error():
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DeleteBackupRequest, dict])
def test_delete_backup_rest(request_type):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/backups/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_backup(request)
    assert response.operation.name == 'operations/spam'

def test_delete_backup_rest_required_fields(request_type=service.DeleteBackupRequest):
    if False:
        print('Hello World!')
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_backup._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_backup._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('etag', 'request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_backup(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_backup_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_backup._get_unset_required_fields({})
    assert set(unset_fields) == set(('etag', 'requestId', 'validateOnly')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_backup_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_delete_backup') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_delete_backup') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DeleteBackupRequest.pb(service.DeleteBackupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.DeleteBackupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_backup(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_backup_rest_bad_request(transport: str='rest', request_type=service.DeleteBackupRequest):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/backups/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_backup(request)

def test_delete_backup_rest_flattened():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/backups/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_backup(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/backups/*}' % client.transport._host, args[1])

def test_delete_backup_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_backup(service.DeleteBackupRequest(), name='name_value')

def test_delete_backup_rest_error():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListSupportedDatabaseFlagsRequest, dict])
def test_list_supported_database_flags_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListSupportedDatabaseFlagsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListSupportedDatabaseFlagsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_supported_database_flags(request)
    assert isinstance(response, pagers.ListSupportedDatabaseFlagsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_supported_database_flags_rest_required_fields(request_type=service.ListSupportedDatabaseFlagsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_supported_database_flags._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_supported_database_flags._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListSupportedDatabaseFlagsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListSupportedDatabaseFlagsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_supported_database_flags(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_supported_database_flags_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_supported_database_flags._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_supported_database_flags_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_list_supported_database_flags') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_list_supported_database_flags') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListSupportedDatabaseFlagsRequest.pb(service.ListSupportedDatabaseFlagsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListSupportedDatabaseFlagsResponse.to_json(service.ListSupportedDatabaseFlagsResponse())
        request = service.ListSupportedDatabaseFlagsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListSupportedDatabaseFlagsResponse()
        client.list_supported_database_flags(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_supported_database_flags_rest_bad_request(transport: str='rest', request_type=service.ListSupportedDatabaseFlagsRequest):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_supported_database_flags(request)

def test_list_supported_database_flags_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListSupportedDatabaseFlagsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListSupportedDatabaseFlagsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_supported_database_flags(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*}/supportedDatabaseFlags' % client.transport._host, args[1])

def test_list_supported_database_flags_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_supported_database_flags(service.ListSupportedDatabaseFlagsRequest(), parent='parent_value')

def test_list_supported_database_flags_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[resources.SupportedDatabaseFlag(), resources.SupportedDatabaseFlag(), resources.SupportedDatabaseFlag()], next_page_token='abc'), service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[], next_page_token='def'), service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[resources.SupportedDatabaseFlag()], next_page_token='ghi'), service.ListSupportedDatabaseFlagsResponse(supported_database_flags=[resources.SupportedDatabaseFlag(), resources.SupportedDatabaseFlag()]))
        response = response + response
        response = tuple((service.ListSupportedDatabaseFlagsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_supported_database_flags(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.SupportedDatabaseFlag) for i in results))
        pages = list(client.list_supported_database_flags(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GenerateClientCertificateRequest, dict])
def test_generate_client_certificate_rest(request_type):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.GenerateClientCertificateResponse(pem_certificate='pem_certificate_value', pem_certificate_chain=['pem_certificate_chain_value'], ca_cert='ca_cert_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = service.GenerateClientCertificateResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.generate_client_certificate(request)
    assert isinstance(response, service.GenerateClientCertificateResponse)
    assert response.pem_certificate == 'pem_certificate_value'
    assert response.pem_certificate_chain == ['pem_certificate_chain_value']
    assert response.ca_cert == 'ca_cert_value'

def test_generate_client_certificate_rest_required_fields(request_type=service.GenerateClientCertificateRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_client_certificate._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_client_certificate._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.GenerateClientCertificateResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.GenerateClientCertificateResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.generate_client_certificate(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_generate_client_certificate_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.generate_client_certificate._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_generate_client_certificate_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_generate_client_certificate') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_generate_client_certificate') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GenerateClientCertificateRequest.pb(service.GenerateClientCertificateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.GenerateClientCertificateResponse.to_json(service.GenerateClientCertificateResponse())
        request = service.GenerateClientCertificateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.GenerateClientCertificateResponse()
        client.generate_client_certificate(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_generate_client_certificate_rest_bad_request(transport: str='rest', request_type=service.GenerateClientCertificateRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.generate_client_certificate(request)

def test_generate_client_certificate_rest_flattened():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.GenerateClientCertificateResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.GenerateClientCertificateResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.generate_client_certificate(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*/clusters/*}:generateClientCertificate' % client.transport._host, args[1])

def test_generate_client_certificate_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.generate_client_certificate(service.GenerateClientCertificateRequest(), parent='parent_value')

def test_generate_client_certificate_rest_error():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.GetConnectionInfoRequest, dict])
def test_get_connection_info_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.ConnectionInfo(name='name_value', ip_address='ip_address_value', pem_certificate_chain=['pem_certificate_chain_value'], instance_uid='instance_uid_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.ConnectionInfo.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_connection_info(request)
    assert isinstance(response, resources.ConnectionInfo)
    assert response.name == 'name_value'
    assert response.ip_address == 'ip_address_value'
    assert response.pem_certificate_chain == ['pem_certificate_chain_value']
    assert response.instance_uid == 'instance_uid_value'

def test_get_connection_info_rest_required_fields(request_type=service.GetConnectionInfoRequest):
    if False:
        return 10
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_connection_info._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_connection_info._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.ConnectionInfo()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.ConnectionInfo.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_connection_info(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_connection_info_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_connection_info._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_connection_info_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_get_connection_info') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_get_connection_info') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetConnectionInfoRequest.pb(service.GetConnectionInfoRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.ConnectionInfo.to_json(resources.ConnectionInfo())
        request = service.GetConnectionInfoRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.ConnectionInfo()
        client.get_connection_info(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_connection_info_rest_bad_request(transport: str='rest', request_type=service.GetConnectionInfoRequest):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_connection_info(request)

def test_get_connection_info_rest_flattened():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.ConnectionInfo()
        sample_request = {'parent': 'projects/sample1/locations/sample2/clusters/sample3/instances/sample4'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.ConnectionInfo.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_connection_info(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*/clusters/*/instances/*}/connectionInfo' % client.transport._host, args[1])

def test_get_connection_info_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_connection_info(service.GetConnectionInfoRequest(), parent='parent_value')

def test_get_connection_info_rest_error():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListUsersRequest, dict])
def test_list_users_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListUsersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListUsersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_users(request)
    assert isinstance(response, pagers.ListUsersPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_users_rest_required_fields(request_type=service.ListUsersRequest):
    if False:
        print('Hello World!')
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_users._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_users._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListUsersResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListUsersResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_users(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_users_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_users._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_users_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_list_users') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_list_users') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListUsersRequest.pb(service.ListUsersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListUsersResponse.to_json(service.ListUsersResponse())
        request = service.ListUsersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListUsersResponse()
        client.list_users(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_users_rest_bad_request(transport: str='rest', request_type=service.ListUsersRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_users(request)

def test_list_users_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListUsersResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListUsersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_users(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*/clusters/*}/users' % client.transport._host, args[1])

def test_list_users_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_users(service.ListUsersRequest(), parent='parent_value')

def test_list_users_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListUsersResponse(users=[resources.User(), resources.User(), resources.User()], next_page_token='abc'), service.ListUsersResponse(users=[], next_page_token='def'), service.ListUsersResponse(users=[resources.User()], next_page_token='ghi'), service.ListUsersResponse(users=[resources.User(), resources.User()]))
        response = response + response
        response = tuple((service.ListUsersResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
        pager = client.list_users(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.User) for i in results))
        pages = list(client.list_users(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetUserRequest, dict])
def test_get_user_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3/users/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.User(name='name_value', password='password_value', database_roles=['database_roles_value'], user_type=resources.User.UserType.ALLOYDB_BUILT_IN)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.User.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_user(request)
    assert isinstance(response, resources.User)
    assert response.name == 'name_value'
    assert response.password == 'password_value'
    assert response.database_roles == ['database_roles_value']
    assert response.user_type == resources.User.UserType.ALLOYDB_BUILT_IN

def test_get_user_rest_required_fields(request_type=service.GetUserRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_user._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_user._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.User()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.User.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_user(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_user_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_user._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_user_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_get_user') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_get_user') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetUserRequest.pb(service.GetUserRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.User.to_json(resources.User())
        request = service.GetUserRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.User()
        client.get_user(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_user_rest_bad_request(transport: str='rest', request_type=service.GetUserRequest):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3/users/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_user(request)

def test_get_user_rest_flattened():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.User()
        sample_request = {'name': 'projects/sample1/locations/sample2/clusters/sample3/users/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.User.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_user(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/clusters/*/users/*}' % client.transport._host, args[1])

def test_get_user_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_user(service.GetUserRequest(), name='name_value')

def test_get_user_rest_error():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateUserRequest, dict])
def test_create_user_rest(request_type):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
    request_init['user'] = {'name': 'name_value', 'password': 'password_value', 'database_roles': ['database_roles_value1', 'database_roles_value2'], 'user_type': 1}
    test_field = service.CreateUserRequest.meta.fields['user']

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
    for (field, value) in request_init['user'].items():
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
                for i in range(0, len(request_init['user'][field])):
                    del request_init['user'][field][i][subfield]
            else:
                del request_init['user'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.User(name='name_value', password='password_value', database_roles=['database_roles_value'], user_type=resources.User.UserType.ALLOYDB_BUILT_IN)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.User.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_user(request)
    assert isinstance(response, resources.User)
    assert response.name == 'name_value'
    assert response.password == 'password_value'
    assert response.database_roles == ['database_roles_value']
    assert response.user_type == resources.User.UserType.ALLOYDB_BUILT_IN

def test_create_user_rest_required_fields(request_type=service.CreateUserRequest):
    if False:
        print('Hello World!')
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['user_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'userId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_user._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'userId' in jsonified_request
    assert jsonified_request['userId'] == request_init['user_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['userId'] = 'user_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_user._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'user_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'userId' in jsonified_request
    assert jsonified_request['userId'] == 'user_id_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.User()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.User.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_user(request)
            expected_params = [('userId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_user_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_user._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'userId', 'validateOnly')) & set(('parent', 'userId', 'user'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_user_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_create_user') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_create_user') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateUserRequest.pb(service.CreateUserRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.User.to_json(resources.User())
        request = service.CreateUserRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.User()
        client.create_user(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_user_rest_bad_request(transport: str='rest', request_type=service.CreateUserRequest):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_user(request)

def test_create_user_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.User()
        sample_request = {'parent': 'projects/sample1/locations/sample2/clusters/sample3'}
        mock_args = dict(parent='parent_value', user=resources.User(name='name_value'), user_id='user_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.User.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_user(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{parent=projects/*/locations/*/clusters/*}/users' % client.transport._host, args[1])

def test_create_user_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_user(service.CreateUserRequest(), parent='parent_value', user=resources.User(name='name_value'), user_id='user_id_value')

def test_create_user_rest_error():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateUserRequest, dict])
def test_update_user_rest(request_type):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'user': {'name': 'projects/sample1/locations/sample2/clusters/sample3/users/sample4'}}
    request_init['user'] = {'name': 'projects/sample1/locations/sample2/clusters/sample3/users/sample4', 'password': 'password_value', 'database_roles': ['database_roles_value1', 'database_roles_value2'], 'user_type': 1}
    test_field = service.UpdateUserRequest.meta.fields['user']

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
    for (field, value) in request_init['user'].items():
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
                for i in range(0, len(request_init['user'][field])):
                    del request_init['user'][field][i][subfield]
            else:
                del request_init['user'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.User(name='name_value', password='password_value', database_roles=['database_roles_value'], user_type=resources.User.UserType.ALLOYDB_BUILT_IN)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.User.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_user(request)
    assert isinstance(response, resources.User)
    assert response.name == 'name_value'
    assert response.password == 'password_value'
    assert response.database_roles == ['database_roles_value']
    assert response.user_type == resources.User.UserType.ALLOYDB_BUILT_IN

def test_update_user_rest_required_fields(request_type=service.UpdateUserRequest):
    if False:
        return 10
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_user._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_user._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'request_id', 'update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.User()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.User.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_user(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_user_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_user._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'requestId', 'updateMask', 'validateOnly')) & set(('user',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_user_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'post_update_user') as post, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_update_user') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateUserRequest.pb(service.UpdateUserRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.User.to_json(resources.User())
        request = service.UpdateUserRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.User()
        client.update_user(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_user_rest_bad_request(transport: str='rest', request_type=service.UpdateUserRequest):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'user': {'name': 'projects/sample1/locations/sample2/clusters/sample3/users/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_user(request)

def test_update_user_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.User()
        sample_request = {'user': {'name': 'projects/sample1/locations/sample2/clusters/sample3/users/sample4'}}
        mock_args = dict(user=resources.User(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.User.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_user(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{user.name=projects/*/locations/*/clusters/*/users/*}' % client.transport._host, args[1])

def test_update_user_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_user(service.UpdateUserRequest(), user=resources.User(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_user_rest_error():
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DeleteUserRequest, dict])
def test_delete_user_rest(request_type):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3/users/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_user(request)
    assert response is None

def test_delete_user_rest_required_fields(request_type=service.DeleteUserRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AlloyDBAdminRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_user._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_user._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_user(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_user_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_user._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'validateOnly')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_user_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.AlloyDBAdminRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AlloyDBAdminRestInterceptor())
    client = AlloyDBAdminClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AlloyDBAdminRestInterceptor, 'pre_delete_user') as pre:
        pre.assert_not_called()
        pb_message = service.DeleteUserRequest.pb(service.DeleteUserRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = service.DeleteUserRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_user(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_user_rest_bad_request(transport: str='rest', request_type=service.DeleteUserRequest):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/clusters/sample3/users/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_user(request)

def test_delete_user_rest_flattened():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/clusters/sample3/users/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_user(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta/{name=projects/*/locations/*/clusters/*/users/*}' % client.transport._host, args[1])

def test_delete_user_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_user(service.DeleteUserRequest(), name='name_value')

def test_delete_user_rest_error():
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.AlloyDBAdminGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.AlloyDBAdminGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AlloyDBAdminClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.AlloyDBAdminGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AlloyDBAdminClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AlloyDBAdminClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.AlloyDBAdminGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AlloyDBAdminClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.AlloyDBAdminGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = AlloyDBAdminClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.AlloyDBAdminGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.AlloyDBAdminGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.AlloyDBAdminGrpcTransport, transports.AlloyDBAdminGrpcAsyncIOTransport, transports.AlloyDBAdminRestTransport])
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
        for i in range(10):
            print('nop')
    transport = AlloyDBAdminClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.AlloyDBAdminGrpcTransport)

def test_alloy_db_admin_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.AlloyDBAdminTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_alloy_db_admin_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.alloydb_v1beta.services.alloy_db_admin.transports.AlloyDBAdminTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.AlloyDBAdminTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_clusters', 'get_cluster', 'create_cluster', 'update_cluster', 'delete_cluster', 'promote_cluster', 'restore_cluster', 'create_secondary_cluster', 'list_instances', 'get_instance', 'create_instance', 'create_secondary_instance', 'batch_create_instances', 'update_instance', 'delete_instance', 'failover_instance', 'inject_fault', 'restart_instance', 'list_backups', 'get_backup', 'create_backup', 'update_backup', 'delete_backup', 'list_supported_database_flags', 'generate_client_certificate', 'get_connection_info', 'list_users', 'get_user', 'create_user', 'update_user', 'delete_user', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_alloy_db_admin_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.alloydb_v1beta.services.alloy_db_admin.transports.AlloyDBAdminTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AlloyDBAdminTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_alloy_db_admin_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.alloydb_v1beta.services.alloy_db_admin.transports.AlloyDBAdminTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AlloyDBAdminTransport()
        adc.assert_called_once()

def test_alloy_db_admin_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        AlloyDBAdminClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.AlloyDBAdminGrpcTransport, transports.AlloyDBAdminGrpcAsyncIOTransport])
def test_alloy_db_admin_transport_auth_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.AlloyDBAdminGrpcTransport, transports.AlloyDBAdminGrpcAsyncIOTransport, transports.AlloyDBAdminRestTransport])
def test_alloy_db_admin_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.AlloyDBAdminGrpcTransport, grpc_helpers), (transports.AlloyDBAdminGrpcAsyncIOTransport, grpc_helpers_async)])
def test_alloy_db_admin_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('alloydb.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='alloydb.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.AlloyDBAdminGrpcTransport, transports.AlloyDBAdminGrpcAsyncIOTransport])
def test_alloy_db_admin_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_alloy_db_admin_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.AlloyDBAdminRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_alloy_db_admin_rest_lro_client():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_alloy_db_admin_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='alloydb.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('alloydb.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://alloydb.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_alloy_db_admin_host_with_port(transport_name):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='alloydb.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('alloydb.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://alloydb.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_alloy_db_admin_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = AlloyDBAdminClient(credentials=creds1, transport=transport_name)
    client2 = AlloyDBAdminClient(credentials=creds2, transport=transport_name)
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
    session1 = client1.transport.promote_cluster._session
    session2 = client2.transport.promote_cluster._session
    assert session1 != session2
    session1 = client1.transport.restore_cluster._session
    session2 = client2.transport.restore_cluster._session
    assert session1 != session2
    session1 = client1.transport.create_secondary_cluster._session
    session2 = client2.transport.create_secondary_cluster._session
    assert session1 != session2
    session1 = client1.transport.list_instances._session
    session2 = client2.transport.list_instances._session
    assert session1 != session2
    session1 = client1.transport.get_instance._session
    session2 = client2.transport.get_instance._session
    assert session1 != session2
    session1 = client1.transport.create_instance._session
    session2 = client2.transport.create_instance._session
    assert session1 != session2
    session1 = client1.transport.create_secondary_instance._session
    session2 = client2.transport.create_secondary_instance._session
    assert session1 != session2
    session1 = client1.transport.batch_create_instances._session
    session2 = client2.transport.batch_create_instances._session
    assert session1 != session2
    session1 = client1.transport.update_instance._session
    session2 = client2.transport.update_instance._session
    assert session1 != session2
    session1 = client1.transport.delete_instance._session
    session2 = client2.transport.delete_instance._session
    assert session1 != session2
    session1 = client1.transport.failover_instance._session
    session2 = client2.transport.failover_instance._session
    assert session1 != session2
    session1 = client1.transport.inject_fault._session
    session2 = client2.transport.inject_fault._session
    assert session1 != session2
    session1 = client1.transport.restart_instance._session
    session2 = client2.transport.restart_instance._session
    assert session1 != session2
    session1 = client1.transport.list_backups._session
    session2 = client2.transport.list_backups._session
    assert session1 != session2
    session1 = client1.transport.get_backup._session
    session2 = client2.transport.get_backup._session
    assert session1 != session2
    session1 = client1.transport.create_backup._session
    session2 = client2.transport.create_backup._session
    assert session1 != session2
    session1 = client1.transport.update_backup._session
    session2 = client2.transport.update_backup._session
    assert session1 != session2
    session1 = client1.transport.delete_backup._session
    session2 = client2.transport.delete_backup._session
    assert session1 != session2
    session1 = client1.transport.list_supported_database_flags._session
    session2 = client2.transport.list_supported_database_flags._session
    assert session1 != session2
    session1 = client1.transport.generate_client_certificate._session
    session2 = client2.transport.generate_client_certificate._session
    assert session1 != session2
    session1 = client1.transport.get_connection_info._session
    session2 = client2.transport.get_connection_info._session
    assert session1 != session2
    session1 = client1.transport.list_users._session
    session2 = client2.transport.list_users._session
    assert session1 != session2
    session1 = client1.transport.get_user._session
    session2 = client2.transport.get_user._session
    assert session1 != session2
    session1 = client1.transport.create_user._session
    session2 = client2.transport.create_user._session
    assert session1 != session2
    session1 = client1.transport.update_user._session
    session2 = client2.transport.update_user._session
    assert session1 != session2
    session1 = client1.transport.delete_user._session
    session2 = client2.transport.delete_user._session
    assert session1 != session2

def test_alloy_db_admin_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AlloyDBAdminGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_alloy_db_admin_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AlloyDBAdminGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.AlloyDBAdminGrpcTransport, transports.AlloyDBAdminGrpcAsyncIOTransport])
def test_alloy_db_admin_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.AlloyDBAdminGrpcTransport, transports.AlloyDBAdminGrpcAsyncIOTransport])
def test_alloy_db_admin_transport_channel_mtls_with_adc(transport_class):
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

def test_alloy_db_admin_grpc_lro_client():
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_alloy_db_admin_grpc_lro_async_client():
    if False:
        i = 10
        return i + 15
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_backup_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    backup = 'whelk'
    expected = 'projects/{project}/locations/{location}/backups/{backup}'.format(project=project, location=location, backup=backup)
    actual = AlloyDBAdminClient.backup_path(project, location, backup)
    assert expected == actual

def test_parse_backup_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus', 'location': 'oyster', 'backup': 'nudibranch'}
    path = AlloyDBAdminClient.backup_path(**expected)
    actual = AlloyDBAdminClient.parse_backup_path(path)
    assert expected == actual

def test_cluster_path():
    if False:
        return 10
    project = 'cuttlefish'
    location = 'mussel'
    cluster = 'winkle'
    expected = 'projects/{project}/locations/{location}/clusters/{cluster}'.format(project=project, location=location, cluster=cluster)
    actual = AlloyDBAdminClient.cluster_path(project, location, cluster)
    assert expected == actual

def test_parse_cluster_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'nautilus', 'location': 'scallop', 'cluster': 'abalone'}
    path = AlloyDBAdminClient.cluster_path(**expected)
    actual = AlloyDBAdminClient.parse_cluster_path(path)
    assert expected == actual

def test_connection_info_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    cluster = 'whelk'
    instance = 'octopus'
    expected = 'projects/{project}/locations/{location}/clusters/{cluster}/instances/{instance}/connectionInfo'.format(project=project, location=location, cluster=cluster, instance=instance)
    actual = AlloyDBAdminClient.connection_info_path(project, location, cluster, instance)
    assert expected == actual

def test_parse_connection_info_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'oyster', 'location': 'nudibranch', 'cluster': 'cuttlefish', 'instance': 'mussel'}
    path = AlloyDBAdminClient.connection_info_path(**expected)
    actual = AlloyDBAdminClient.parse_connection_info_path(path)
    assert expected == actual

def test_crypto_key_version_path():
    if False:
        i = 10
        return i + 15
    project = 'winkle'
    location = 'nautilus'
    key_ring = 'scallop'
    crypto_key = 'abalone'
    crypto_key_version = 'squid'
    expected = 'projects/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}/cryptoKeyVersions/{crypto_key_version}'.format(project=project, location=location, key_ring=key_ring, crypto_key=crypto_key, crypto_key_version=crypto_key_version)
    actual = AlloyDBAdminClient.crypto_key_version_path(project, location, key_ring, crypto_key, crypto_key_version)
    assert expected == actual

def test_parse_crypto_key_version_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'clam', 'location': 'whelk', 'key_ring': 'octopus', 'crypto_key': 'oyster', 'crypto_key_version': 'nudibranch'}
    path = AlloyDBAdminClient.crypto_key_version_path(**expected)
    actual = AlloyDBAdminClient.parse_crypto_key_version_path(path)
    assert expected == actual

def test_instance_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    location = 'mussel'
    cluster = 'winkle'
    instance = 'nautilus'
    expected = 'projects/{project}/locations/{location}/clusters/{cluster}/instances/{instance}'.format(project=project, location=location, cluster=cluster, instance=instance)
    actual = AlloyDBAdminClient.instance_path(project, location, cluster, instance)
    assert expected == actual

def test_parse_instance_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'scallop', 'location': 'abalone', 'cluster': 'squid', 'instance': 'clam'}
    path = AlloyDBAdminClient.instance_path(**expected)
    actual = AlloyDBAdminClient.parse_instance_path(path)
    assert expected == actual

def test_network_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    network = 'octopus'
    expected = 'projects/{project}/global/networks/{network}'.format(project=project, network=network)
    actual = AlloyDBAdminClient.network_path(project, network)
    assert expected == actual

def test_parse_network_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'oyster', 'network': 'nudibranch'}
    path = AlloyDBAdminClient.network_path(**expected)
    actual = AlloyDBAdminClient.parse_network_path(path)
    assert expected == actual

def test_supported_database_flag_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    location = 'mussel'
    flag = 'winkle'
    expected = 'projects/{project}/locations/{location}/flags/{flag}'.format(project=project, location=location, flag=flag)
    actual = AlloyDBAdminClient.supported_database_flag_path(project, location, flag)
    assert expected == actual

def test_parse_supported_database_flag_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nautilus', 'location': 'scallop', 'flag': 'abalone'}
    path = AlloyDBAdminClient.supported_database_flag_path(**expected)
    actual = AlloyDBAdminClient.parse_supported_database_flag_path(path)
    assert expected == actual

def test_user_path():
    if False:
        return 10
    project = 'squid'
    location = 'clam'
    cluster = 'whelk'
    user = 'octopus'
    expected = 'projects/{project}/locations/{location}/clusters/{cluster}/users/{user}'.format(project=project, location=location, cluster=cluster, user=user)
    actual = AlloyDBAdminClient.user_path(project, location, cluster, user)
    assert expected == actual

def test_parse_user_path():
    if False:
        return 10
    expected = {'project': 'oyster', 'location': 'nudibranch', 'cluster': 'cuttlefish', 'user': 'mussel'}
    path = AlloyDBAdminClient.user_path(**expected)
    actual = AlloyDBAdminClient.parse_user_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'winkle'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = AlloyDBAdminClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'nautilus'}
    path = AlloyDBAdminClient.common_billing_account_path(**expected)
    actual = AlloyDBAdminClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'scallop'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = AlloyDBAdminClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'abalone'}
    path = AlloyDBAdminClient.common_folder_path(**expected)
    actual = AlloyDBAdminClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'squid'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = AlloyDBAdminClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'clam'}
    path = AlloyDBAdminClient.common_organization_path(**expected)
    actual = AlloyDBAdminClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'whelk'
    expected = 'projects/{project}'.format(project=project)
    actual = AlloyDBAdminClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'octopus'}
    path = AlloyDBAdminClient.common_project_path(**expected)
    actual = AlloyDBAdminClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'oyster'
    location = 'nudibranch'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = AlloyDBAdminClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'cuttlefish', 'location': 'mussel'}
    path = AlloyDBAdminClient.common_location_path(**expected)
    actual = AlloyDBAdminClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.AlloyDBAdminTransport, '_prep_wrapped_messages') as prep:
        client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.AlloyDBAdminTransport, '_prep_wrapped_messages') as prep:
        transport_class = AlloyDBAdminClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_cancel_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.CancelOperationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        return 10
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = AlloyDBAdminAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = AlloyDBAdminClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(AlloyDBAdminClient, transports.AlloyDBAdminGrpcTransport), (AlloyDBAdminAsyncClient, transports.AlloyDBAdminGrpcAsyncIOTransport)])
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