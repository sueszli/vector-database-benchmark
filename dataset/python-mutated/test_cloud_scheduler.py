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
from google.cloud.location import locations_pb2
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
from google.cloud.scheduler_v1beta1.services.cloud_scheduler import CloudSchedulerAsyncClient, CloudSchedulerClient, pagers, transports
from google.cloud.scheduler_v1beta1.types import cloudscheduler
from google.cloud.scheduler_v1beta1.types import job
from google.cloud.scheduler_v1beta1.types import job as gcs_job
from google.cloud.scheduler_v1beta1.types import target

def client_cert_source_callback():
    if False:
        while True:
            i = 10
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        return 10
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        return 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert CloudSchedulerClient._get_default_mtls_endpoint(None) is None
    assert CloudSchedulerClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert CloudSchedulerClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert CloudSchedulerClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert CloudSchedulerClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert CloudSchedulerClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(CloudSchedulerClient, 'grpc'), (CloudSchedulerAsyncClient, 'grpc_asyncio'), (CloudSchedulerClient, 'rest')])
def test_cloud_scheduler_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('cloudscheduler.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudscheduler.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.CloudSchedulerGrpcTransport, 'grpc'), (transports.CloudSchedulerGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.CloudSchedulerRestTransport, 'rest')])
def test_cloud_scheduler_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(CloudSchedulerClient, 'grpc'), (CloudSchedulerAsyncClient, 'grpc_asyncio'), (CloudSchedulerClient, 'rest')])
def test_cloud_scheduler_client_from_service_account_file(client_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_file') as factory:
        factory.return_value = creds
        client = client_class.from_service_account_file('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        client = client_class.from_service_account_json('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('cloudscheduler.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudscheduler.googleapis.com')

def test_cloud_scheduler_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = CloudSchedulerClient.get_transport_class()
    available_transports = [transports.CloudSchedulerGrpcTransport, transports.CloudSchedulerRestTransport]
    assert transport in available_transports
    transport = CloudSchedulerClient.get_transport_class('grpc')
    assert transport == transports.CloudSchedulerGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CloudSchedulerClient, transports.CloudSchedulerGrpcTransport, 'grpc'), (CloudSchedulerAsyncClient, transports.CloudSchedulerGrpcAsyncIOTransport, 'grpc_asyncio'), (CloudSchedulerClient, transports.CloudSchedulerRestTransport, 'rest')])
@mock.patch.object(CloudSchedulerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudSchedulerClient))
@mock.patch.object(CloudSchedulerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudSchedulerAsyncClient))
def test_cloud_scheduler_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(CloudSchedulerClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(CloudSchedulerClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(CloudSchedulerClient, transports.CloudSchedulerGrpcTransport, 'grpc', 'true'), (CloudSchedulerAsyncClient, transports.CloudSchedulerGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (CloudSchedulerClient, transports.CloudSchedulerGrpcTransport, 'grpc', 'false'), (CloudSchedulerAsyncClient, transports.CloudSchedulerGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (CloudSchedulerClient, transports.CloudSchedulerRestTransport, 'rest', 'true'), (CloudSchedulerClient, transports.CloudSchedulerRestTransport, 'rest', 'false')])
@mock.patch.object(CloudSchedulerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudSchedulerClient))
@mock.patch.object(CloudSchedulerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudSchedulerAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_cloud_scheduler_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [CloudSchedulerClient, CloudSchedulerAsyncClient])
@mock.patch.object(CloudSchedulerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudSchedulerClient))
@mock.patch.object(CloudSchedulerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudSchedulerAsyncClient))
def test_cloud_scheduler_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CloudSchedulerClient, transports.CloudSchedulerGrpcTransport, 'grpc'), (CloudSchedulerAsyncClient, transports.CloudSchedulerGrpcAsyncIOTransport, 'grpc_asyncio'), (CloudSchedulerClient, transports.CloudSchedulerRestTransport, 'rest')])
def test_cloud_scheduler_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CloudSchedulerClient, transports.CloudSchedulerGrpcTransport, 'grpc', grpc_helpers), (CloudSchedulerAsyncClient, transports.CloudSchedulerGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (CloudSchedulerClient, transports.CloudSchedulerRestTransport, 'rest', None)])
def test_cloud_scheduler_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_cloud_scheduler_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.scheduler_v1beta1.services.cloud_scheduler.transports.CloudSchedulerGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = CloudSchedulerClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CloudSchedulerClient, transports.CloudSchedulerGrpcTransport, 'grpc', grpc_helpers), (CloudSchedulerAsyncClient, transports.CloudSchedulerGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_cloud_scheduler_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('cloudscheduler.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='cloudscheduler.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [cloudscheduler.ListJobsRequest, dict])
def test_list_jobs(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = cloudscheduler.ListJobsResponse(next_page_token='next_page_token_value')
        response = client.list_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.ListJobsRequest()
    assert isinstance(response, pagers.ListJobsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_jobs_empty_call():
    if False:
        print('Hello World!')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        client.list_jobs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.ListJobsRequest()

@pytest.mark.asyncio
async def test_list_jobs_async(transport: str='grpc_asyncio', request_type=cloudscheduler.ListJobsRequest):
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloudscheduler.ListJobsResponse(next_page_token='next_page_token_value'))
        response = await client.list_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.ListJobsRequest()
    assert isinstance(response, pagers.ListJobsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_jobs_async_from_dict():
    await test_list_jobs_async(request_type=dict)

def test_list_jobs_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudscheduler.ListJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = cloudscheduler.ListJobsResponse()
        client.list_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_jobs_field_headers_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudscheduler.ListJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloudscheduler.ListJobsResponse())
        await client.list_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_jobs_flattened():
    if False:
        print('Hello World!')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = cloudscheduler.ListJobsResponse()
        client.list_jobs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_jobs_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_jobs(cloudscheduler.ListJobsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_jobs_flattened_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = cloudscheduler.ListJobsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloudscheduler.ListJobsResponse())
        response = await client.list_jobs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_jobs_flattened_error_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_jobs(cloudscheduler.ListJobsRequest(), parent='parent_value')

def test_list_jobs_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.side_effect = (cloudscheduler.ListJobsResponse(jobs=[job.Job(), job.Job(), job.Job()], next_page_token='abc'), cloudscheduler.ListJobsResponse(jobs=[], next_page_token='def'), cloudscheduler.ListJobsResponse(jobs=[job.Job()], next_page_token='ghi'), cloudscheduler.ListJobsResponse(jobs=[job.Job(), job.Job()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_jobs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, job.Job) for i in results))

def test_list_jobs_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.side_effect = (cloudscheduler.ListJobsResponse(jobs=[job.Job(), job.Job(), job.Job()], next_page_token='abc'), cloudscheduler.ListJobsResponse(jobs=[], next_page_token='def'), cloudscheduler.ListJobsResponse(jobs=[job.Job()], next_page_token='ghi'), cloudscheduler.ListJobsResponse(jobs=[job.Job(), job.Job()]), RuntimeError)
        pages = list(client.list_jobs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_jobs_async_pager():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloudscheduler.ListJobsResponse(jobs=[job.Job(), job.Job(), job.Job()], next_page_token='abc'), cloudscheduler.ListJobsResponse(jobs=[], next_page_token='def'), cloudscheduler.ListJobsResponse(jobs=[job.Job()], next_page_token='ghi'), cloudscheduler.ListJobsResponse(jobs=[job.Job(), job.Job()]), RuntimeError)
        async_pager = await client.list_jobs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, job.Job) for i in responses))

@pytest.mark.asyncio
async def test_list_jobs_async_pages():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloudscheduler.ListJobsResponse(jobs=[job.Job(), job.Job(), job.Job()], next_page_token='abc'), cloudscheduler.ListJobsResponse(jobs=[], next_page_token='def'), cloudscheduler.ListJobsResponse(jobs=[job.Job()], next_page_token='ghi'), cloudscheduler.ListJobsResponse(jobs=[job.Job(), job.Job()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_jobs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloudscheduler.GetJobRequest, dict])
def test_get_job(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = job.Job(name='name_value', description='description_value', schedule='schedule_value', time_zone='time_zone_value', state=job.Job.State.ENABLED, legacy_app_engine_cron=True)
        response = client.get_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.GetJobRequest()
    assert isinstance(response, job.Job)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.schedule == 'schedule_value'
    assert response.time_zone == 'time_zone_value'
    assert response.state == job.Job.State.ENABLED
    assert response.legacy_app_engine_cron is True

def test_get_job_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        client.get_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.GetJobRequest()

@pytest.mark.asyncio
async def test_get_job_async(transport: str='grpc_asyncio', request_type=cloudscheduler.GetJobRequest):
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job.Job(name='name_value', description='description_value', schedule='schedule_value', time_zone='time_zone_value', state=job.Job.State.ENABLED, legacy_app_engine_cron=True))
        response = await client.get_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.GetJobRequest()
    assert isinstance(response, job.Job)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.schedule == 'schedule_value'
    assert response.time_zone == 'time_zone_value'
    assert response.state == job.Job.State.ENABLED
    assert response.legacy_app_engine_cron is True

@pytest.mark.asyncio
async def test_get_job_async_from_dict():
    await test_get_job_async(request_type=dict)

def test_get_job_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudscheduler.GetJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = job.Job()
        client.get_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_job_field_headers_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudscheduler.GetJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job.Job())
        await client.get_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_job_flattened():
    if False:
        while True:
            i = 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = job.Job()
        client.get_job(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_job_flattened_error():
    if False:
        print('Hello World!')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_job(cloudscheduler.GetJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_job_flattened_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = job.Job()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job.Job())
        response = await client.get_job(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_job_flattened_error_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_job(cloudscheduler.GetJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloudscheduler.CreateJobRequest, dict])
def test_create_job(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = gcs_job.Job(name='name_value', description='description_value', schedule='schedule_value', time_zone='time_zone_value', state=gcs_job.Job.State.ENABLED, legacy_app_engine_cron=True)
        response = client.create_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.CreateJobRequest()
    assert isinstance(response, gcs_job.Job)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.schedule == 'schedule_value'
    assert response.time_zone == 'time_zone_value'
    assert response.state == gcs_job.Job.State.ENABLED
    assert response.legacy_app_engine_cron is True

def test_create_job_empty_call():
    if False:
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        client.create_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.CreateJobRequest()

@pytest.mark.asyncio
async def test_create_job_async(transport: str='grpc_asyncio', request_type=cloudscheduler.CreateJobRequest):
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_job.Job(name='name_value', description='description_value', schedule='schedule_value', time_zone='time_zone_value', state=gcs_job.Job.State.ENABLED, legacy_app_engine_cron=True))
        response = await client.create_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.CreateJobRequest()
    assert isinstance(response, gcs_job.Job)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.schedule == 'schedule_value'
    assert response.time_zone == 'time_zone_value'
    assert response.state == gcs_job.Job.State.ENABLED
    assert response.legacy_app_engine_cron is True

@pytest.mark.asyncio
async def test_create_job_async_from_dict():
    await test_create_job_async(request_type=dict)

def test_create_job_field_headers():
    if False:
        while True:
            i = 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudscheduler.CreateJobRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = gcs_job.Job()
        client.create_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_job_field_headers_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudscheduler.CreateJobRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_job.Job())
        await client.create_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_job_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = gcs_job.Job()
        client.create_job(parent='parent_value', job=gcs_job.Job(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].job
        mock_val = gcs_job.Job(name='name_value')
        assert arg == mock_val

def test_create_job_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_job(cloudscheduler.CreateJobRequest(), parent='parent_value', job=gcs_job.Job(name='name_value'))

@pytest.mark.asyncio
async def test_create_job_flattened_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = gcs_job.Job()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_job.Job())
        response = await client.create_job(parent='parent_value', job=gcs_job.Job(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].job
        mock_val = gcs_job.Job(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_job_flattened_error_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_job(cloudscheduler.CreateJobRequest(), parent='parent_value', job=gcs_job.Job(name='name_value'))

@pytest.mark.parametrize('request_type', [cloudscheduler.UpdateJobRequest, dict])
def test_update_job(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = gcs_job.Job(name='name_value', description='description_value', schedule='schedule_value', time_zone='time_zone_value', state=gcs_job.Job.State.ENABLED, legacy_app_engine_cron=True)
        response = client.update_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.UpdateJobRequest()
    assert isinstance(response, gcs_job.Job)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.schedule == 'schedule_value'
    assert response.time_zone == 'time_zone_value'
    assert response.state == gcs_job.Job.State.ENABLED
    assert response.legacy_app_engine_cron is True

def test_update_job_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        client.update_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.UpdateJobRequest()

@pytest.mark.asyncio
async def test_update_job_async(transport: str='grpc_asyncio', request_type=cloudscheduler.UpdateJobRequest):
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_job.Job(name='name_value', description='description_value', schedule='schedule_value', time_zone='time_zone_value', state=gcs_job.Job.State.ENABLED, legacy_app_engine_cron=True))
        response = await client.update_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.UpdateJobRequest()
    assert isinstance(response, gcs_job.Job)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.schedule == 'schedule_value'
    assert response.time_zone == 'time_zone_value'
    assert response.state == gcs_job.Job.State.ENABLED
    assert response.legacy_app_engine_cron is True

@pytest.mark.asyncio
async def test_update_job_async_from_dict():
    await test_update_job_async(request_type=dict)

def test_update_job_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudscheduler.UpdateJobRequest()
    request.job.name = 'name_value'
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = gcs_job.Job()
        client.update_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'job.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_job_field_headers_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudscheduler.UpdateJobRequest()
    request.job.name = 'name_value'
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_job.Job())
        await client.update_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'job.name=name_value') in kw['metadata']

def test_update_job_flattened():
    if False:
        while True:
            i = 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = gcs_job.Job()
        client.update_job(job=gcs_job.Job(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].job
        mock_val = gcs_job.Job(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_job_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_job(cloudscheduler.UpdateJobRequest(), job=gcs_job.Job(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_job_flattened_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = gcs_job.Job()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_job.Job())
        response = await client.update_job(job=gcs_job.Job(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].job
        mock_val = gcs_job.Job(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_job_flattened_error_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_job(cloudscheduler.UpdateJobRequest(), job=gcs_job.Job(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [cloudscheduler.DeleteJobRequest, dict])
def test_delete_job(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        call.return_value = None
        response = client.delete_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.DeleteJobRequest()
    assert response is None

def test_delete_job_empty_call():
    if False:
        while True:
            i = 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        client.delete_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.DeleteJobRequest()

@pytest.mark.asyncio
async def test_delete_job_async(transport: str='grpc_asyncio', request_type=cloudscheduler.DeleteJobRequest):
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.DeleteJobRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_job_async_from_dict():
    await test_delete_job_async(request_type=dict)

def test_delete_job_field_headers():
    if False:
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudscheduler.DeleteJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        call.return_value = None
        client.delete_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_job_field_headers_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudscheduler.DeleteJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_job_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        call.return_value = None
        client.delete_job(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_job_flattened_error():
    if False:
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_job(cloudscheduler.DeleteJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_job_flattened_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_job(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_job_flattened_error_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_job(cloudscheduler.DeleteJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloudscheduler.PauseJobRequest, dict])
def test_pause_job(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.pause_job), '__call__') as call:
        call.return_value = job.Job(name='name_value', description='description_value', schedule='schedule_value', time_zone='time_zone_value', state=job.Job.State.ENABLED, legacy_app_engine_cron=True)
        response = client.pause_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.PauseJobRequest()
    assert isinstance(response, job.Job)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.schedule == 'schedule_value'
    assert response.time_zone == 'time_zone_value'
    assert response.state == job.Job.State.ENABLED
    assert response.legacy_app_engine_cron is True

def test_pause_job_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.pause_job), '__call__') as call:
        client.pause_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.PauseJobRequest()

@pytest.mark.asyncio
async def test_pause_job_async(transport: str='grpc_asyncio', request_type=cloudscheduler.PauseJobRequest):
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.pause_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job.Job(name='name_value', description='description_value', schedule='schedule_value', time_zone='time_zone_value', state=job.Job.State.ENABLED, legacy_app_engine_cron=True))
        response = await client.pause_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.PauseJobRequest()
    assert isinstance(response, job.Job)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.schedule == 'schedule_value'
    assert response.time_zone == 'time_zone_value'
    assert response.state == job.Job.State.ENABLED
    assert response.legacy_app_engine_cron is True

@pytest.mark.asyncio
async def test_pause_job_async_from_dict():
    await test_pause_job_async(request_type=dict)

def test_pause_job_field_headers():
    if False:
        print('Hello World!')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudscheduler.PauseJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.pause_job), '__call__') as call:
        call.return_value = job.Job()
        client.pause_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_pause_job_field_headers_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudscheduler.PauseJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.pause_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job.Job())
        await client.pause_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_pause_job_flattened():
    if False:
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.pause_job), '__call__') as call:
        call.return_value = job.Job()
        client.pause_job(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_pause_job_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.pause_job(cloudscheduler.PauseJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_pause_job_flattened_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.pause_job), '__call__') as call:
        call.return_value = job.Job()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job.Job())
        response = await client.pause_job(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_pause_job_flattened_error_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.pause_job(cloudscheduler.PauseJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloudscheduler.ResumeJobRequest, dict])
def test_resume_job(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resume_job), '__call__') as call:
        call.return_value = job.Job(name='name_value', description='description_value', schedule='schedule_value', time_zone='time_zone_value', state=job.Job.State.ENABLED, legacy_app_engine_cron=True)
        response = client.resume_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.ResumeJobRequest()
    assert isinstance(response, job.Job)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.schedule == 'schedule_value'
    assert response.time_zone == 'time_zone_value'
    assert response.state == job.Job.State.ENABLED
    assert response.legacy_app_engine_cron is True

def test_resume_job_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.resume_job), '__call__') as call:
        client.resume_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.ResumeJobRequest()

@pytest.mark.asyncio
async def test_resume_job_async(transport: str='grpc_asyncio', request_type=cloudscheduler.ResumeJobRequest):
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resume_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job.Job(name='name_value', description='description_value', schedule='schedule_value', time_zone='time_zone_value', state=job.Job.State.ENABLED, legacy_app_engine_cron=True))
        response = await client.resume_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.ResumeJobRequest()
    assert isinstance(response, job.Job)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.schedule == 'schedule_value'
    assert response.time_zone == 'time_zone_value'
    assert response.state == job.Job.State.ENABLED
    assert response.legacy_app_engine_cron is True

@pytest.mark.asyncio
async def test_resume_job_async_from_dict():
    await test_resume_job_async(request_type=dict)

def test_resume_job_field_headers():
    if False:
        print('Hello World!')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudscheduler.ResumeJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.resume_job), '__call__') as call:
        call.return_value = job.Job()
        client.resume_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_resume_job_field_headers_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudscheduler.ResumeJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.resume_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job.Job())
        await client.resume_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_resume_job_flattened():
    if False:
        while True:
            i = 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.resume_job), '__call__') as call:
        call.return_value = job.Job()
        client.resume_job(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_resume_job_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.resume_job(cloudscheduler.ResumeJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_resume_job_flattened_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.resume_job), '__call__') as call:
        call.return_value = job.Job()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job.Job())
        response = await client.resume_job(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_resume_job_flattened_error_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.resume_job(cloudscheduler.ResumeJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloudscheduler.RunJobRequest, dict])
def test_run_job(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.run_job), '__call__') as call:
        call.return_value = job.Job(name='name_value', description='description_value', schedule='schedule_value', time_zone='time_zone_value', state=job.Job.State.ENABLED, legacy_app_engine_cron=True)
        response = client.run_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.RunJobRequest()
    assert isinstance(response, job.Job)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.schedule == 'schedule_value'
    assert response.time_zone == 'time_zone_value'
    assert response.state == job.Job.State.ENABLED
    assert response.legacy_app_engine_cron is True

def test_run_job_empty_call():
    if False:
        while True:
            i = 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.run_job), '__call__') as call:
        client.run_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.RunJobRequest()

@pytest.mark.asyncio
async def test_run_job_async(transport: str='grpc_asyncio', request_type=cloudscheduler.RunJobRequest):
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.run_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job.Job(name='name_value', description='description_value', schedule='schedule_value', time_zone='time_zone_value', state=job.Job.State.ENABLED, legacy_app_engine_cron=True))
        response = await client.run_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloudscheduler.RunJobRequest()
    assert isinstance(response, job.Job)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.schedule == 'schedule_value'
    assert response.time_zone == 'time_zone_value'
    assert response.state == job.Job.State.ENABLED
    assert response.legacy_app_engine_cron is True

@pytest.mark.asyncio
async def test_run_job_async_from_dict():
    await test_run_job_async(request_type=dict)

def test_run_job_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudscheduler.RunJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.run_job), '__call__') as call:
        call.return_value = job.Job()
        client.run_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_run_job_field_headers_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloudscheduler.RunJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.run_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job.Job())
        await client.run_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_run_job_flattened():
    if False:
        print('Hello World!')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.run_job), '__call__') as call:
        call.return_value = job.Job()
        client.run_job(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_run_job_flattened_error():
    if False:
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.run_job(cloudscheduler.RunJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_run_job_flattened_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.run_job), '__call__') as call:
        call.return_value = job.Job()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job.Job())
        response = await client.run_job(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_run_job_flattened_error_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.run_job(cloudscheduler.RunJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloudscheduler.ListJobsRequest, dict])
def test_list_jobs_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloudscheduler.ListJobsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = cloudscheduler.ListJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_jobs(request)
    assert isinstance(response, pagers.ListJobsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_jobs_rest_required_fields(request_type=cloudscheduler.ListJobsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudSchedulerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_jobs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'legacy_app_engine_cron', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloudscheduler.ListJobsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloudscheduler.ListJobsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_jobs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_jobs_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudSchedulerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_jobs._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'legacyAppEngineCron', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_jobs_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudSchedulerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudSchedulerRestInterceptor())
    client = CloudSchedulerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudSchedulerRestInterceptor, 'post_list_jobs') as post, mock.patch.object(transports.CloudSchedulerRestInterceptor, 'pre_list_jobs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudscheduler.ListJobsRequest.pb(cloudscheduler.ListJobsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloudscheduler.ListJobsResponse.to_json(cloudscheduler.ListJobsResponse())
        request = cloudscheduler.ListJobsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloudscheduler.ListJobsResponse()
        client.list_jobs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_jobs_rest_bad_request(transport: str='rest', request_type=cloudscheduler.ListJobsRequest):
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_jobs(request)

def test_list_jobs_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloudscheduler.ListJobsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloudscheduler.ListJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_jobs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*/locations/*}/jobs' % client.transport._host, args[1])

def test_list_jobs_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_jobs(cloudscheduler.ListJobsRequest(), parent='parent_value')

def test_list_jobs_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (cloudscheduler.ListJobsResponse(jobs=[job.Job(), job.Job(), job.Job()], next_page_token='abc'), cloudscheduler.ListJobsResponse(jobs=[], next_page_token='def'), cloudscheduler.ListJobsResponse(jobs=[job.Job()], next_page_token='ghi'), cloudscheduler.ListJobsResponse(jobs=[job.Job(), job.Job()]))
        response = response + response
        response = tuple((cloudscheduler.ListJobsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_jobs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, job.Job) for i in results))
        pages = list(client.list_jobs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloudscheduler.GetJobRequest, dict])
def test_get_job_rest(request_type):
    if False:
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/jobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = job.Job(name='name_value', description='description_value', schedule='schedule_value', time_zone='time_zone_value', state=job.Job.State.ENABLED, legacy_app_engine_cron=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = job.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_job(request)
    assert isinstance(response, job.Job)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.schedule == 'schedule_value'
    assert response.time_zone == 'time_zone_value'
    assert response.state == job.Job.State.ENABLED
    assert response.legacy_app_engine_cron is True

def test_get_job_rest_required_fields(request_type=cloudscheduler.GetJobRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudSchedulerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = job.Job()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = job.Job.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_job_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudSchedulerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_job_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudSchedulerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudSchedulerRestInterceptor())
    client = CloudSchedulerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudSchedulerRestInterceptor, 'post_get_job') as post, mock.patch.object(transports.CloudSchedulerRestInterceptor, 'pre_get_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudscheduler.GetJobRequest.pb(cloudscheduler.GetJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = job.Job.to_json(job.Job())
        request = cloudscheduler.GetJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = job.Job()
        client.get_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_job_rest_bad_request(transport: str='rest', request_type=cloudscheduler.GetJobRequest):
    if False:
        while True:
            i = 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/jobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_job(request)

def test_get_job_rest_flattened():
    if False:
        while True:
            i = 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = job.Job()
        sample_request = {'name': 'projects/sample1/locations/sample2/jobs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = job.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/jobs/*}' % client.transport._host, args[1])

def test_get_job_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_job(cloudscheduler.GetJobRequest(), name='name_value')

def test_get_job_rest_error():
    if False:
        print('Hello World!')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudscheduler.CreateJobRequest, dict])
def test_create_job_rest(request_type):
    if False:
        while True:
            i = 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['job'] = {'name': 'name_value', 'description': 'description_value', 'pubsub_target': {'topic_name': 'topic_name_value', 'data': b'data_blob', 'attributes': {}}, 'app_engine_http_target': {'http_method': 1, 'app_engine_routing': {'service': 'service_value', 'version': 'version_value', 'instance': 'instance_value', 'host': 'host_value'}, 'relative_uri': 'relative_uri_value', 'headers': {}, 'body': b'body_blob'}, 'http_target': {'uri': 'uri_value', 'http_method': 1, 'headers': {}, 'body': b'body_blob', 'oauth_token': {'service_account_email': 'service_account_email_value', 'scope': 'scope_value'}, 'oidc_token': {'service_account_email': 'service_account_email_value', 'audience': 'audience_value'}}, 'schedule': 'schedule_value', 'time_zone': 'time_zone_value', 'user_update_time': {'seconds': 751, 'nanos': 543}, 'state': 1, 'status': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}, 'schedule_time': {}, 'last_attempt_time': {}, 'retry_config': {'retry_count': 1214, 'max_retry_duration': {'seconds': 751, 'nanos': 543}, 'min_backoff_duration': {}, 'max_backoff_duration': {}, 'max_doublings': 1388}, 'attempt_deadline': {}, 'legacy_app_engine_cron': True}
    test_field = cloudscheduler.CreateJobRequest.meta.fields['job']

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
    for (field, value) in request_init['job'].items():
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
                for i in range(0, len(request_init['job'][field])):
                    del request_init['job'][field][i][subfield]
            else:
                del request_init['job'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_job.Job(name='name_value', description='description_value', schedule='schedule_value', time_zone='time_zone_value', state=gcs_job.Job.State.ENABLED, legacy_app_engine_cron=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_job.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_job(request)
    assert isinstance(response, gcs_job.Job)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.schedule == 'schedule_value'
    assert response.time_zone == 'time_zone_value'
    assert response.state == gcs_job.Job.State.ENABLED
    assert response.legacy_app_engine_cron is True

def test_create_job_rest_required_fields(request_type=cloudscheduler.CreateJobRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudSchedulerRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcs_job.Job()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcs_job.Job.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_job_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudSchedulerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'job'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_job_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudSchedulerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudSchedulerRestInterceptor())
    client = CloudSchedulerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudSchedulerRestInterceptor, 'post_create_job') as post, mock.patch.object(transports.CloudSchedulerRestInterceptor, 'pre_create_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudscheduler.CreateJobRequest.pb(cloudscheduler.CreateJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcs_job.Job.to_json(gcs_job.Job())
        request = cloudscheduler.CreateJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcs_job.Job()
        client.create_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_job_rest_bad_request(transport: str='rest', request_type=cloudscheduler.CreateJobRequest):
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_job(request)

def test_create_job_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_job.Job()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', job=gcs_job.Job(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_job.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*/locations/*}/jobs' % client.transport._host, args[1])

def test_create_job_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_job(cloudscheduler.CreateJobRequest(), parent='parent_value', job=gcs_job.Job(name='name_value'))

def test_create_job_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudscheduler.UpdateJobRequest, dict])
def test_update_job_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'job': {'name': 'projects/sample1/locations/sample2/jobs/sample3'}}
    request_init['job'] = {'name': 'projects/sample1/locations/sample2/jobs/sample3', 'description': 'description_value', 'pubsub_target': {'topic_name': 'topic_name_value', 'data': b'data_blob', 'attributes': {}}, 'app_engine_http_target': {'http_method': 1, 'app_engine_routing': {'service': 'service_value', 'version': 'version_value', 'instance': 'instance_value', 'host': 'host_value'}, 'relative_uri': 'relative_uri_value', 'headers': {}, 'body': b'body_blob'}, 'http_target': {'uri': 'uri_value', 'http_method': 1, 'headers': {}, 'body': b'body_blob', 'oauth_token': {'service_account_email': 'service_account_email_value', 'scope': 'scope_value'}, 'oidc_token': {'service_account_email': 'service_account_email_value', 'audience': 'audience_value'}}, 'schedule': 'schedule_value', 'time_zone': 'time_zone_value', 'user_update_time': {'seconds': 751, 'nanos': 543}, 'state': 1, 'status': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}, 'schedule_time': {}, 'last_attempt_time': {}, 'retry_config': {'retry_count': 1214, 'max_retry_duration': {'seconds': 751, 'nanos': 543}, 'min_backoff_duration': {}, 'max_backoff_duration': {}, 'max_doublings': 1388}, 'attempt_deadline': {}, 'legacy_app_engine_cron': True}
    test_field = cloudscheduler.UpdateJobRequest.meta.fields['job']

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
    for (field, value) in request_init['job'].items():
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
                for i in range(0, len(request_init['job'][field])):
                    del request_init['job'][field][i][subfield]
            else:
                del request_init['job'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_job.Job(name='name_value', description='description_value', schedule='schedule_value', time_zone='time_zone_value', state=gcs_job.Job.State.ENABLED, legacy_app_engine_cron=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_job.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_job(request)
    assert isinstance(response, gcs_job.Job)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.schedule == 'schedule_value'
    assert response.time_zone == 'time_zone_value'
    assert response.state == gcs_job.Job.State.ENABLED
    assert response.legacy_app_engine_cron is True

def test_update_job_rest_required_fields(request_type=cloudscheduler.UpdateJobRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudSchedulerRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_job._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcs_job.Job()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcs_job.Job.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_job_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudSchedulerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_job._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('job',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_job_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudSchedulerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudSchedulerRestInterceptor())
    client = CloudSchedulerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudSchedulerRestInterceptor, 'post_update_job') as post, mock.patch.object(transports.CloudSchedulerRestInterceptor, 'pre_update_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudscheduler.UpdateJobRequest.pb(cloudscheduler.UpdateJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcs_job.Job.to_json(gcs_job.Job())
        request = cloudscheduler.UpdateJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcs_job.Job()
        client.update_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_job_rest_bad_request(transport: str='rest', request_type=cloudscheduler.UpdateJobRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'job': {'name': 'projects/sample1/locations/sample2/jobs/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_job(request)

def test_update_job_rest_flattened():
    if False:
        while True:
            i = 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_job.Job()
        sample_request = {'job': {'name': 'projects/sample1/locations/sample2/jobs/sample3'}}
        mock_args = dict(job=gcs_job.Job(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_job.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{job.name=projects/*/locations/*/jobs/*}' % client.transport._host, args[1])

def test_update_job_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_job(cloudscheduler.UpdateJobRequest(), job=gcs_job.Job(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_job_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudscheduler.DeleteJobRequest, dict])
def test_delete_job_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/jobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_job(request)
    assert response is None

def test_delete_job_rest_required_fields(request_type=cloudscheduler.DeleteJobRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CloudSchedulerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_job._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('legacy_app_engine_cron',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_job_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudSchedulerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_job._get_unset_required_fields({})
    assert set(unset_fields) == set(('legacyAppEngineCron',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_job_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudSchedulerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudSchedulerRestInterceptor())
    client = CloudSchedulerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudSchedulerRestInterceptor, 'pre_delete_job') as pre:
        pre.assert_not_called()
        pb_message = cloudscheduler.DeleteJobRequest.pb(cloudscheduler.DeleteJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = cloudscheduler.DeleteJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_job_rest_bad_request(transport: str='rest', request_type=cloudscheduler.DeleteJobRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/jobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_job(request)

def test_delete_job_rest_flattened():
    if False:
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/jobs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/jobs/*}' % client.transport._host, args[1])

def test_delete_job_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_job(cloudscheduler.DeleteJobRequest(), name='name_value')

def test_delete_job_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudscheduler.PauseJobRequest, dict])
def test_pause_job_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/jobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = job.Job(name='name_value', description='description_value', schedule='schedule_value', time_zone='time_zone_value', state=job.Job.State.ENABLED, legacy_app_engine_cron=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = job.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.pause_job(request)
    assert isinstance(response, job.Job)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.schedule == 'schedule_value'
    assert response.time_zone == 'time_zone_value'
    assert response.state == job.Job.State.ENABLED
    assert response.legacy_app_engine_cron is True

def test_pause_job_rest_required_fields(request_type=cloudscheduler.PauseJobRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudSchedulerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).pause_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).pause_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = job.Job()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = job.Job.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.pause_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_pause_job_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudSchedulerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.pause_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_pause_job_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudSchedulerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudSchedulerRestInterceptor())
    client = CloudSchedulerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudSchedulerRestInterceptor, 'post_pause_job') as post, mock.patch.object(transports.CloudSchedulerRestInterceptor, 'pre_pause_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudscheduler.PauseJobRequest.pb(cloudscheduler.PauseJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = job.Job.to_json(job.Job())
        request = cloudscheduler.PauseJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = job.Job()
        client.pause_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_pause_job_rest_bad_request(transport: str='rest', request_type=cloudscheduler.PauseJobRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/jobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.pause_job(request)

def test_pause_job_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = job.Job()
        sample_request = {'name': 'projects/sample1/locations/sample2/jobs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = job.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.pause_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/jobs/*}:pause' % client.transport._host, args[1])

def test_pause_job_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.pause_job(cloudscheduler.PauseJobRequest(), name='name_value')

def test_pause_job_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudscheduler.ResumeJobRequest, dict])
def test_resume_job_rest(request_type):
    if False:
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/jobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = job.Job(name='name_value', description='description_value', schedule='schedule_value', time_zone='time_zone_value', state=job.Job.State.ENABLED, legacy_app_engine_cron=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = job.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.resume_job(request)
    assert isinstance(response, job.Job)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.schedule == 'schedule_value'
    assert response.time_zone == 'time_zone_value'
    assert response.state == job.Job.State.ENABLED
    assert response.legacy_app_engine_cron is True

def test_resume_job_rest_required_fields(request_type=cloudscheduler.ResumeJobRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudSchedulerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resume_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resume_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = job.Job()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = job.Job.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.resume_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_resume_job_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudSchedulerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.resume_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_resume_job_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudSchedulerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudSchedulerRestInterceptor())
    client = CloudSchedulerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudSchedulerRestInterceptor, 'post_resume_job') as post, mock.patch.object(transports.CloudSchedulerRestInterceptor, 'pre_resume_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudscheduler.ResumeJobRequest.pb(cloudscheduler.ResumeJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = job.Job.to_json(job.Job())
        request = cloudscheduler.ResumeJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = job.Job()
        client.resume_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_resume_job_rest_bad_request(transport: str='rest', request_type=cloudscheduler.ResumeJobRequest):
    if False:
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/jobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.resume_job(request)

def test_resume_job_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = job.Job()
        sample_request = {'name': 'projects/sample1/locations/sample2/jobs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = job.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.resume_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/jobs/*}:resume' % client.transport._host, args[1])

def test_resume_job_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.resume_job(cloudscheduler.ResumeJobRequest(), name='name_value')

def test_resume_job_rest_error():
    if False:
        print('Hello World!')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloudscheduler.RunJobRequest, dict])
def test_run_job_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/jobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = job.Job(name='name_value', description='description_value', schedule='schedule_value', time_zone='time_zone_value', state=job.Job.State.ENABLED, legacy_app_engine_cron=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = job.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.run_job(request)
    assert isinstance(response, job.Job)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.schedule == 'schedule_value'
    assert response.time_zone == 'time_zone_value'
    assert response.state == job.Job.State.ENABLED
    assert response.legacy_app_engine_cron is True

def test_run_job_rest_required_fields(request_type=cloudscheduler.RunJobRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudSchedulerRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).run_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).run_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = job.Job()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = job.Job.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.run_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_run_job_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudSchedulerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.run_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_run_job_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudSchedulerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudSchedulerRestInterceptor())
    client = CloudSchedulerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudSchedulerRestInterceptor, 'post_run_job') as post, mock.patch.object(transports.CloudSchedulerRestInterceptor, 'pre_run_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloudscheduler.RunJobRequest.pb(cloudscheduler.RunJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = job.Job.to_json(job.Job())
        request = cloudscheduler.RunJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = job.Job()
        client.run_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_run_job_rest_bad_request(transport: str='rest', request_type=cloudscheduler.RunJobRequest):
    if False:
        i = 10
        return i + 15
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/jobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.run_job(request)

def test_run_job_rest_flattened():
    if False:
        print('Hello World!')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = job.Job()
        sample_request = {'name': 'projects/sample1/locations/sample2/jobs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = job.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.run_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/jobs/*}:run' % client.transport._host, args[1])

def test_run_job_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.run_job(cloudscheduler.RunJobRequest(), name='name_value')

def test_run_job_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudSchedulerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.CloudSchedulerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudSchedulerClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.CloudSchedulerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CloudSchedulerClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CloudSchedulerClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.CloudSchedulerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudSchedulerClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.CloudSchedulerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = CloudSchedulerClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.CloudSchedulerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.CloudSchedulerGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.CloudSchedulerGrpcTransport, transports.CloudSchedulerGrpcAsyncIOTransport, transports.CloudSchedulerRestTransport])
def test_transport_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        i = 10
        return i + 15
    transport = CloudSchedulerClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.CloudSchedulerGrpcTransport)

def test_cloud_scheduler_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.CloudSchedulerTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_cloud_scheduler_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.scheduler_v1beta1.services.cloud_scheduler.transports.CloudSchedulerTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.CloudSchedulerTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_jobs', 'get_job', 'create_job', 'update_job', 'delete_job', 'pause_job', 'resume_job', 'run_job', 'get_location', 'list_locations')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_cloud_scheduler_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.scheduler_v1beta1.services.cloud_scheduler.transports.CloudSchedulerTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CloudSchedulerTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_cloud_scheduler_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.scheduler_v1beta1.services.cloud_scheduler.transports.CloudSchedulerTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CloudSchedulerTransport()
        adc.assert_called_once()

def test_cloud_scheduler_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        CloudSchedulerClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.CloudSchedulerGrpcTransport, transports.CloudSchedulerGrpcAsyncIOTransport])
def test_cloud_scheduler_transport_auth_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.CloudSchedulerGrpcTransport, transports.CloudSchedulerGrpcAsyncIOTransport, transports.CloudSchedulerRestTransport])
def test_cloud_scheduler_transport_auth_gdch_credentials(transport_class):
    if False:
        return 10
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.CloudSchedulerGrpcTransport, grpc_helpers), (transports.CloudSchedulerGrpcAsyncIOTransport, grpc_helpers_async)])
def test_cloud_scheduler_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('cloudscheduler.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='cloudscheduler.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.CloudSchedulerGrpcTransport, transports.CloudSchedulerGrpcAsyncIOTransport])
def test_cloud_scheduler_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_cloud_scheduler_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.CloudSchedulerRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_cloud_scheduler_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudscheduler.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('cloudscheduler.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudscheduler.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_cloud_scheduler_host_with_port(transport_name):
    if False:
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudscheduler.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('cloudscheduler.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudscheduler.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_cloud_scheduler_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = CloudSchedulerClient(credentials=creds1, transport=transport_name)
    client2 = CloudSchedulerClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_jobs._session
    session2 = client2.transport.list_jobs._session
    assert session1 != session2
    session1 = client1.transport.get_job._session
    session2 = client2.transport.get_job._session
    assert session1 != session2
    session1 = client1.transport.create_job._session
    session2 = client2.transport.create_job._session
    assert session1 != session2
    session1 = client1.transport.update_job._session
    session2 = client2.transport.update_job._session
    assert session1 != session2
    session1 = client1.transport.delete_job._session
    session2 = client2.transport.delete_job._session
    assert session1 != session2
    session1 = client1.transport.pause_job._session
    session2 = client2.transport.pause_job._session
    assert session1 != session2
    session1 = client1.transport.resume_job._session
    session2 = client2.transport.resume_job._session
    assert session1 != session2
    session1 = client1.transport.run_job._session
    session2 = client2.transport.run_job._session
    assert session1 != session2

def test_cloud_scheduler_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CloudSchedulerGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_cloud_scheduler_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CloudSchedulerGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.CloudSchedulerGrpcTransport, transports.CloudSchedulerGrpcAsyncIOTransport])
def test_cloud_scheduler_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.CloudSchedulerGrpcTransport, transports.CloudSchedulerGrpcAsyncIOTransport])
def test_cloud_scheduler_transport_channel_mtls_with_adc(transport_class):
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

def test_job_path():
    if False:
        return 10
    project = 'squid'
    location = 'clam'
    job = 'whelk'
    expected = 'projects/{project}/locations/{location}/jobs/{job}'.format(project=project, location=location, job=job)
    actual = CloudSchedulerClient.job_path(project, location, job)
    assert expected == actual

def test_parse_job_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus', 'location': 'oyster', 'job': 'nudibranch'}
    path = CloudSchedulerClient.job_path(**expected)
    actual = CloudSchedulerClient.parse_job_path(path)
    assert expected == actual

def test_topic_path():
    if False:
        i = 10
        return i + 15
    project = 'cuttlefish'
    topic = 'mussel'
    expected = 'projects/{project}/topics/{topic}'.format(project=project, topic=topic)
    actual = CloudSchedulerClient.topic_path(project, topic)
    assert expected == actual

def test_parse_topic_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'winkle', 'topic': 'nautilus'}
    path = CloudSchedulerClient.topic_path(**expected)
    actual = CloudSchedulerClient.parse_topic_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'scallop'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = CloudSchedulerClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'abalone'}
    path = CloudSchedulerClient.common_billing_account_path(**expected)
    actual = CloudSchedulerClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'squid'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = CloudSchedulerClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'clam'}
    path = CloudSchedulerClient.common_folder_path(**expected)
    actual = CloudSchedulerClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'whelk'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = CloudSchedulerClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        i = 10
        return i + 15
    expected = {'organization': 'octopus'}
    path = CloudSchedulerClient.common_organization_path(**expected)
    actual = CloudSchedulerClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'oyster'
    expected = 'projects/{project}'.format(project=project)
    actual = CloudSchedulerClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'nudibranch'}
    path = CloudSchedulerClient.common_project_path(**expected)
    actual = CloudSchedulerClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'cuttlefish'
    location = 'mussel'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = CloudSchedulerClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'winkle', 'location': 'nautilus'}
    path = CloudSchedulerClient.common_location_path(**expected)
    actual = CloudSchedulerClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.CloudSchedulerTransport, '_prep_wrapped_messages') as prep:
        client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.CloudSchedulerTransport, '_prep_wrapped_messages') as prep:
        transport_class = CloudSchedulerClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = CloudSchedulerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = CloudSchedulerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(CloudSchedulerClient, transports.CloudSchedulerGrpcTransport), (CloudSchedulerAsyncClient, transports.CloudSchedulerGrpcAsyncIOTransport)])
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