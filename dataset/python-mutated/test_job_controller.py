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
from google.cloud.dataproc_v1.services.job_controller import JobControllerAsyncClient, JobControllerClient, pagers, transports
from google.cloud.dataproc_v1.types import jobs

def client_cert_source_callback():
    if False:
        return 10
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
    assert JobControllerClient._get_default_mtls_endpoint(None) is None
    assert JobControllerClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert JobControllerClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert JobControllerClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert JobControllerClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert JobControllerClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(JobControllerClient, 'grpc'), (JobControllerAsyncClient, 'grpc_asyncio'), (JobControllerClient, 'rest')])
def test_job_controller_client_from_service_account_info(client_class, transport_name):
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

@pytest.mark.parametrize('transport_class,transport_name', [(transports.JobControllerGrpcTransport, 'grpc'), (transports.JobControllerGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.JobControllerRestTransport, 'rest')])
def test_job_controller_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(JobControllerClient, 'grpc'), (JobControllerAsyncClient, 'grpc_asyncio'), (JobControllerClient, 'rest')])
def test_job_controller_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('dataproc.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dataproc.googleapis.com')

def test_job_controller_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = JobControllerClient.get_transport_class()
    available_transports = [transports.JobControllerGrpcTransport, transports.JobControllerRestTransport]
    assert transport in available_transports
    transport = JobControllerClient.get_transport_class('grpc')
    assert transport == transports.JobControllerGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(JobControllerClient, transports.JobControllerGrpcTransport, 'grpc'), (JobControllerAsyncClient, transports.JobControllerGrpcAsyncIOTransport, 'grpc_asyncio'), (JobControllerClient, transports.JobControllerRestTransport, 'rest')])
@mock.patch.object(JobControllerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(JobControllerClient))
@mock.patch.object(JobControllerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(JobControllerAsyncClient))
def test_job_controller_client_client_options(client_class, transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(JobControllerClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(JobControllerClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(JobControllerClient, transports.JobControllerGrpcTransport, 'grpc', 'true'), (JobControllerAsyncClient, transports.JobControllerGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (JobControllerClient, transports.JobControllerGrpcTransport, 'grpc', 'false'), (JobControllerAsyncClient, transports.JobControllerGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (JobControllerClient, transports.JobControllerRestTransport, 'rest', 'true'), (JobControllerClient, transports.JobControllerRestTransport, 'rest', 'false')])
@mock.patch.object(JobControllerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(JobControllerClient))
@mock.patch.object(JobControllerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(JobControllerAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_job_controller_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [JobControllerClient, JobControllerAsyncClient])
@mock.patch.object(JobControllerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(JobControllerClient))
@mock.patch.object(JobControllerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(JobControllerAsyncClient))
def test_job_controller_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(JobControllerClient, transports.JobControllerGrpcTransport, 'grpc'), (JobControllerAsyncClient, transports.JobControllerGrpcAsyncIOTransport, 'grpc_asyncio'), (JobControllerClient, transports.JobControllerRestTransport, 'rest')])
def test_job_controller_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(JobControllerClient, transports.JobControllerGrpcTransport, 'grpc', grpc_helpers), (JobControllerAsyncClient, transports.JobControllerGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (JobControllerClient, transports.JobControllerRestTransport, 'rest', None)])
def test_job_controller_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_job_controller_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.dataproc_v1.services.job_controller.transports.JobControllerGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = JobControllerClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(JobControllerClient, transports.JobControllerGrpcTransport, 'grpc', grpc_helpers), (JobControllerAsyncClient, transports.JobControllerGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_job_controller_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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

@pytest.mark.parametrize('request_type', [jobs.SubmitJobRequest, dict])
def test_submit_job(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.submit_job), '__call__') as call:
        call.return_value = jobs.Job(driver_output_resource_uri='driver_output_resource_uri_value', driver_control_files_uri='driver_control_files_uri_value', job_uuid='job_uuid_value', done=True)
        response = client.submit_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.SubmitJobRequest()
    assert isinstance(response, jobs.Job)
    assert response.driver_output_resource_uri == 'driver_output_resource_uri_value'
    assert response.driver_control_files_uri == 'driver_control_files_uri_value'
    assert response.job_uuid == 'job_uuid_value'
    assert response.done is True

def test_submit_job_empty_call():
    if False:
        while True:
            i = 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.submit_job), '__call__') as call:
        client.submit_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.SubmitJobRequest()

@pytest.mark.asyncio
async def test_submit_job_async(transport: str='grpc_asyncio', request_type=jobs.SubmitJobRequest):
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.submit_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.Job(driver_output_resource_uri='driver_output_resource_uri_value', driver_control_files_uri='driver_control_files_uri_value', job_uuid='job_uuid_value', done=True))
        response = await client.submit_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.SubmitJobRequest()
    assert isinstance(response, jobs.Job)
    assert response.driver_output_resource_uri == 'driver_output_resource_uri_value'
    assert response.driver_control_files_uri == 'driver_control_files_uri_value'
    assert response.job_uuid == 'job_uuid_value'
    assert response.done is True

@pytest.mark.asyncio
async def test_submit_job_async_from_dict():
    await test_submit_job_async(request_type=dict)

def test_submit_job_field_headers():
    if False:
        print('Hello World!')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.SubmitJobRequest()
    request.project_id = 'project_id_value'
    request.region = 'region_value'
    with mock.patch.object(type(client.transport.submit_job), '__call__') as call:
        call.return_value = jobs.Job()
        client.submit_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&region=region_value') in kw['metadata']

@pytest.mark.asyncio
async def test_submit_job_field_headers_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.SubmitJobRequest()
    request.project_id = 'project_id_value'
    request.region = 'region_value'
    with mock.patch.object(type(client.transport.submit_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.Job())
        await client.submit_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&region=region_value') in kw['metadata']

def test_submit_job_flattened():
    if False:
        i = 10
        return i + 15
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.submit_job), '__call__') as call:
        call.return_value = jobs.Job()
        client.submit_job(project_id='project_id_value', region='region_value', job=jobs.Job(reference=jobs.JobReference(project_id='project_id_value')))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].project_id
        mock_val = 'project_id_value'
        assert arg == mock_val
        arg = args[0].region
        mock_val = 'region_value'
        assert arg == mock_val
        arg = args[0].job
        mock_val = jobs.Job(reference=jobs.JobReference(project_id='project_id_value'))
        assert arg == mock_val

def test_submit_job_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.submit_job(jobs.SubmitJobRequest(), project_id='project_id_value', region='region_value', job=jobs.Job(reference=jobs.JobReference(project_id='project_id_value')))

@pytest.mark.asyncio
async def test_submit_job_flattened_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.submit_job), '__call__') as call:
        call.return_value = jobs.Job()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.Job())
        response = await client.submit_job(project_id='project_id_value', region='region_value', job=jobs.Job(reference=jobs.JobReference(project_id='project_id_value')))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].project_id
        mock_val = 'project_id_value'
        assert arg == mock_val
        arg = args[0].region
        mock_val = 'region_value'
        assert arg == mock_val
        arg = args[0].job
        mock_val = jobs.Job(reference=jobs.JobReference(project_id='project_id_value'))
        assert arg == mock_val

@pytest.mark.asyncio
async def test_submit_job_flattened_error_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.submit_job(jobs.SubmitJobRequest(), project_id='project_id_value', region='region_value', job=jobs.Job(reference=jobs.JobReference(project_id='project_id_value')))

@pytest.mark.parametrize('request_type', [jobs.SubmitJobRequest, dict])
def test_submit_job_as_operation(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.submit_job_as_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.submit_job_as_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.SubmitJobRequest()
    assert isinstance(response, future.Future)

def test_submit_job_as_operation_empty_call():
    if False:
        print('Hello World!')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.submit_job_as_operation), '__call__') as call:
        client.submit_job_as_operation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.SubmitJobRequest()

@pytest.mark.asyncio
async def test_submit_job_as_operation_async(transport: str='grpc_asyncio', request_type=jobs.SubmitJobRequest):
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.submit_job_as_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.submit_job_as_operation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.SubmitJobRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_submit_job_as_operation_async_from_dict():
    await test_submit_job_as_operation_async(request_type=dict)

def test_submit_job_as_operation_field_headers():
    if False:
        i = 10
        return i + 15
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.SubmitJobRequest()
    request.project_id = 'project_id_value'
    request.region = 'region_value'
    with mock.patch.object(type(client.transport.submit_job_as_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.submit_job_as_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&region=region_value') in kw['metadata']

@pytest.mark.asyncio
async def test_submit_job_as_operation_field_headers_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.SubmitJobRequest()
    request.project_id = 'project_id_value'
    request.region = 'region_value'
    with mock.patch.object(type(client.transport.submit_job_as_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.submit_job_as_operation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&region=region_value') in kw['metadata']

def test_submit_job_as_operation_flattened():
    if False:
        print('Hello World!')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.submit_job_as_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.submit_job_as_operation(project_id='project_id_value', region='region_value', job=jobs.Job(reference=jobs.JobReference(project_id='project_id_value')))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].project_id
        mock_val = 'project_id_value'
        assert arg == mock_val
        arg = args[0].region
        mock_val = 'region_value'
        assert arg == mock_val
        arg = args[0].job
        mock_val = jobs.Job(reference=jobs.JobReference(project_id='project_id_value'))
        assert arg == mock_val

def test_submit_job_as_operation_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.submit_job_as_operation(jobs.SubmitJobRequest(), project_id='project_id_value', region='region_value', job=jobs.Job(reference=jobs.JobReference(project_id='project_id_value')))

@pytest.mark.asyncio
async def test_submit_job_as_operation_flattened_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.submit_job_as_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.submit_job_as_operation(project_id='project_id_value', region='region_value', job=jobs.Job(reference=jobs.JobReference(project_id='project_id_value')))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].project_id
        mock_val = 'project_id_value'
        assert arg == mock_val
        arg = args[0].region
        mock_val = 'region_value'
        assert arg == mock_val
        arg = args[0].job
        mock_val = jobs.Job(reference=jobs.JobReference(project_id='project_id_value'))
        assert arg == mock_val

@pytest.mark.asyncio
async def test_submit_job_as_operation_flattened_error_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.submit_job_as_operation(jobs.SubmitJobRequest(), project_id='project_id_value', region='region_value', job=jobs.Job(reference=jobs.JobReference(project_id='project_id_value')))

@pytest.mark.parametrize('request_type', [jobs.GetJobRequest, dict])
def test_get_job(request_type, transport: str='grpc'):
    if False:
        return 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = jobs.Job(driver_output_resource_uri='driver_output_resource_uri_value', driver_control_files_uri='driver_control_files_uri_value', job_uuid='job_uuid_value', done=True)
        response = client.get_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.GetJobRequest()
    assert isinstance(response, jobs.Job)
    assert response.driver_output_resource_uri == 'driver_output_resource_uri_value'
    assert response.driver_control_files_uri == 'driver_control_files_uri_value'
    assert response.job_uuid == 'job_uuid_value'
    assert response.done is True

def test_get_job_empty_call():
    if False:
        print('Hello World!')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        client.get_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.GetJobRequest()

@pytest.mark.asyncio
async def test_get_job_async(transport: str='grpc_asyncio', request_type=jobs.GetJobRequest):
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.Job(driver_output_resource_uri='driver_output_resource_uri_value', driver_control_files_uri='driver_control_files_uri_value', job_uuid='job_uuid_value', done=True))
        response = await client.get_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.GetJobRequest()
    assert isinstance(response, jobs.Job)
    assert response.driver_output_resource_uri == 'driver_output_resource_uri_value'
    assert response.driver_control_files_uri == 'driver_control_files_uri_value'
    assert response.job_uuid == 'job_uuid_value'
    assert response.done is True

@pytest.mark.asyncio
async def test_get_job_async_from_dict():
    await test_get_job_async(request_type=dict)

def test_get_job_field_headers():
    if False:
        i = 10
        return i + 15
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.GetJobRequest()
    request.project_id = 'project_id_value'
    request.region = 'region_value'
    request.job_id = 'job_id_value'
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = jobs.Job()
        client.get_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&region=region_value&job_id=job_id_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_job_field_headers_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.GetJobRequest()
    request.project_id = 'project_id_value'
    request.region = 'region_value'
    request.job_id = 'job_id_value'
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.Job())
        await client.get_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&region=region_value&job_id=job_id_value') in kw['metadata']

def test_get_job_flattened():
    if False:
        print('Hello World!')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = jobs.Job()
        client.get_job(project_id='project_id_value', region='region_value', job_id='job_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].project_id
        mock_val = 'project_id_value'
        assert arg == mock_val
        arg = args[0].region
        mock_val = 'region_value'
        assert arg == mock_val
        arg = args[0].job_id
        mock_val = 'job_id_value'
        assert arg == mock_val

def test_get_job_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_job(jobs.GetJobRequest(), project_id='project_id_value', region='region_value', job_id='job_id_value')

@pytest.mark.asyncio
async def test_get_job_flattened_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = jobs.Job()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.Job())
        response = await client.get_job(project_id='project_id_value', region='region_value', job_id='job_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].project_id
        mock_val = 'project_id_value'
        assert arg == mock_val
        arg = args[0].region
        mock_val = 'region_value'
        assert arg == mock_val
        arg = args[0].job_id
        mock_val = 'job_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_job_flattened_error_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_job(jobs.GetJobRequest(), project_id='project_id_value', region='region_value', job_id='job_id_value')

@pytest.mark.parametrize('request_type', [jobs.ListJobsRequest, dict])
def test_list_jobs(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = jobs.ListJobsResponse(next_page_token='next_page_token_value')
        response = client.list_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.ListJobsRequest()
    assert isinstance(response, pagers.ListJobsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_jobs_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        client.list_jobs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.ListJobsRequest()

@pytest.mark.asyncio
async def test_list_jobs_async(transport: str='grpc_asyncio', request_type=jobs.ListJobsRequest):
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.ListJobsResponse(next_page_token='next_page_token_value'))
        response = await client.list_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.ListJobsRequest()
    assert isinstance(response, pagers.ListJobsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_jobs_async_from_dict():
    await test_list_jobs_async(request_type=dict)

def test_list_jobs_field_headers():
    if False:
        i = 10
        return i + 15
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.ListJobsRequest()
    request.project_id = 'project_id_value'
    request.region = 'region_value'
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = jobs.ListJobsResponse()
        client.list_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&region=region_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_jobs_field_headers_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.ListJobsRequest()
    request.project_id = 'project_id_value'
    request.region = 'region_value'
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.ListJobsResponse())
        await client.list_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&region=region_value') in kw['metadata']

def test_list_jobs_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = jobs.ListJobsResponse()
        client.list_jobs(project_id='project_id_value', region='region_value', filter='filter_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].project_id
        mock_val = 'project_id_value'
        assert arg == mock_val
        arg = args[0].region
        mock_val = 'region_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

def test_list_jobs_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_jobs(jobs.ListJobsRequest(), project_id='project_id_value', region='region_value', filter='filter_value')

@pytest.mark.asyncio
async def test_list_jobs_flattened_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = jobs.ListJobsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.ListJobsResponse())
        response = await client.list_jobs(project_id='project_id_value', region='region_value', filter='filter_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].project_id
        mock_val = 'project_id_value'
        assert arg == mock_val
        arg = args[0].region
        mock_val = 'region_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_jobs_flattened_error_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_jobs(jobs.ListJobsRequest(), project_id='project_id_value', region='region_value', filter='filter_value')

def test_list_jobs_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.side_effect = (jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job(), jobs.Job()], next_page_token='abc'), jobs.ListJobsResponse(jobs=[], next_page_token='def'), jobs.ListJobsResponse(jobs=[jobs.Job()], next_page_token='ghi'), jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('project_id', ''), ('region', ''))),)
        pager = client.list_jobs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, jobs.Job) for i in results))

def test_list_jobs_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.side_effect = (jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job(), jobs.Job()], next_page_token='abc'), jobs.ListJobsResponse(jobs=[], next_page_token='def'), jobs.ListJobsResponse(jobs=[jobs.Job()], next_page_token='ghi'), jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job()]), RuntimeError)
        pages = list(client.list_jobs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_jobs_async_pager():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job(), jobs.Job()], next_page_token='abc'), jobs.ListJobsResponse(jobs=[], next_page_token='def'), jobs.ListJobsResponse(jobs=[jobs.Job()], next_page_token='ghi'), jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job()]), RuntimeError)
        async_pager = await client.list_jobs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, jobs.Job) for i in responses))

@pytest.mark.asyncio
async def test_list_jobs_async_pages():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job(), jobs.Job()], next_page_token='abc'), jobs.ListJobsResponse(jobs=[], next_page_token='def'), jobs.ListJobsResponse(jobs=[jobs.Job()], next_page_token='ghi'), jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_jobs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [jobs.UpdateJobRequest, dict])
def test_update_job(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = jobs.Job(driver_output_resource_uri='driver_output_resource_uri_value', driver_control_files_uri='driver_control_files_uri_value', job_uuid='job_uuid_value', done=True)
        response = client.update_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.UpdateJobRequest()
    assert isinstance(response, jobs.Job)
    assert response.driver_output_resource_uri == 'driver_output_resource_uri_value'
    assert response.driver_control_files_uri == 'driver_control_files_uri_value'
    assert response.job_uuid == 'job_uuid_value'
    assert response.done is True

def test_update_job_empty_call():
    if False:
        while True:
            i = 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        client.update_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.UpdateJobRequest()

@pytest.mark.asyncio
async def test_update_job_async(transport: str='grpc_asyncio', request_type=jobs.UpdateJobRequest):
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.Job(driver_output_resource_uri='driver_output_resource_uri_value', driver_control_files_uri='driver_control_files_uri_value', job_uuid='job_uuid_value', done=True))
        response = await client.update_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.UpdateJobRequest()
    assert isinstance(response, jobs.Job)
    assert response.driver_output_resource_uri == 'driver_output_resource_uri_value'
    assert response.driver_control_files_uri == 'driver_control_files_uri_value'
    assert response.job_uuid == 'job_uuid_value'
    assert response.done is True

@pytest.mark.asyncio
async def test_update_job_async_from_dict():
    await test_update_job_async(request_type=dict)

def test_update_job_field_headers():
    if False:
        return 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.UpdateJobRequest()
    request.project_id = 'project_id_value'
    request.region = 'region_value'
    request.job_id = 'job_id_value'
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = jobs.Job()
        client.update_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&region=region_value&job_id=job_id_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_job_field_headers_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.UpdateJobRequest()
    request.project_id = 'project_id_value'
    request.region = 'region_value'
    request.job_id = 'job_id_value'
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.Job())
        await client.update_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&region=region_value&job_id=job_id_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [jobs.CancelJobRequest, dict])
def test_cancel_job(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.cancel_job), '__call__') as call:
        call.return_value = jobs.Job(driver_output_resource_uri='driver_output_resource_uri_value', driver_control_files_uri='driver_control_files_uri_value', job_uuid='job_uuid_value', done=True)
        response = client.cancel_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.CancelJobRequest()
    assert isinstance(response, jobs.Job)
    assert response.driver_output_resource_uri == 'driver_output_resource_uri_value'
    assert response.driver_control_files_uri == 'driver_control_files_uri_value'
    assert response.job_uuid == 'job_uuid_value'
    assert response.done is True

def test_cancel_job_empty_call():
    if False:
        while True:
            i = 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.cancel_job), '__call__') as call:
        client.cancel_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.CancelJobRequest()

@pytest.mark.asyncio
async def test_cancel_job_async(transport: str='grpc_asyncio', request_type=jobs.CancelJobRequest):
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.cancel_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.Job(driver_output_resource_uri='driver_output_resource_uri_value', driver_control_files_uri='driver_control_files_uri_value', job_uuid='job_uuid_value', done=True))
        response = await client.cancel_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.CancelJobRequest()
    assert isinstance(response, jobs.Job)
    assert response.driver_output_resource_uri == 'driver_output_resource_uri_value'
    assert response.driver_control_files_uri == 'driver_control_files_uri_value'
    assert response.job_uuid == 'job_uuid_value'
    assert response.done is True

@pytest.mark.asyncio
async def test_cancel_job_async_from_dict():
    await test_cancel_job_async(request_type=dict)

def test_cancel_job_field_headers():
    if False:
        while True:
            i = 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.CancelJobRequest()
    request.project_id = 'project_id_value'
    request.region = 'region_value'
    request.job_id = 'job_id_value'
    with mock.patch.object(type(client.transport.cancel_job), '__call__') as call:
        call.return_value = jobs.Job()
        client.cancel_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&region=region_value&job_id=job_id_value') in kw['metadata']

@pytest.mark.asyncio
async def test_cancel_job_field_headers_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.CancelJobRequest()
    request.project_id = 'project_id_value'
    request.region = 'region_value'
    request.job_id = 'job_id_value'
    with mock.patch.object(type(client.transport.cancel_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.Job())
        await client.cancel_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&region=region_value&job_id=job_id_value') in kw['metadata']

def test_cancel_job_flattened():
    if False:
        while True:
            i = 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_job), '__call__') as call:
        call.return_value = jobs.Job()
        client.cancel_job(project_id='project_id_value', region='region_value', job_id='job_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].project_id
        mock_val = 'project_id_value'
        assert arg == mock_val
        arg = args[0].region
        mock_val = 'region_value'
        assert arg == mock_val
        arg = args[0].job_id
        mock_val = 'job_id_value'
        assert arg == mock_val

def test_cancel_job_flattened_error():
    if False:
        while True:
            i = 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.cancel_job(jobs.CancelJobRequest(), project_id='project_id_value', region='region_value', job_id='job_id_value')

@pytest.mark.asyncio
async def test_cancel_job_flattened_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_job), '__call__') as call:
        call.return_value = jobs.Job()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.Job())
        response = await client.cancel_job(project_id='project_id_value', region='region_value', job_id='job_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].project_id
        mock_val = 'project_id_value'
        assert arg == mock_val
        arg = args[0].region
        mock_val = 'region_value'
        assert arg == mock_val
        arg = args[0].job_id
        mock_val = 'job_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_cancel_job_flattened_error_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.cancel_job(jobs.CancelJobRequest(), project_id='project_id_value', region='region_value', job_id='job_id_value')

@pytest.mark.parametrize('request_type', [jobs.DeleteJobRequest, dict])
def test_delete_job(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        call.return_value = None
        response = client.delete_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.DeleteJobRequest()
    assert response is None

def test_delete_job_empty_call():
    if False:
        return 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        client.delete_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.DeleteJobRequest()

@pytest.mark.asyncio
async def test_delete_job_async(transport: str='grpc_asyncio', request_type=jobs.DeleteJobRequest):
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.DeleteJobRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_job_async_from_dict():
    await test_delete_job_async(request_type=dict)

def test_delete_job_field_headers():
    if False:
        while True:
            i = 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.DeleteJobRequest()
    request.project_id = 'project_id_value'
    request.region = 'region_value'
    request.job_id = 'job_id_value'
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        call.return_value = None
        client.delete_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&region=region_value&job_id=job_id_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_job_field_headers_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.DeleteJobRequest()
    request.project_id = 'project_id_value'
    request.region = 'region_value'
    request.job_id = 'job_id_value'
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&region=region_value&job_id=job_id_value') in kw['metadata']

def test_delete_job_flattened():
    if False:
        return 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        call.return_value = None
        client.delete_job(project_id='project_id_value', region='region_value', job_id='job_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].project_id
        mock_val = 'project_id_value'
        assert arg == mock_val
        arg = args[0].region
        mock_val = 'region_value'
        assert arg == mock_val
        arg = args[0].job_id
        mock_val = 'job_id_value'
        assert arg == mock_val

def test_delete_job_flattened_error():
    if False:
        i = 10
        return i + 15
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_job(jobs.DeleteJobRequest(), project_id='project_id_value', region='region_value', job_id='job_id_value')

@pytest.mark.asyncio
async def test_delete_job_flattened_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_job(project_id='project_id_value', region='region_value', job_id='job_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].project_id
        mock_val = 'project_id_value'
        assert arg == mock_val
        arg = args[0].region
        mock_val = 'region_value'
        assert arg == mock_val
        arg = args[0].job_id
        mock_val = 'job_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_job_flattened_error_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_job(jobs.DeleteJobRequest(), project_id='project_id_value', region='region_value', job_id='job_id_value')

@pytest.mark.parametrize('request_type', [jobs.SubmitJobRequest, dict])
def test_submit_job_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project_id': 'sample1', 'region': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = jobs.Job(driver_output_resource_uri='driver_output_resource_uri_value', driver_control_files_uri='driver_control_files_uri_value', job_uuid='job_uuid_value', done=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = jobs.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.submit_job(request)
    assert isinstance(response, jobs.Job)
    assert response.driver_output_resource_uri == 'driver_output_resource_uri_value'
    assert response.driver_control_files_uri == 'driver_control_files_uri_value'
    assert response.job_uuid == 'job_uuid_value'
    assert response.done is True

def test_submit_job_rest_required_fields(request_type=jobs.SubmitJobRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.JobControllerRestTransport
    request_init = {}
    request_init['project_id'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).submit_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['projectId'] = 'project_id_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).submit_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'projectId' in jsonified_request
    assert jsonified_request['projectId'] == 'project_id_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = jobs.Job()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = jobs.Job.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.submit_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_submit_job_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.JobControllerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.submit_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('projectId', 'region', 'job'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_submit_job_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.JobControllerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobControllerRestInterceptor())
    client = JobControllerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobControllerRestInterceptor, 'post_submit_job') as post, mock.patch.object(transports.JobControllerRestInterceptor, 'pre_submit_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = jobs.SubmitJobRequest.pb(jobs.SubmitJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = jobs.Job.to_json(jobs.Job())
        request = jobs.SubmitJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = jobs.Job()
        client.submit_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_submit_job_rest_bad_request(transport: str='rest', request_type=jobs.SubmitJobRequest):
    if False:
        i = 10
        return i + 15
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project_id': 'sample1', 'region': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.submit_job(request)

def test_submit_job_rest_flattened():
    if False:
        return 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = jobs.Job()
        sample_request = {'project_id': 'sample1', 'region': 'sample2'}
        mock_args = dict(project_id='project_id_value', region='region_value', job=jobs.Job(reference=jobs.JobReference(project_id='project_id_value')))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = jobs.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.submit_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/projects/{project_id}/regions/{region}/jobs:submit' % client.transport._host, args[1])

def test_submit_job_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.submit_job(jobs.SubmitJobRequest(), project_id='project_id_value', region='region_value', job=jobs.Job(reference=jobs.JobReference(project_id='project_id_value')))

def test_submit_job_rest_error():
    if False:
        return 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [jobs.SubmitJobRequest, dict])
def test_submit_job_as_operation_rest(request_type):
    if False:
        print('Hello World!')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project_id': 'sample1', 'region': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.submit_job_as_operation(request)
    assert response.operation.name == 'operations/spam'

def test_submit_job_as_operation_rest_required_fields(request_type=jobs.SubmitJobRequest):
    if False:
        return 10
    transport_class = transports.JobControllerRestTransport
    request_init = {}
    request_init['project_id'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).submit_job_as_operation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['projectId'] = 'project_id_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).submit_job_as_operation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'projectId' in jsonified_request
    assert jsonified_request['projectId'] == 'project_id_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.submit_job_as_operation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_submit_job_as_operation_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.JobControllerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.submit_job_as_operation._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('projectId', 'region', 'job'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_submit_job_as_operation_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.JobControllerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobControllerRestInterceptor())
    client = JobControllerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.JobControllerRestInterceptor, 'post_submit_job_as_operation') as post, mock.patch.object(transports.JobControllerRestInterceptor, 'pre_submit_job_as_operation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = jobs.SubmitJobRequest.pb(jobs.SubmitJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = jobs.SubmitJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.submit_job_as_operation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_submit_job_as_operation_rest_bad_request(transport: str='rest', request_type=jobs.SubmitJobRequest):
    if False:
        return 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project_id': 'sample1', 'region': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.submit_job_as_operation(request)

def test_submit_job_as_operation_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'project_id': 'sample1', 'region': 'sample2'}
        mock_args = dict(project_id='project_id_value', region='region_value', job=jobs.Job(reference=jobs.JobReference(project_id='project_id_value')))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.submit_job_as_operation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/projects/{project_id}/regions/{region}/jobs:submitAsOperation' % client.transport._host, args[1])

def test_submit_job_as_operation_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.submit_job_as_operation(jobs.SubmitJobRequest(), project_id='project_id_value', region='region_value', job=jobs.Job(reference=jobs.JobReference(project_id='project_id_value')))

def test_submit_job_as_operation_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [jobs.GetJobRequest, dict])
def test_get_job_rest(request_type):
    if False:
        return 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project_id': 'sample1', 'region': 'sample2', 'job_id': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = jobs.Job(driver_output_resource_uri='driver_output_resource_uri_value', driver_control_files_uri='driver_control_files_uri_value', job_uuid='job_uuid_value', done=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = jobs.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_job(request)
    assert isinstance(response, jobs.Job)
    assert response.driver_output_resource_uri == 'driver_output_resource_uri_value'
    assert response.driver_control_files_uri == 'driver_control_files_uri_value'
    assert response.job_uuid == 'job_uuid_value'
    assert response.done is True

def test_get_job_rest_required_fields(request_type=jobs.GetJobRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.JobControllerRestTransport
    request_init = {}
    request_init['project_id'] = ''
    request_init['region'] = ''
    request_init['job_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['projectId'] = 'project_id_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['jobId'] = 'job_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'projectId' in jsonified_request
    assert jsonified_request['projectId'] == 'project_id_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'jobId' in jsonified_request
    assert jsonified_request['jobId'] == 'job_id_value'
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = jobs.Job()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = jobs.Job.pb(return_value)
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
    transport = transports.JobControllerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('projectId', 'region', 'jobId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_job_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.JobControllerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobControllerRestInterceptor())
    client = JobControllerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobControllerRestInterceptor, 'post_get_job') as post, mock.patch.object(transports.JobControllerRestInterceptor, 'pre_get_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = jobs.GetJobRequest.pb(jobs.GetJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = jobs.Job.to_json(jobs.Job())
        request = jobs.GetJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = jobs.Job()
        client.get_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_job_rest_bad_request(transport: str='rest', request_type=jobs.GetJobRequest):
    if False:
        print('Hello World!')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project_id': 'sample1', 'region': 'sample2', 'job_id': 'sample3'}
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = jobs.Job()
        sample_request = {'project_id': 'sample1', 'region': 'sample2', 'job_id': 'sample3'}
        mock_args = dict(project_id='project_id_value', region='region_value', job_id='job_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = jobs.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/projects/{project_id}/regions/{region}/jobs/{job_id}' % client.transport._host, args[1])

def test_get_job_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_job(jobs.GetJobRequest(), project_id='project_id_value', region='region_value', job_id='job_id_value')

def test_get_job_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [jobs.ListJobsRequest, dict])
def test_list_jobs_rest(request_type):
    if False:
        while True:
            i = 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project_id': 'sample1', 'region': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = jobs.ListJobsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = jobs.ListJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_jobs(request)
    assert isinstance(response, pagers.ListJobsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_jobs_rest_required_fields(request_type=jobs.ListJobsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.JobControllerRestTransport
    request_init = {}
    request_init['project_id'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['projectId'] = 'project_id_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_jobs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('cluster_name', 'filter', 'job_state_matcher', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'projectId' in jsonified_request
    assert jsonified_request['projectId'] == 'project_id_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = jobs.ListJobsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = jobs.ListJobsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_jobs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_jobs_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.JobControllerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_jobs._get_unset_required_fields({})
    assert set(unset_fields) == set(('clusterName', 'filter', 'jobStateMatcher', 'pageSize', 'pageToken')) & set(('projectId', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_jobs_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.JobControllerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobControllerRestInterceptor())
    client = JobControllerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobControllerRestInterceptor, 'post_list_jobs') as post, mock.patch.object(transports.JobControllerRestInterceptor, 'pre_list_jobs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = jobs.ListJobsRequest.pb(jobs.ListJobsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = jobs.ListJobsResponse.to_json(jobs.ListJobsResponse())
        request = jobs.ListJobsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = jobs.ListJobsResponse()
        client.list_jobs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_jobs_rest_bad_request(transport: str='rest', request_type=jobs.ListJobsRequest):
    if False:
        print('Hello World!')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project_id': 'sample1', 'region': 'sample2'}
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = jobs.ListJobsResponse()
        sample_request = {'project_id': 'sample1', 'region': 'sample2'}
        mock_args = dict(project_id='project_id_value', region='region_value', filter='filter_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = jobs.ListJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_jobs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/projects/{project_id}/regions/{region}/jobs' % client.transport._host, args[1])

def test_list_jobs_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_jobs(jobs.ListJobsRequest(), project_id='project_id_value', region='region_value', filter='filter_value')

def test_list_jobs_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job(), jobs.Job()], next_page_token='abc'), jobs.ListJobsResponse(jobs=[], next_page_token='def'), jobs.ListJobsResponse(jobs=[jobs.Job()], next_page_token='ghi'), jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job()]))
        response = response + response
        response = tuple((jobs.ListJobsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'project_id': 'sample1', 'region': 'sample2'}
        pager = client.list_jobs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, jobs.Job) for i in results))
        pages = list(client.list_jobs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [jobs.UpdateJobRequest, dict])
def test_update_job_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project_id': 'sample1', 'region': 'sample2', 'job_id': 'sample3'}
    request_init['job'] = {'reference': {'project_id': 'project_id_value', 'job_id': 'job_id_value'}, 'placement': {'cluster_name': 'cluster_name_value', 'cluster_uuid': 'cluster_uuid_value', 'cluster_labels': {}}, 'hadoop_job': {'main_jar_file_uri': 'main_jar_file_uri_value', 'main_class': 'main_class_value', 'args': ['args_value1', 'args_value2'], 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'file_uris': ['file_uris_value1', 'file_uris_value2'], 'archive_uris': ['archive_uris_value1', 'archive_uris_value2'], 'properties': {}, 'logging_config': {'driver_log_levels': {}}}, 'spark_job': {'main_jar_file_uri': 'main_jar_file_uri_value', 'main_class': 'main_class_value', 'args': ['args_value1', 'args_value2'], 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'file_uris': ['file_uris_value1', 'file_uris_value2'], 'archive_uris': ['archive_uris_value1', 'archive_uris_value2'], 'properties': {}, 'logging_config': {}}, 'pyspark_job': {'main_python_file_uri': 'main_python_file_uri_value', 'args': ['args_value1', 'args_value2'], 'python_file_uris': ['python_file_uris_value1', 'python_file_uris_value2'], 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'file_uris': ['file_uris_value1', 'file_uris_value2'], 'archive_uris': ['archive_uris_value1', 'archive_uris_value2'], 'properties': {}, 'logging_config': {}}, 'hive_job': {'query_file_uri': 'query_file_uri_value', 'query_list': {'queries': ['queries_value1', 'queries_value2']}, 'continue_on_failure': True, 'script_variables': {}, 'properties': {}, 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2']}, 'pig_job': {'query_file_uri': 'query_file_uri_value', 'query_list': {}, 'continue_on_failure': True, 'script_variables': {}, 'properties': {}, 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'logging_config': {}}, 'spark_r_job': {'main_r_file_uri': 'main_r_file_uri_value', 'args': ['args_value1', 'args_value2'], 'file_uris': ['file_uris_value1', 'file_uris_value2'], 'archive_uris': ['archive_uris_value1', 'archive_uris_value2'], 'properties': {}, 'logging_config': {}}, 'spark_sql_job': {'query_file_uri': 'query_file_uri_value', 'query_list': {}, 'script_variables': {}, 'properties': {}, 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'logging_config': {}}, 'presto_job': {'query_file_uri': 'query_file_uri_value', 'query_list': {}, 'continue_on_failure': True, 'output_format': 'output_format_value', 'client_tags': ['client_tags_value1', 'client_tags_value2'], 'properties': {}, 'logging_config': {}}, 'trino_job': {'query_file_uri': 'query_file_uri_value', 'query_list': {}, 'continue_on_failure': True, 'output_format': 'output_format_value', 'client_tags': ['client_tags_value1', 'client_tags_value2'], 'properties': {}, 'logging_config': {}}, 'status': {'state': 1, 'details': 'details_value', 'state_start_time': {'seconds': 751, 'nanos': 543}, 'substate': 1}, 'status_history': {}, 'yarn_applications': [{'name': 'name_value', 'state': 1, 'progress': 0.885, 'tracking_url': 'tracking_url_value'}], 'driver_output_resource_uri': 'driver_output_resource_uri_value', 'driver_control_files_uri': 'driver_control_files_uri_value', 'labels': {}, 'scheduling': {'max_failures_per_hour': 2243, 'max_failures_total': 1923}, 'job_uuid': 'job_uuid_value', 'done': True, 'driver_scheduling_config': {'memory_mb': 967, 'vcores': 658}}
    test_field = jobs.UpdateJobRequest.meta.fields['job']

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
        return_value = jobs.Job(driver_output_resource_uri='driver_output_resource_uri_value', driver_control_files_uri='driver_control_files_uri_value', job_uuid='job_uuid_value', done=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = jobs.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_job(request)
    assert isinstance(response, jobs.Job)
    assert response.driver_output_resource_uri == 'driver_output_resource_uri_value'
    assert response.driver_control_files_uri == 'driver_control_files_uri_value'
    assert response.job_uuid == 'job_uuid_value'
    assert response.done is True

def test_update_job_rest_required_fields(request_type=jobs.UpdateJobRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.JobControllerRestTransport
    request_init = {}
    request_init['project_id'] = ''
    request_init['region'] = ''
    request_init['job_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['projectId'] = 'project_id_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['jobId'] = 'job_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_job._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    assert 'projectId' in jsonified_request
    assert jsonified_request['projectId'] == 'project_id_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'jobId' in jsonified_request
    assert jsonified_request['jobId'] == 'job_id_value'
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = jobs.Job()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = jobs.Job.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_job_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.JobControllerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_job._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('projectId', 'region', 'jobId', 'job', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_job_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.JobControllerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobControllerRestInterceptor())
    client = JobControllerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobControllerRestInterceptor, 'post_update_job') as post, mock.patch.object(transports.JobControllerRestInterceptor, 'pre_update_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = jobs.UpdateJobRequest.pb(jobs.UpdateJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = jobs.Job.to_json(jobs.Job())
        request = jobs.UpdateJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = jobs.Job()
        client.update_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_job_rest_bad_request(transport: str='rest', request_type=jobs.UpdateJobRequest):
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project_id': 'sample1', 'region': 'sample2', 'job_id': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_job(request)

def test_update_job_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [jobs.CancelJobRequest, dict])
def test_cancel_job_rest(request_type):
    if False:
        return 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project_id': 'sample1', 'region': 'sample2', 'job_id': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = jobs.Job(driver_output_resource_uri='driver_output_resource_uri_value', driver_control_files_uri='driver_control_files_uri_value', job_uuid='job_uuid_value', done=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = jobs.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.cancel_job(request)
    assert isinstance(response, jobs.Job)
    assert response.driver_output_resource_uri == 'driver_output_resource_uri_value'
    assert response.driver_control_files_uri == 'driver_control_files_uri_value'
    assert response.job_uuid == 'job_uuid_value'
    assert response.done is True

def test_cancel_job_rest_required_fields(request_type=jobs.CancelJobRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.JobControllerRestTransport
    request_init = {}
    request_init['project_id'] = ''
    request_init['region'] = ''
    request_init['job_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).cancel_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['projectId'] = 'project_id_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['jobId'] = 'job_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).cancel_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'projectId' in jsonified_request
    assert jsonified_request['projectId'] == 'project_id_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'jobId' in jsonified_request
    assert jsonified_request['jobId'] == 'job_id_value'
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = jobs.Job()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = jobs.Job.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.cancel_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_cancel_job_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.JobControllerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.cancel_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('projectId', 'region', 'jobId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_cancel_job_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.JobControllerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobControllerRestInterceptor())
    client = JobControllerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobControllerRestInterceptor, 'post_cancel_job') as post, mock.patch.object(transports.JobControllerRestInterceptor, 'pre_cancel_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = jobs.CancelJobRequest.pb(jobs.CancelJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = jobs.Job.to_json(jobs.Job())
        request = jobs.CancelJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = jobs.Job()
        client.cancel_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_cancel_job_rest_bad_request(transport: str='rest', request_type=jobs.CancelJobRequest):
    if False:
        while True:
            i = 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project_id': 'sample1', 'region': 'sample2', 'job_id': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.cancel_job(request)

def test_cancel_job_rest_flattened():
    if False:
        return 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = jobs.Job()
        sample_request = {'project_id': 'sample1', 'region': 'sample2', 'job_id': 'sample3'}
        mock_args = dict(project_id='project_id_value', region='region_value', job_id='job_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = jobs.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.cancel_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/projects/{project_id}/regions/{region}/jobs/{job_id}:cancel' % client.transport._host, args[1])

def test_cancel_job_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.cancel_job(jobs.CancelJobRequest(), project_id='project_id_value', region='region_value', job_id='job_id_value')

def test_cancel_job_rest_error():
    if False:
        return 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [jobs.DeleteJobRequest, dict])
def test_delete_job_rest(request_type):
    if False:
        return 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project_id': 'sample1', 'region': 'sample2', 'job_id': 'sample3'}
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

def test_delete_job_rest_required_fields(request_type=jobs.DeleteJobRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.JobControllerRestTransport
    request_init = {}
    request_init['project_id'] = ''
    request_init['region'] = ''
    request_init['job_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['projectId'] = 'project_id_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['jobId'] = 'job_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'projectId' in jsonified_request
    assert jsonified_request['projectId'] == 'project_id_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'jobId' in jsonified_request
    assert jsonified_request['jobId'] == 'job_id_value'
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    transport = transports.JobControllerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('projectId', 'region', 'jobId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_job_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.JobControllerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobControllerRestInterceptor())
    client = JobControllerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobControllerRestInterceptor, 'pre_delete_job') as pre:
        pre.assert_not_called()
        pb_message = jobs.DeleteJobRequest.pb(jobs.DeleteJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = jobs.DeleteJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_job_rest_bad_request(transport: str='rest', request_type=jobs.DeleteJobRequest):
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project_id': 'sample1', 'region': 'sample2', 'job_id': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_job(request)

def test_delete_job_rest_flattened():
    if False:
        while True:
            i = 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'project_id': 'sample1', 'region': 'sample2', 'job_id': 'sample3'}
        mock_args = dict(project_id='project_id_value', region='region_value', job_id='job_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/projects/{project_id}/regions/{region}/jobs/{job_id}' % client.transport._host, args[1])

def test_delete_job_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_job(jobs.DeleteJobRequest(), project_id='project_id_value', region='region_value', job_id='job_id_value')

def test_delete_job_rest_error():
    if False:
        i = 10
        return i + 15
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.JobControllerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.JobControllerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = JobControllerClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.JobControllerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = JobControllerClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = JobControllerClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.JobControllerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = JobControllerClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.JobControllerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = JobControllerClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.JobControllerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.JobControllerGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.JobControllerGrpcTransport, transports.JobControllerGrpcAsyncIOTransport, transports.JobControllerRestTransport])
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
        return 10
    transport = JobControllerClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.JobControllerGrpcTransport)

def test_job_controller_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.JobControllerTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_job_controller_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.dataproc_v1.services.job_controller.transports.JobControllerTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.JobControllerTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('submit_job', 'submit_job_as_operation', 'get_job', 'list_jobs', 'update_job', 'cancel_job', 'delete_job', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_job_controller_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.dataproc_v1.services.job_controller.transports.JobControllerTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.JobControllerTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_job_controller_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.dataproc_v1.services.job_controller.transports.JobControllerTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.JobControllerTransport()
        adc.assert_called_once()

def test_job_controller_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        JobControllerClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.JobControllerGrpcTransport, transports.JobControllerGrpcAsyncIOTransport])
def test_job_controller_transport_auth_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.JobControllerGrpcTransport, transports.JobControllerGrpcAsyncIOTransport, transports.JobControllerRestTransport])
def test_job_controller_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.JobControllerGrpcTransport, grpc_helpers), (transports.JobControllerGrpcAsyncIOTransport, grpc_helpers_async)])
def test_job_controller_transport_create_channel(transport_class, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('dataproc.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='dataproc.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.JobControllerGrpcTransport, transports.JobControllerGrpcAsyncIOTransport])
def test_job_controller_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_job_controller_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.JobControllerRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_job_controller_rest_lro_client():
    if False:
        i = 10
        return i + 15
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_job_controller_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dataproc.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('dataproc.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dataproc.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_job_controller_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dataproc.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('dataproc.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dataproc.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_job_controller_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = JobControllerClient(credentials=creds1, transport=transport_name)
    client2 = JobControllerClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.submit_job._session
    session2 = client2.transport.submit_job._session
    assert session1 != session2
    session1 = client1.transport.submit_job_as_operation._session
    session2 = client2.transport.submit_job_as_operation._session
    assert session1 != session2
    session1 = client1.transport.get_job._session
    session2 = client2.transport.get_job._session
    assert session1 != session2
    session1 = client1.transport.list_jobs._session
    session2 = client2.transport.list_jobs._session
    assert session1 != session2
    session1 = client1.transport.update_job._session
    session2 = client2.transport.update_job._session
    assert session1 != session2
    session1 = client1.transport.cancel_job._session
    session2 = client2.transport.cancel_job._session
    assert session1 != session2
    session1 = client1.transport.delete_job._session
    session2 = client2.transport.delete_job._session
    assert session1 != session2

def test_job_controller_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.JobControllerGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_job_controller_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.JobControllerGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.JobControllerGrpcTransport, transports.JobControllerGrpcAsyncIOTransport])
def test_job_controller_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.JobControllerGrpcTransport, transports.JobControllerGrpcAsyncIOTransport])
def test_job_controller_transport_channel_mtls_with_adc(transport_class):
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

def test_job_controller_grpc_lro_client():
    if False:
        while True:
            i = 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_job_controller_grpc_lro_async_client():
    if False:
        while True:
            i = 10
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = JobControllerClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'clam'}
    path = JobControllerClient.common_billing_account_path(**expected)
    actual = JobControllerClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = JobControllerClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'octopus'}
    path = JobControllerClient.common_folder_path(**expected)
    actual = JobControllerClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = JobControllerClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'nudibranch'}
    path = JobControllerClient.common_organization_path(**expected)
    actual = JobControllerClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = JobControllerClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'mussel'}
    path = JobControllerClient.common_project_path(**expected)
    actual = JobControllerClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = JobControllerClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = JobControllerClient.common_location_path(**expected)
    actual = JobControllerClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.JobControllerTransport, '_prep_wrapped_messages') as prep:
        client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.JobControllerTransport, '_prep_wrapped_messages') as prep:
        transport_class = JobControllerClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        while True:
            i = 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        return 10
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = JobControllerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        while True:
            i = 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = JobControllerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(JobControllerClient, transports.JobControllerGrpcTransport), (JobControllerAsyncClient, transports.JobControllerGrpcAsyncIOTransport)])
def test_api_key_credentials(client_class, transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth._default, 'get_api_key_credentials', create=True) as get_api_key_credentials:
        mock_cred = mock.Mock()
        get_api_key_credentials.return_value = mock_cred
        options = client_options.ClientOptions()
        options.api_key = 'api_key'
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options)
            patched.assert_called_once_with(credentials=mock_cred, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)