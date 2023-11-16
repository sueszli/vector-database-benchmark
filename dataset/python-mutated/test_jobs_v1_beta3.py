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
from google.oauth2 import service_account
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import json_format
from google.protobuf import struct_pb2
from google.protobuf import timestamp_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.dataflow_v1beta3.services.jobs_v1_beta3 import JobsV1Beta3AsyncClient, JobsV1Beta3Client, pagers, transports
from google.cloud.dataflow_v1beta3.types import environment, jobs, snapshots

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        for i in range(10):
            print('nop')
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
    assert JobsV1Beta3Client._get_default_mtls_endpoint(None) is None
    assert JobsV1Beta3Client._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert JobsV1Beta3Client._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert JobsV1Beta3Client._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert JobsV1Beta3Client._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert JobsV1Beta3Client._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(JobsV1Beta3Client, 'grpc'), (JobsV1Beta3AsyncClient, 'grpc_asyncio'), (JobsV1Beta3Client, 'rest')])
def test_jobs_v1_beta3_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('dataflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dataflow.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.JobsV1Beta3GrpcTransport, 'grpc'), (transports.JobsV1Beta3GrpcAsyncIOTransport, 'grpc_asyncio'), (transports.JobsV1Beta3RestTransport, 'rest')])
def test_jobs_v1_beta3_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(JobsV1Beta3Client, 'grpc'), (JobsV1Beta3AsyncClient, 'grpc_asyncio'), (JobsV1Beta3Client, 'rest')])
def test_jobs_v1_beta3_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('dataflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dataflow.googleapis.com')

def test_jobs_v1_beta3_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = JobsV1Beta3Client.get_transport_class()
    available_transports = [transports.JobsV1Beta3GrpcTransport, transports.JobsV1Beta3RestTransport]
    assert transport in available_transports
    transport = JobsV1Beta3Client.get_transport_class('grpc')
    assert transport == transports.JobsV1Beta3GrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(JobsV1Beta3Client, transports.JobsV1Beta3GrpcTransport, 'grpc'), (JobsV1Beta3AsyncClient, transports.JobsV1Beta3GrpcAsyncIOTransport, 'grpc_asyncio'), (JobsV1Beta3Client, transports.JobsV1Beta3RestTransport, 'rest')])
@mock.patch.object(JobsV1Beta3Client, 'DEFAULT_ENDPOINT', modify_default_endpoint(JobsV1Beta3Client))
@mock.patch.object(JobsV1Beta3AsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(JobsV1Beta3AsyncClient))
def test_jobs_v1_beta3_client_client_options(client_class, transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(JobsV1Beta3Client, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(JobsV1Beta3Client, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(JobsV1Beta3Client, transports.JobsV1Beta3GrpcTransport, 'grpc', 'true'), (JobsV1Beta3AsyncClient, transports.JobsV1Beta3GrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (JobsV1Beta3Client, transports.JobsV1Beta3GrpcTransport, 'grpc', 'false'), (JobsV1Beta3AsyncClient, transports.JobsV1Beta3GrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (JobsV1Beta3Client, transports.JobsV1Beta3RestTransport, 'rest', 'true'), (JobsV1Beta3Client, transports.JobsV1Beta3RestTransport, 'rest', 'false')])
@mock.patch.object(JobsV1Beta3Client, 'DEFAULT_ENDPOINT', modify_default_endpoint(JobsV1Beta3Client))
@mock.patch.object(JobsV1Beta3AsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(JobsV1Beta3AsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_jobs_v1_beta3_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [JobsV1Beta3Client, JobsV1Beta3AsyncClient])
@mock.patch.object(JobsV1Beta3Client, 'DEFAULT_ENDPOINT', modify_default_endpoint(JobsV1Beta3Client))
@mock.patch.object(JobsV1Beta3AsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(JobsV1Beta3AsyncClient))
def test_jobs_v1_beta3_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(JobsV1Beta3Client, transports.JobsV1Beta3GrpcTransport, 'grpc'), (JobsV1Beta3AsyncClient, transports.JobsV1Beta3GrpcAsyncIOTransport, 'grpc_asyncio'), (JobsV1Beta3Client, transports.JobsV1Beta3RestTransport, 'rest')])
def test_jobs_v1_beta3_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(JobsV1Beta3Client, transports.JobsV1Beta3GrpcTransport, 'grpc', grpc_helpers), (JobsV1Beta3AsyncClient, transports.JobsV1Beta3GrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (JobsV1Beta3Client, transports.JobsV1Beta3RestTransport, 'rest', None)])
def test_jobs_v1_beta3_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_jobs_v1_beta3_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.dataflow_v1beta3.services.jobs_v1_beta3.transports.JobsV1Beta3GrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = JobsV1Beta3Client(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(JobsV1Beta3Client, transports.JobsV1Beta3GrpcTransport, 'grpc', grpc_helpers), (JobsV1Beta3AsyncClient, transports.JobsV1Beta3GrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_jobs_v1_beta3_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('dataflow.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/compute.readonly', 'https://www.googleapis.com/auth/userinfo.email'), scopes=None, default_host='dataflow.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [jobs.CreateJobRequest, dict])
def test_create_job(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = jobs.Job(id='id_value', project_id='project_id_value', name='name_value', type_=environment.JobType.JOB_TYPE_BATCH, steps_location='steps_location_value', current_state=jobs.JobState.JOB_STATE_STOPPED, requested_state=jobs.JobState.JOB_STATE_STOPPED, replace_job_id='replace_job_id_value', client_request_id='client_request_id_value', replaced_by_job_id='replaced_by_job_id_value', temp_files=['temp_files_value'], location='location_value', created_from_snapshot_id='created_from_snapshot_id_value', satisfies_pzs=True)
        response = client.create_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.CreateJobRequest()
    assert isinstance(response, jobs.Job)
    assert response.id == 'id_value'
    assert response.project_id == 'project_id_value'
    assert response.name == 'name_value'
    assert response.type_ == environment.JobType.JOB_TYPE_BATCH
    assert response.steps_location == 'steps_location_value'
    assert response.current_state == jobs.JobState.JOB_STATE_STOPPED
    assert response.requested_state == jobs.JobState.JOB_STATE_STOPPED
    assert response.replace_job_id == 'replace_job_id_value'
    assert response.client_request_id == 'client_request_id_value'
    assert response.replaced_by_job_id == 'replaced_by_job_id_value'
    assert response.temp_files == ['temp_files_value']
    assert response.location == 'location_value'
    assert response.created_from_snapshot_id == 'created_from_snapshot_id_value'
    assert response.satisfies_pzs is True

def test_create_job_empty_call():
    if False:
        while True:
            i = 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        client.create_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.CreateJobRequest()

@pytest.mark.asyncio
async def test_create_job_async(transport: str='grpc_asyncio', request_type=jobs.CreateJobRequest):
    client = JobsV1Beta3AsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.Job(id='id_value', project_id='project_id_value', name='name_value', type_=environment.JobType.JOB_TYPE_BATCH, steps_location='steps_location_value', current_state=jobs.JobState.JOB_STATE_STOPPED, requested_state=jobs.JobState.JOB_STATE_STOPPED, replace_job_id='replace_job_id_value', client_request_id='client_request_id_value', replaced_by_job_id='replaced_by_job_id_value', temp_files=['temp_files_value'], location='location_value', created_from_snapshot_id='created_from_snapshot_id_value', satisfies_pzs=True))
        response = await client.create_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.CreateJobRequest()
    assert isinstance(response, jobs.Job)
    assert response.id == 'id_value'
    assert response.project_id == 'project_id_value'
    assert response.name == 'name_value'
    assert response.type_ == environment.JobType.JOB_TYPE_BATCH
    assert response.steps_location == 'steps_location_value'
    assert response.current_state == jobs.JobState.JOB_STATE_STOPPED
    assert response.requested_state == jobs.JobState.JOB_STATE_STOPPED
    assert response.replace_job_id == 'replace_job_id_value'
    assert response.client_request_id == 'client_request_id_value'
    assert response.replaced_by_job_id == 'replaced_by_job_id_value'
    assert response.temp_files == ['temp_files_value']
    assert response.location == 'location_value'
    assert response.created_from_snapshot_id == 'created_from_snapshot_id_value'
    assert response.satisfies_pzs is True

@pytest.mark.asyncio
async def test_create_job_async_from_dict():
    await test_create_job_async(request_type=dict)

def test_create_job_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.CreateJobRequest()
    request.project_id = 'project_id_value'
    request.location = 'location_value'
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = jobs.Job()
        client.create_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&location=location_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_job_field_headers_async():
    client = JobsV1Beta3AsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.CreateJobRequest()
    request.project_id = 'project_id_value'
    request.location = 'location_value'
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.Job())
        await client.create_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&location=location_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [jobs.GetJobRequest, dict])
def test_get_job(request_type, transport: str='grpc'):
    if False:
        return 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = jobs.Job(id='id_value', project_id='project_id_value', name='name_value', type_=environment.JobType.JOB_TYPE_BATCH, steps_location='steps_location_value', current_state=jobs.JobState.JOB_STATE_STOPPED, requested_state=jobs.JobState.JOB_STATE_STOPPED, replace_job_id='replace_job_id_value', client_request_id='client_request_id_value', replaced_by_job_id='replaced_by_job_id_value', temp_files=['temp_files_value'], location='location_value', created_from_snapshot_id='created_from_snapshot_id_value', satisfies_pzs=True)
        response = client.get_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.GetJobRequest()
    assert isinstance(response, jobs.Job)
    assert response.id == 'id_value'
    assert response.project_id == 'project_id_value'
    assert response.name == 'name_value'
    assert response.type_ == environment.JobType.JOB_TYPE_BATCH
    assert response.steps_location == 'steps_location_value'
    assert response.current_state == jobs.JobState.JOB_STATE_STOPPED
    assert response.requested_state == jobs.JobState.JOB_STATE_STOPPED
    assert response.replace_job_id == 'replace_job_id_value'
    assert response.client_request_id == 'client_request_id_value'
    assert response.replaced_by_job_id == 'replaced_by_job_id_value'
    assert response.temp_files == ['temp_files_value']
    assert response.location == 'location_value'
    assert response.created_from_snapshot_id == 'created_from_snapshot_id_value'
    assert response.satisfies_pzs is True

def test_get_job_empty_call():
    if False:
        return 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        client.get_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.GetJobRequest()

@pytest.mark.asyncio
async def test_get_job_async(transport: str='grpc_asyncio', request_type=jobs.GetJobRequest):
    client = JobsV1Beta3AsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.Job(id='id_value', project_id='project_id_value', name='name_value', type_=environment.JobType.JOB_TYPE_BATCH, steps_location='steps_location_value', current_state=jobs.JobState.JOB_STATE_STOPPED, requested_state=jobs.JobState.JOB_STATE_STOPPED, replace_job_id='replace_job_id_value', client_request_id='client_request_id_value', replaced_by_job_id='replaced_by_job_id_value', temp_files=['temp_files_value'], location='location_value', created_from_snapshot_id='created_from_snapshot_id_value', satisfies_pzs=True))
        response = await client.get_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.GetJobRequest()
    assert isinstance(response, jobs.Job)
    assert response.id == 'id_value'
    assert response.project_id == 'project_id_value'
    assert response.name == 'name_value'
    assert response.type_ == environment.JobType.JOB_TYPE_BATCH
    assert response.steps_location == 'steps_location_value'
    assert response.current_state == jobs.JobState.JOB_STATE_STOPPED
    assert response.requested_state == jobs.JobState.JOB_STATE_STOPPED
    assert response.replace_job_id == 'replace_job_id_value'
    assert response.client_request_id == 'client_request_id_value'
    assert response.replaced_by_job_id == 'replaced_by_job_id_value'
    assert response.temp_files == ['temp_files_value']
    assert response.location == 'location_value'
    assert response.created_from_snapshot_id == 'created_from_snapshot_id_value'
    assert response.satisfies_pzs is True

@pytest.mark.asyncio
async def test_get_job_async_from_dict():
    await test_get_job_async(request_type=dict)

def test_get_job_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.GetJobRequest()
    request.project_id = 'project_id_value'
    request.location = 'location_value'
    request.job_id = 'job_id_value'
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = jobs.Job()
        client.get_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&location=location_value&job_id=job_id_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_job_field_headers_async():
    client = JobsV1Beta3AsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.GetJobRequest()
    request.project_id = 'project_id_value'
    request.location = 'location_value'
    request.job_id = 'job_id_value'
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.Job())
        await client.get_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&location=location_value&job_id=job_id_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [jobs.UpdateJobRequest, dict])
def test_update_job(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = jobs.Job(id='id_value', project_id='project_id_value', name='name_value', type_=environment.JobType.JOB_TYPE_BATCH, steps_location='steps_location_value', current_state=jobs.JobState.JOB_STATE_STOPPED, requested_state=jobs.JobState.JOB_STATE_STOPPED, replace_job_id='replace_job_id_value', client_request_id='client_request_id_value', replaced_by_job_id='replaced_by_job_id_value', temp_files=['temp_files_value'], location='location_value', created_from_snapshot_id='created_from_snapshot_id_value', satisfies_pzs=True)
        response = client.update_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.UpdateJobRequest()
    assert isinstance(response, jobs.Job)
    assert response.id == 'id_value'
    assert response.project_id == 'project_id_value'
    assert response.name == 'name_value'
    assert response.type_ == environment.JobType.JOB_TYPE_BATCH
    assert response.steps_location == 'steps_location_value'
    assert response.current_state == jobs.JobState.JOB_STATE_STOPPED
    assert response.requested_state == jobs.JobState.JOB_STATE_STOPPED
    assert response.replace_job_id == 'replace_job_id_value'
    assert response.client_request_id == 'client_request_id_value'
    assert response.replaced_by_job_id == 'replaced_by_job_id_value'
    assert response.temp_files == ['temp_files_value']
    assert response.location == 'location_value'
    assert response.created_from_snapshot_id == 'created_from_snapshot_id_value'
    assert response.satisfies_pzs is True

def test_update_job_empty_call():
    if False:
        i = 10
        return i + 15
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        client.update_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.UpdateJobRequest()

@pytest.mark.asyncio
async def test_update_job_async(transport: str='grpc_asyncio', request_type=jobs.UpdateJobRequest):
    client = JobsV1Beta3AsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.Job(id='id_value', project_id='project_id_value', name='name_value', type_=environment.JobType.JOB_TYPE_BATCH, steps_location='steps_location_value', current_state=jobs.JobState.JOB_STATE_STOPPED, requested_state=jobs.JobState.JOB_STATE_STOPPED, replace_job_id='replace_job_id_value', client_request_id='client_request_id_value', replaced_by_job_id='replaced_by_job_id_value', temp_files=['temp_files_value'], location='location_value', created_from_snapshot_id='created_from_snapshot_id_value', satisfies_pzs=True))
        response = await client.update_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.UpdateJobRequest()
    assert isinstance(response, jobs.Job)
    assert response.id == 'id_value'
    assert response.project_id == 'project_id_value'
    assert response.name == 'name_value'
    assert response.type_ == environment.JobType.JOB_TYPE_BATCH
    assert response.steps_location == 'steps_location_value'
    assert response.current_state == jobs.JobState.JOB_STATE_STOPPED
    assert response.requested_state == jobs.JobState.JOB_STATE_STOPPED
    assert response.replace_job_id == 'replace_job_id_value'
    assert response.client_request_id == 'client_request_id_value'
    assert response.replaced_by_job_id == 'replaced_by_job_id_value'
    assert response.temp_files == ['temp_files_value']
    assert response.location == 'location_value'
    assert response.created_from_snapshot_id == 'created_from_snapshot_id_value'
    assert response.satisfies_pzs is True

@pytest.mark.asyncio
async def test_update_job_async_from_dict():
    await test_update_job_async(request_type=dict)

def test_update_job_field_headers():
    if False:
        print('Hello World!')
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.UpdateJobRequest()
    request.project_id = 'project_id_value'
    request.location = 'location_value'
    request.job_id = 'job_id_value'
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = jobs.Job()
        client.update_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&location=location_value&job_id=job_id_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_job_field_headers_async():
    client = JobsV1Beta3AsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.UpdateJobRequest()
    request.project_id = 'project_id_value'
    request.location = 'location_value'
    request.job_id = 'job_id_value'
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.Job())
        await client.update_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&location=location_value&job_id=job_id_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [jobs.ListJobsRequest, dict])
def test_list_jobs(request_type, transport: str='grpc'):
    if False:
        return 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        client.list_jobs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.ListJobsRequest()

@pytest.mark.asyncio
async def test_list_jobs_async(transport: str='grpc_asyncio', request_type=jobs.ListJobsRequest):
    client = JobsV1Beta3AsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.ListJobsRequest()
    request.project_id = 'project_id_value'
    request.location = 'location_value'
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = jobs.ListJobsResponse()
        client.list_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&location=location_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_jobs_field_headers_async():
    client = JobsV1Beta3AsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.ListJobsRequest()
    request.project_id = 'project_id_value'
    request.location = 'location_value'
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.ListJobsResponse())
        await client.list_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&location=location_value') in kw['metadata']

def test_list_jobs_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.side_effect = (jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job(), jobs.Job()], next_page_token='abc'), jobs.ListJobsResponse(jobs=[], next_page_token='def'), jobs.ListJobsResponse(jobs=[jobs.Job()], next_page_token='ghi'), jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('project_id', ''), ('location', ''))),)
        pager = client.list_jobs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, jobs.Job) for i in results))

def test_list_jobs_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.side_effect = (jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job(), jobs.Job()], next_page_token='abc'), jobs.ListJobsResponse(jobs=[], next_page_token='def'), jobs.ListJobsResponse(jobs=[jobs.Job()], next_page_token='ghi'), jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job()]), RuntimeError)
        pages = list(client.list_jobs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_jobs_async_pager():
    client = JobsV1Beta3AsyncClient(credentials=ga_credentials.AnonymousCredentials)
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
    client = JobsV1Beta3AsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job(), jobs.Job()], next_page_token='abc'), jobs.ListJobsResponse(jobs=[], next_page_token='def'), jobs.ListJobsResponse(jobs=[jobs.Job()], next_page_token='ghi'), jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_jobs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [jobs.ListJobsRequest, dict])
def test_aggregated_list_jobs(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.aggregated_list_jobs), '__call__') as call:
        call.return_value = jobs.ListJobsResponse(next_page_token='next_page_token_value')
        response = client.aggregated_list_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.ListJobsRequest()
    assert isinstance(response, pagers.AggregatedListJobsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_aggregated_list_jobs_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.aggregated_list_jobs), '__call__') as call:
        client.aggregated_list_jobs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.ListJobsRequest()

@pytest.mark.asyncio
async def test_aggregated_list_jobs_async(transport: str='grpc_asyncio', request_type=jobs.ListJobsRequest):
    client = JobsV1Beta3AsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.aggregated_list_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.ListJobsResponse(next_page_token='next_page_token_value'))
        response = await client.aggregated_list_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.ListJobsRequest()
    assert isinstance(response, pagers.AggregatedListJobsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_aggregated_list_jobs_async_from_dict():
    await test_aggregated_list_jobs_async(request_type=dict)

def test_aggregated_list_jobs_field_headers():
    if False:
        return 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.ListJobsRequest()
    request.project_id = 'project_id_value'
    with mock.patch.object(type(client.transport.aggregated_list_jobs), '__call__') as call:
        call.return_value = jobs.ListJobsResponse()
        client.aggregated_list_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value') in kw['metadata']

@pytest.mark.asyncio
async def test_aggregated_list_jobs_field_headers_async():
    client = JobsV1Beta3AsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.ListJobsRequest()
    request.project_id = 'project_id_value'
    with mock.patch.object(type(client.transport.aggregated_list_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.ListJobsResponse())
        await client.aggregated_list_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value') in kw['metadata']

def test_aggregated_list_jobs_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.aggregated_list_jobs), '__call__') as call:
        call.side_effect = (jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job(), jobs.Job()], next_page_token='abc'), jobs.ListJobsResponse(jobs=[], next_page_token='def'), jobs.ListJobsResponse(jobs=[jobs.Job()], next_page_token='ghi'), jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('project_id', ''),)),)
        pager = client.aggregated_list_jobs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, jobs.Job) for i in results))

def test_aggregated_list_jobs_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.aggregated_list_jobs), '__call__') as call:
        call.side_effect = (jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job(), jobs.Job()], next_page_token='abc'), jobs.ListJobsResponse(jobs=[], next_page_token='def'), jobs.ListJobsResponse(jobs=[jobs.Job()], next_page_token='ghi'), jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job()]), RuntimeError)
        pages = list(client.aggregated_list_jobs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_aggregated_list_jobs_async_pager():
    client = JobsV1Beta3AsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.aggregated_list_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job(), jobs.Job()], next_page_token='abc'), jobs.ListJobsResponse(jobs=[], next_page_token='def'), jobs.ListJobsResponse(jobs=[jobs.Job()], next_page_token='ghi'), jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job()]), RuntimeError)
        async_pager = await client.aggregated_list_jobs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, jobs.Job) for i in responses))

@pytest.mark.asyncio
async def test_aggregated_list_jobs_async_pages():
    client = JobsV1Beta3AsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.aggregated_list_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job(), jobs.Job()], next_page_token='abc'), jobs.ListJobsResponse(jobs=[], next_page_token='def'), jobs.ListJobsResponse(jobs=[jobs.Job()], next_page_token='ghi'), jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job()]), RuntimeError)
        pages = []
        async for page_ in (await client.aggregated_list_jobs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [jobs.CheckActiveJobsRequest, dict])
def test_check_active_jobs(request_type, transport: str='grpc'):
    if False:
        return 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.check_active_jobs), '__call__') as call:
        call.return_value = jobs.CheckActiveJobsResponse(active_jobs_exist=True)
        response = client.check_active_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.CheckActiveJobsRequest()
    assert isinstance(response, jobs.CheckActiveJobsResponse)
    assert response.active_jobs_exist is True

def test_check_active_jobs_empty_call():
    if False:
        print('Hello World!')
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.check_active_jobs), '__call__') as call:
        client.check_active_jobs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.CheckActiveJobsRequest()

@pytest.mark.asyncio
async def test_check_active_jobs_async(transport: str='grpc_asyncio', request_type=jobs.CheckActiveJobsRequest):
    client = JobsV1Beta3AsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.check_active_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(jobs.CheckActiveJobsResponse(active_jobs_exist=True))
        response = await client.check_active_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.CheckActiveJobsRequest()
    assert isinstance(response, jobs.CheckActiveJobsResponse)
    assert response.active_jobs_exist is True

@pytest.mark.asyncio
async def test_check_active_jobs_async_from_dict():
    await test_check_active_jobs_async(request_type=dict)

@pytest.mark.parametrize('request_type', [jobs.SnapshotJobRequest, dict])
def test_snapshot_job(request_type, transport: str='grpc'):
    if False:
        return 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.snapshot_job), '__call__') as call:
        call.return_value = snapshots.Snapshot(id='id_value', project_id='project_id_value', source_job_id='source_job_id_value', state=snapshots.SnapshotState.PENDING, description='description_value', disk_size_bytes=1611, region='region_value')
        response = client.snapshot_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.SnapshotJobRequest()
    assert isinstance(response, snapshots.Snapshot)
    assert response.id == 'id_value'
    assert response.project_id == 'project_id_value'
    assert response.source_job_id == 'source_job_id_value'
    assert response.state == snapshots.SnapshotState.PENDING
    assert response.description == 'description_value'
    assert response.disk_size_bytes == 1611
    assert response.region == 'region_value'

def test_snapshot_job_empty_call():
    if False:
        print('Hello World!')
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.snapshot_job), '__call__') as call:
        client.snapshot_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.SnapshotJobRequest()

@pytest.mark.asyncio
async def test_snapshot_job_async(transport: str='grpc_asyncio', request_type=jobs.SnapshotJobRequest):
    client = JobsV1Beta3AsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.snapshot_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(snapshots.Snapshot(id='id_value', project_id='project_id_value', source_job_id='source_job_id_value', state=snapshots.SnapshotState.PENDING, description='description_value', disk_size_bytes=1611, region='region_value'))
        response = await client.snapshot_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == jobs.SnapshotJobRequest()
    assert isinstance(response, snapshots.Snapshot)
    assert response.id == 'id_value'
    assert response.project_id == 'project_id_value'
    assert response.source_job_id == 'source_job_id_value'
    assert response.state == snapshots.SnapshotState.PENDING
    assert response.description == 'description_value'
    assert response.disk_size_bytes == 1611
    assert response.region == 'region_value'

@pytest.mark.asyncio
async def test_snapshot_job_async_from_dict():
    await test_snapshot_job_async(request_type=dict)

def test_snapshot_job_field_headers():
    if False:
        return 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.SnapshotJobRequest()
    request.project_id = 'project_id_value'
    request.location = 'location_value'
    request.job_id = 'job_id_value'
    with mock.patch.object(type(client.transport.snapshot_job), '__call__') as call:
        call.return_value = snapshots.Snapshot()
        client.snapshot_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&location=location_value&job_id=job_id_value') in kw['metadata']

@pytest.mark.asyncio
async def test_snapshot_job_field_headers_async():
    client = JobsV1Beta3AsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = jobs.SnapshotJobRequest()
    request.project_id = 'project_id_value'
    request.location = 'location_value'
    request.job_id = 'job_id_value'
    with mock.patch.object(type(client.transport.snapshot_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(snapshots.Snapshot())
        await client.snapshot_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'project_id=project_id_value&location=location_value&job_id=job_id_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [jobs.CreateJobRequest, dict])
def test_create_job_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project_id': 'sample1', 'location': 'sample2'}
    request_init['job'] = {'id': 'id_value', 'project_id': 'project_id_value', 'name': 'name_value', 'type_': 1, 'environment': {'temp_storage_prefix': 'temp_storage_prefix_value', 'cluster_manager_api_service': 'cluster_manager_api_service_value', 'experiments': ['experiments_value1', 'experiments_value2'], 'service_options': ['service_options_value1', 'service_options_value2'], 'service_kms_key_name': 'service_kms_key_name_value', 'worker_pools': [{'kind': 'kind_value', 'num_workers': 1212, 'packages': [{'name': 'name_value', 'location': 'location_value'}], 'default_package_set': 1, 'machine_type': 'machine_type_value', 'teardown_policy': 1, 'disk_size_gb': 1261, 'disk_type': 'disk_type_value', 'disk_source_image': 'disk_source_image_value', 'zone': 'zone_value', 'taskrunner_settings': {'task_user': 'task_user_value', 'task_group': 'task_group_value', 'oauth_scopes': ['oauth_scopes_value1', 'oauth_scopes_value2'], 'base_url': 'base_url_value', 'dataflow_api_version': 'dataflow_api_version_value', 'parallel_worker_settings': {'base_url': 'base_url_value', 'reporting_enabled': True, 'service_path': 'service_path_value', 'shuffle_service_path': 'shuffle_service_path_value', 'worker_id': 'worker_id_value', 'temp_storage_prefix': 'temp_storage_prefix_value'}, 'base_task_dir': 'base_task_dir_value', 'continue_on_exception': True, 'log_to_serialconsole': True, 'alsologtostderr': True, 'log_upload_location': 'log_upload_location_value', 'log_dir': 'log_dir_value', 'temp_storage_prefix': 'temp_storage_prefix_value', 'harness_command': 'harness_command_value', 'workflow_file_name': 'workflow_file_name_value', 'commandlines_file_name': 'commandlines_file_name_value', 'vm_id': 'vm_id_value', 'language_hint': 'language_hint_value', 'streaming_worker_main_class': 'streaming_worker_main_class_value'}, 'on_host_maintenance': 'on_host_maintenance_value', 'data_disks': [{'size_gb': 739, 'disk_type': 'disk_type_value', 'mount_point': 'mount_point_value'}], 'metadata': {}, 'autoscaling_settings': {'algorithm': 1, 'max_num_workers': 1633}, 'pool_args': {'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}, 'network': 'network_value', 'subnetwork': 'subnetwork_value', 'worker_harness_container_image': 'worker_harness_container_image_value', 'num_threads_per_worker': 2361, 'ip_configuration': 1, 'sdk_harness_container_images': [{'container_image': 'container_image_value', 'use_single_core_per_container': True, 'environment_id': 'environment_id_value', 'capabilities': ['capabilities_value1', 'capabilities_value2']}]}], 'user_agent': {'fields': {}}, 'version': {}, 'dataset': 'dataset_value', 'sdk_pipeline_options': {}, 'internal_experiments': {}, 'service_account_email': 'service_account_email_value', 'flex_resource_scheduling_goal': 1, 'worker_region': 'worker_region_value', 'worker_zone': 'worker_zone_value', 'shuffle_mode': 1, 'debug_options': {'enable_hot_key_logging': True}}, 'steps': [{'kind': 'kind_value', 'name': 'name_value', 'properties': {}}], 'steps_location': 'steps_location_value', 'current_state': 1, 'current_state_time': {'seconds': 751, 'nanos': 543}, 'requested_state': 1, 'execution_info': {'stages': {}}, 'create_time': {}, 'replace_job_id': 'replace_job_id_value', 'transform_name_mapping': {}, 'client_request_id': 'client_request_id_value', 'replaced_by_job_id': 'replaced_by_job_id_value', 'temp_files': ['temp_files_value1', 'temp_files_value2'], 'labels': {}, 'location': 'location_value', 'pipeline_description': {'original_pipeline_transform': [{'kind': 1, 'id': 'id_value', 'name': 'name_value', 'display_data': [{'key': 'key_value', 'namespace': 'namespace_value', 'str_value': 'str_value_value', 'int64_value': 1073, 'float_value': 0.117, 'java_class_value': 'java_class_value_value', 'timestamp_value': {}, 'duration_value': {'seconds': 751, 'nanos': 543}, 'bool_value': True, 'short_str_value': 'short_str_value_value', 'url': 'url_value', 'label': 'label_value'}], 'output_collection_name': ['output_collection_name_value1', 'output_collection_name_value2'], 'input_collection_name': ['input_collection_name_value1', 'input_collection_name_value2']}], 'execution_pipeline_stage': [{'name': 'name_value', 'id': 'id_value', 'kind': 1, 'input_source': [{'user_name': 'user_name_value', 'name': 'name_value', 'original_transform_or_collection': 'original_transform_or_collection_value', 'size_bytes': 1089}], 'output_source': {}, 'prerequisite_stage': ['prerequisite_stage_value1', 'prerequisite_stage_value2'], 'component_transform': [{'user_name': 'user_name_value', 'name': 'name_value', 'original_transform': 'original_transform_value'}], 'component_source': [{'user_name': 'user_name_value', 'name': 'name_value', 'original_transform_or_collection': 'original_transform_or_collection_value'}]}], 'display_data': {}}, 'stage_states': [{'execution_stage_name': 'execution_stage_name_value', 'execution_stage_state': 1, 'current_state_time': {}}], 'job_metadata': {'sdk_version': {'version': 'version_value', 'version_display_name': 'version_display_name_value', 'sdk_support_status': 1}, 'spanner_details': [{'project_id': 'project_id_value', 'instance_id': 'instance_id_value', 'database_id': 'database_id_value'}], 'bigquery_details': [{'table': 'table_value', 'dataset': 'dataset_value', 'project_id': 'project_id_value', 'query': 'query_value'}], 'big_table_details': [{'project_id': 'project_id_value', 'instance_id': 'instance_id_value', 'table_id': 'table_id_value'}], 'pubsub_details': [{'topic': 'topic_value', 'subscription': 'subscription_value'}], 'file_details': [{'file_pattern': 'file_pattern_value'}], 'datastore_details': [{'namespace': 'namespace_value', 'project_id': 'project_id_value'}]}, 'start_time': {}, 'created_from_snapshot_id': 'created_from_snapshot_id_value', 'satisfies_pzs': True}
    test_field = jobs.CreateJobRequest.meta.fields['job']

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
        return_value = jobs.Job(id='id_value', project_id='project_id_value', name='name_value', type_=environment.JobType.JOB_TYPE_BATCH, steps_location='steps_location_value', current_state=jobs.JobState.JOB_STATE_STOPPED, requested_state=jobs.JobState.JOB_STATE_STOPPED, replace_job_id='replace_job_id_value', client_request_id='client_request_id_value', replaced_by_job_id='replaced_by_job_id_value', temp_files=['temp_files_value'], location='location_value', created_from_snapshot_id='created_from_snapshot_id_value', satisfies_pzs=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = jobs.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_job(request)
    assert isinstance(response, jobs.Job)
    assert response.id == 'id_value'
    assert response.project_id == 'project_id_value'
    assert response.name == 'name_value'
    assert response.type_ == environment.JobType.JOB_TYPE_BATCH
    assert response.steps_location == 'steps_location_value'
    assert response.current_state == jobs.JobState.JOB_STATE_STOPPED
    assert response.requested_state == jobs.JobState.JOB_STATE_STOPPED
    assert response.replace_job_id == 'replace_job_id_value'
    assert response.client_request_id == 'client_request_id_value'
    assert response.replaced_by_job_id == 'replaced_by_job_id_value'
    assert response.temp_files == ['temp_files_value']
    assert response.location == 'location_value'
    assert response.created_from_snapshot_id == 'created_from_snapshot_id_value'
    assert response.satisfies_pzs is True

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_job_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.JobsV1Beta3RestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobsV1Beta3RestInterceptor())
    client = JobsV1Beta3Client(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobsV1Beta3RestInterceptor, 'post_create_job') as post, mock.patch.object(transports.JobsV1Beta3RestInterceptor, 'pre_create_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = jobs.CreateJobRequest.pb(jobs.CreateJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = jobs.Job.to_json(jobs.Job())
        request = jobs.CreateJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = jobs.Job()
        client.create_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_job_rest_bad_request(transport: str='rest', request_type=jobs.CreateJobRequest):
    if False:
        while True:
            i = 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project_id': 'sample1', 'location': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_job(request)

def test_create_job_rest_error():
    if False:
        return 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [jobs.GetJobRequest, dict])
def test_get_job_rest(request_type):
    if False:
        print('Hello World!')
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project_id': 'sample1', 'location': 'sample2', 'job_id': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = jobs.Job(id='id_value', project_id='project_id_value', name='name_value', type_=environment.JobType.JOB_TYPE_BATCH, steps_location='steps_location_value', current_state=jobs.JobState.JOB_STATE_STOPPED, requested_state=jobs.JobState.JOB_STATE_STOPPED, replace_job_id='replace_job_id_value', client_request_id='client_request_id_value', replaced_by_job_id='replaced_by_job_id_value', temp_files=['temp_files_value'], location='location_value', created_from_snapshot_id='created_from_snapshot_id_value', satisfies_pzs=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = jobs.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_job(request)
    assert isinstance(response, jobs.Job)
    assert response.id == 'id_value'
    assert response.project_id == 'project_id_value'
    assert response.name == 'name_value'
    assert response.type_ == environment.JobType.JOB_TYPE_BATCH
    assert response.steps_location == 'steps_location_value'
    assert response.current_state == jobs.JobState.JOB_STATE_STOPPED
    assert response.requested_state == jobs.JobState.JOB_STATE_STOPPED
    assert response.replace_job_id == 'replace_job_id_value'
    assert response.client_request_id == 'client_request_id_value'
    assert response.replaced_by_job_id == 'replaced_by_job_id_value'
    assert response.temp_files == ['temp_files_value']
    assert response.location == 'location_value'
    assert response.created_from_snapshot_id == 'created_from_snapshot_id_value'
    assert response.satisfies_pzs is True

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_job_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.JobsV1Beta3RestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobsV1Beta3RestInterceptor())
    client = JobsV1Beta3Client(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobsV1Beta3RestInterceptor, 'post_get_job') as post, mock.patch.object(transports.JobsV1Beta3RestInterceptor, 'pre_get_job') as pre:
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
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project_id': 'sample1', 'location': 'sample2', 'job_id': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_job(request)

def test_get_job_rest_error():
    if False:
        i = 10
        return i + 15
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [jobs.UpdateJobRequest, dict])
def test_update_job_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project_id': 'sample1', 'location': 'sample2', 'job_id': 'sample3'}
    request_init['job'] = {'id': 'id_value', 'project_id': 'project_id_value', 'name': 'name_value', 'type_': 1, 'environment': {'temp_storage_prefix': 'temp_storage_prefix_value', 'cluster_manager_api_service': 'cluster_manager_api_service_value', 'experiments': ['experiments_value1', 'experiments_value2'], 'service_options': ['service_options_value1', 'service_options_value2'], 'service_kms_key_name': 'service_kms_key_name_value', 'worker_pools': [{'kind': 'kind_value', 'num_workers': 1212, 'packages': [{'name': 'name_value', 'location': 'location_value'}], 'default_package_set': 1, 'machine_type': 'machine_type_value', 'teardown_policy': 1, 'disk_size_gb': 1261, 'disk_type': 'disk_type_value', 'disk_source_image': 'disk_source_image_value', 'zone': 'zone_value', 'taskrunner_settings': {'task_user': 'task_user_value', 'task_group': 'task_group_value', 'oauth_scopes': ['oauth_scopes_value1', 'oauth_scopes_value2'], 'base_url': 'base_url_value', 'dataflow_api_version': 'dataflow_api_version_value', 'parallel_worker_settings': {'base_url': 'base_url_value', 'reporting_enabled': True, 'service_path': 'service_path_value', 'shuffle_service_path': 'shuffle_service_path_value', 'worker_id': 'worker_id_value', 'temp_storage_prefix': 'temp_storage_prefix_value'}, 'base_task_dir': 'base_task_dir_value', 'continue_on_exception': True, 'log_to_serialconsole': True, 'alsologtostderr': True, 'log_upload_location': 'log_upload_location_value', 'log_dir': 'log_dir_value', 'temp_storage_prefix': 'temp_storage_prefix_value', 'harness_command': 'harness_command_value', 'workflow_file_name': 'workflow_file_name_value', 'commandlines_file_name': 'commandlines_file_name_value', 'vm_id': 'vm_id_value', 'language_hint': 'language_hint_value', 'streaming_worker_main_class': 'streaming_worker_main_class_value'}, 'on_host_maintenance': 'on_host_maintenance_value', 'data_disks': [{'size_gb': 739, 'disk_type': 'disk_type_value', 'mount_point': 'mount_point_value'}], 'metadata': {}, 'autoscaling_settings': {'algorithm': 1, 'max_num_workers': 1633}, 'pool_args': {'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}, 'network': 'network_value', 'subnetwork': 'subnetwork_value', 'worker_harness_container_image': 'worker_harness_container_image_value', 'num_threads_per_worker': 2361, 'ip_configuration': 1, 'sdk_harness_container_images': [{'container_image': 'container_image_value', 'use_single_core_per_container': True, 'environment_id': 'environment_id_value', 'capabilities': ['capabilities_value1', 'capabilities_value2']}]}], 'user_agent': {'fields': {}}, 'version': {}, 'dataset': 'dataset_value', 'sdk_pipeline_options': {}, 'internal_experiments': {}, 'service_account_email': 'service_account_email_value', 'flex_resource_scheduling_goal': 1, 'worker_region': 'worker_region_value', 'worker_zone': 'worker_zone_value', 'shuffle_mode': 1, 'debug_options': {'enable_hot_key_logging': True}}, 'steps': [{'kind': 'kind_value', 'name': 'name_value', 'properties': {}}], 'steps_location': 'steps_location_value', 'current_state': 1, 'current_state_time': {'seconds': 751, 'nanos': 543}, 'requested_state': 1, 'execution_info': {'stages': {}}, 'create_time': {}, 'replace_job_id': 'replace_job_id_value', 'transform_name_mapping': {}, 'client_request_id': 'client_request_id_value', 'replaced_by_job_id': 'replaced_by_job_id_value', 'temp_files': ['temp_files_value1', 'temp_files_value2'], 'labels': {}, 'location': 'location_value', 'pipeline_description': {'original_pipeline_transform': [{'kind': 1, 'id': 'id_value', 'name': 'name_value', 'display_data': [{'key': 'key_value', 'namespace': 'namespace_value', 'str_value': 'str_value_value', 'int64_value': 1073, 'float_value': 0.117, 'java_class_value': 'java_class_value_value', 'timestamp_value': {}, 'duration_value': {'seconds': 751, 'nanos': 543}, 'bool_value': True, 'short_str_value': 'short_str_value_value', 'url': 'url_value', 'label': 'label_value'}], 'output_collection_name': ['output_collection_name_value1', 'output_collection_name_value2'], 'input_collection_name': ['input_collection_name_value1', 'input_collection_name_value2']}], 'execution_pipeline_stage': [{'name': 'name_value', 'id': 'id_value', 'kind': 1, 'input_source': [{'user_name': 'user_name_value', 'name': 'name_value', 'original_transform_or_collection': 'original_transform_or_collection_value', 'size_bytes': 1089}], 'output_source': {}, 'prerequisite_stage': ['prerequisite_stage_value1', 'prerequisite_stage_value2'], 'component_transform': [{'user_name': 'user_name_value', 'name': 'name_value', 'original_transform': 'original_transform_value'}], 'component_source': [{'user_name': 'user_name_value', 'name': 'name_value', 'original_transform_or_collection': 'original_transform_or_collection_value'}]}], 'display_data': {}}, 'stage_states': [{'execution_stage_name': 'execution_stage_name_value', 'execution_stage_state': 1, 'current_state_time': {}}], 'job_metadata': {'sdk_version': {'version': 'version_value', 'version_display_name': 'version_display_name_value', 'sdk_support_status': 1}, 'spanner_details': [{'project_id': 'project_id_value', 'instance_id': 'instance_id_value', 'database_id': 'database_id_value'}], 'bigquery_details': [{'table': 'table_value', 'dataset': 'dataset_value', 'project_id': 'project_id_value', 'query': 'query_value'}], 'big_table_details': [{'project_id': 'project_id_value', 'instance_id': 'instance_id_value', 'table_id': 'table_id_value'}], 'pubsub_details': [{'topic': 'topic_value', 'subscription': 'subscription_value'}], 'file_details': [{'file_pattern': 'file_pattern_value'}], 'datastore_details': [{'namespace': 'namespace_value', 'project_id': 'project_id_value'}]}, 'start_time': {}, 'created_from_snapshot_id': 'created_from_snapshot_id_value', 'satisfies_pzs': True}
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
        return_value = jobs.Job(id='id_value', project_id='project_id_value', name='name_value', type_=environment.JobType.JOB_TYPE_BATCH, steps_location='steps_location_value', current_state=jobs.JobState.JOB_STATE_STOPPED, requested_state=jobs.JobState.JOB_STATE_STOPPED, replace_job_id='replace_job_id_value', client_request_id='client_request_id_value', replaced_by_job_id='replaced_by_job_id_value', temp_files=['temp_files_value'], location='location_value', created_from_snapshot_id='created_from_snapshot_id_value', satisfies_pzs=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = jobs.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_job(request)
    assert isinstance(response, jobs.Job)
    assert response.id == 'id_value'
    assert response.project_id == 'project_id_value'
    assert response.name == 'name_value'
    assert response.type_ == environment.JobType.JOB_TYPE_BATCH
    assert response.steps_location == 'steps_location_value'
    assert response.current_state == jobs.JobState.JOB_STATE_STOPPED
    assert response.requested_state == jobs.JobState.JOB_STATE_STOPPED
    assert response.replace_job_id == 'replace_job_id_value'
    assert response.client_request_id == 'client_request_id_value'
    assert response.replaced_by_job_id == 'replaced_by_job_id_value'
    assert response.temp_files == ['temp_files_value']
    assert response.location == 'location_value'
    assert response.created_from_snapshot_id == 'created_from_snapshot_id_value'
    assert response.satisfies_pzs is True

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_job_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.JobsV1Beta3RestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobsV1Beta3RestInterceptor())
    client = JobsV1Beta3Client(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobsV1Beta3RestInterceptor, 'post_update_job') as post, mock.patch.object(transports.JobsV1Beta3RestInterceptor, 'pre_update_job') as pre:
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
        i = 10
        return i + 15
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project_id': 'sample1', 'location': 'sample2', 'job_id': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_job(request)

def test_update_job_rest_error():
    if False:
        i = 10
        return i + 15
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [jobs.ListJobsRequest, dict])
def test_list_jobs_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project_id': 'sample1', 'location': 'sample2'}
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

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_jobs_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.JobsV1Beta3RestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobsV1Beta3RestInterceptor())
    client = JobsV1Beta3Client(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobsV1Beta3RestInterceptor, 'post_list_jobs') as post, mock.patch.object(transports.JobsV1Beta3RestInterceptor, 'pre_list_jobs') as pre:
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
        for i in range(10):
            print('nop')
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project_id': 'sample1', 'location': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_jobs(request)

def test_list_jobs_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job(), jobs.Job()], next_page_token='abc'), jobs.ListJobsResponse(jobs=[], next_page_token='def'), jobs.ListJobsResponse(jobs=[jobs.Job()], next_page_token='ghi'), jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job()]))
        response = response + response
        response = tuple((jobs.ListJobsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'project_id': 'sample1', 'location': 'sample2'}
        pager = client.list_jobs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, jobs.Job) for i in results))
        pages = list(client.list_jobs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [jobs.ListJobsRequest, dict])
def test_aggregated_list_jobs_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project_id': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = jobs.ListJobsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = jobs.ListJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.aggregated_list_jobs(request)
    assert isinstance(response, pagers.AggregatedListJobsPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_aggregated_list_jobs_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.JobsV1Beta3RestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobsV1Beta3RestInterceptor())
    client = JobsV1Beta3Client(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobsV1Beta3RestInterceptor, 'post_aggregated_list_jobs') as post, mock.patch.object(transports.JobsV1Beta3RestInterceptor, 'pre_aggregated_list_jobs') as pre:
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
        client.aggregated_list_jobs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_aggregated_list_jobs_rest_bad_request(transport: str='rest', request_type=jobs.ListJobsRequest):
    if False:
        return 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project_id': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.aggregated_list_jobs(request)

def test_aggregated_list_jobs_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job(), jobs.Job()], next_page_token='abc'), jobs.ListJobsResponse(jobs=[], next_page_token='def'), jobs.ListJobsResponse(jobs=[jobs.Job()], next_page_token='ghi'), jobs.ListJobsResponse(jobs=[jobs.Job(), jobs.Job()]))
        response = response + response
        response = tuple((jobs.ListJobsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'project_id': 'sample1'}
        pager = client.aggregated_list_jobs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, jobs.Job) for i in results))
        pages = list(client.aggregated_list_jobs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_check_active_jobs_rest_no_http_options():
    if False:
        return 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = jobs.CheckActiveJobsRequest()
    with pytest.raises(RuntimeError):
        client.check_active_jobs(request)

@pytest.mark.parametrize('request_type', [jobs.SnapshotJobRequest, dict])
def test_snapshot_job_rest(request_type):
    if False:
        while True:
            i = 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project_id': 'sample1', 'location': 'sample2', 'job_id': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = snapshots.Snapshot(id='id_value', project_id='project_id_value', source_job_id='source_job_id_value', state=snapshots.SnapshotState.PENDING, description='description_value', disk_size_bytes=1611, region='region_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = snapshots.Snapshot.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.snapshot_job(request)
    assert isinstance(response, snapshots.Snapshot)
    assert response.id == 'id_value'
    assert response.project_id == 'project_id_value'
    assert response.source_job_id == 'source_job_id_value'
    assert response.state == snapshots.SnapshotState.PENDING
    assert response.description == 'description_value'
    assert response.disk_size_bytes == 1611
    assert response.region == 'region_value'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_snapshot_job_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.JobsV1Beta3RestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobsV1Beta3RestInterceptor())
    client = JobsV1Beta3Client(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobsV1Beta3RestInterceptor, 'post_snapshot_job') as post, mock.patch.object(transports.JobsV1Beta3RestInterceptor, 'pre_snapshot_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = jobs.SnapshotJobRequest.pb(jobs.SnapshotJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = snapshots.Snapshot.to_json(snapshots.Snapshot())
        request = jobs.SnapshotJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = snapshots.Snapshot()
        client.snapshot_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_snapshot_job_rest_bad_request(transport: str='rest', request_type=jobs.SnapshotJobRequest):
    if False:
        print('Hello World!')
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project_id': 'sample1', 'location': 'sample2', 'job_id': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.snapshot_job(request)

def test_snapshot_job_rest_error():
    if False:
        print('Hello World!')
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_check_active_jobs_rest_error():
    if False:
        while True:
            i = 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with pytest.raises(NotImplementedError) as not_implemented_error:
        client.check_active_jobs({})
    assert 'Method CheckActiveJobs is not available over REST transport' in str(not_implemented_error.value)

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.JobsV1Beta3GrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.JobsV1Beta3GrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = JobsV1Beta3Client(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.JobsV1Beta3GrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = JobsV1Beta3Client(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = JobsV1Beta3Client(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.JobsV1Beta3GrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = JobsV1Beta3Client(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.JobsV1Beta3GrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = JobsV1Beta3Client(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.JobsV1Beta3GrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.JobsV1Beta3GrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.JobsV1Beta3GrpcTransport, transports.JobsV1Beta3GrpcAsyncIOTransport, transports.JobsV1Beta3RestTransport])
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
        return 10
    transport = JobsV1Beta3Client.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.JobsV1Beta3GrpcTransport)

def test_jobs_v1_beta3_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.JobsV1Beta3Transport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_jobs_v1_beta3_base_transport():
    if False:
        return 10
    with mock.patch('google.cloud.dataflow_v1beta3.services.jobs_v1_beta3.transports.JobsV1Beta3Transport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.JobsV1Beta3Transport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_job', 'get_job', 'update_job', 'list_jobs', 'aggregated_list_jobs', 'check_active_jobs', 'snapshot_job')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_jobs_v1_beta3_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.dataflow_v1beta3.services.jobs_v1_beta3.transports.JobsV1Beta3Transport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.JobsV1Beta3Transport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/compute.readonly', 'https://www.googleapis.com/auth/userinfo.email'), quota_project_id='octopus')

def test_jobs_v1_beta3_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.dataflow_v1beta3.services.jobs_v1_beta3.transports.JobsV1Beta3Transport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.JobsV1Beta3Transport()
        adc.assert_called_once()

def test_jobs_v1_beta3_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        JobsV1Beta3Client()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/compute.readonly', 'https://www.googleapis.com/auth/userinfo.email'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.JobsV1Beta3GrpcTransport, transports.JobsV1Beta3GrpcAsyncIOTransport])
def test_jobs_v1_beta3_transport_auth_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/compute.readonly', 'https://www.googleapis.com/auth/userinfo.email'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.JobsV1Beta3GrpcTransport, transports.JobsV1Beta3GrpcAsyncIOTransport, transports.JobsV1Beta3RestTransport])
def test_jobs_v1_beta3_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.JobsV1Beta3GrpcTransport, grpc_helpers), (transports.JobsV1Beta3GrpcAsyncIOTransport, grpc_helpers_async)])
def test_jobs_v1_beta3_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('dataflow.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/compute.readonly', 'https://www.googleapis.com/auth/userinfo.email'), scopes=['1', '2'], default_host='dataflow.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.JobsV1Beta3GrpcTransport, transports.JobsV1Beta3GrpcAsyncIOTransport])
def test_jobs_v1_beta3_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_jobs_v1_beta3_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.JobsV1Beta3RestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_jobs_v1_beta3_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dataflow.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('dataflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dataflow.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_jobs_v1_beta3_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dataflow.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('dataflow.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dataflow.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_jobs_v1_beta3_client_transport_session_collision(transport_name):
    if False:
        while True:
            i = 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = JobsV1Beta3Client(credentials=creds1, transport=transport_name)
    client2 = JobsV1Beta3Client(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_job._session
    session2 = client2.transport.create_job._session
    assert session1 != session2
    session1 = client1.transport.get_job._session
    session2 = client2.transport.get_job._session
    assert session1 != session2
    session1 = client1.transport.update_job._session
    session2 = client2.transport.update_job._session
    assert session1 != session2
    session1 = client1.transport.list_jobs._session
    session2 = client2.transport.list_jobs._session
    assert session1 != session2
    session1 = client1.transport.aggregated_list_jobs._session
    session2 = client2.transport.aggregated_list_jobs._session
    assert session1 != session2
    session1 = client1.transport.check_active_jobs._session
    session2 = client2.transport.check_active_jobs._session
    assert session1 != session2
    session1 = client1.transport.snapshot_job._session
    session2 = client2.transport.snapshot_job._session
    assert session1 != session2

def test_jobs_v1_beta3_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.JobsV1Beta3GrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_jobs_v1_beta3_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.JobsV1Beta3GrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.JobsV1Beta3GrpcTransport, transports.JobsV1Beta3GrpcAsyncIOTransport])
def test_jobs_v1_beta3_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.JobsV1Beta3GrpcTransport, transports.JobsV1Beta3GrpcAsyncIOTransport])
def test_jobs_v1_beta3_transport_channel_mtls_with_adc(transport_class):
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

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = JobsV1Beta3Client.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'clam'}
    path = JobsV1Beta3Client.common_billing_account_path(**expected)
    actual = JobsV1Beta3Client.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = JobsV1Beta3Client.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'octopus'}
    path = JobsV1Beta3Client.common_folder_path(**expected)
    actual = JobsV1Beta3Client.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = JobsV1Beta3Client.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'nudibranch'}
    path = JobsV1Beta3Client.common_organization_path(**expected)
    actual = JobsV1Beta3Client.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = JobsV1Beta3Client.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'mussel'}
    path = JobsV1Beta3Client.common_project_path(**expected)
    actual = JobsV1Beta3Client.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = JobsV1Beta3Client.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = JobsV1Beta3Client.common_location_path(**expected)
    actual = JobsV1Beta3Client.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.JobsV1Beta3Transport, '_prep_wrapped_messages') as prep:
        client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.JobsV1Beta3Transport, '_prep_wrapped_messages') as prep:
        transport_class = JobsV1Beta3Client.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = JobsV1Beta3AsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        return 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = JobsV1Beta3Client(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(JobsV1Beta3Client, transports.JobsV1Beta3GrpcTransport), (JobsV1Beta3AsyncClient, transports.JobsV1Beta3GrpcAsyncIOTransport)])
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