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
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.protobuf import wrappers_pb2
from google.type import latlng_pb2
from google.type import money_pb2
from google.type import postal_address_pb2
from google.type import timeofday_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.talent_v4beta1.services.job_service import JobServiceAsyncClient, JobServiceClient, pagers, transports
from google.cloud.talent_v4beta1.types import common, filters, histogram
from google.cloud.talent_v4beta1.types import job
from google.cloud.talent_v4beta1.types import job as gct_job
from google.cloud.talent_v4beta1.types import job_service

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
        while True:
            i = 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert JobServiceClient._get_default_mtls_endpoint(None) is None
    assert JobServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert JobServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert JobServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert JobServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert JobServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(JobServiceClient, 'grpc'), (JobServiceAsyncClient, 'grpc_asyncio'), (JobServiceClient, 'rest')])
def test_job_service_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('jobs.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://jobs.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.JobServiceGrpcTransport, 'grpc'), (transports.JobServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.JobServiceRestTransport, 'rest')])
def test_job_service_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(JobServiceClient, 'grpc'), (JobServiceAsyncClient, 'grpc_asyncio'), (JobServiceClient, 'rest')])
def test_job_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('jobs.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://jobs.googleapis.com')

def test_job_service_client_get_transport_class():
    if False:
        return 10
    transport = JobServiceClient.get_transport_class()
    available_transports = [transports.JobServiceGrpcTransport, transports.JobServiceRestTransport]
    assert transport in available_transports
    transport = JobServiceClient.get_transport_class('grpc')
    assert transport == transports.JobServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(JobServiceClient, transports.JobServiceGrpcTransport, 'grpc'), (JobServiceAsyncClient, transports.JobServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (JobServiceClient, transports.JobServiceRestTransport, 'rest')])
@mock.patch.object(JobServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(JobServiceClient))
@mock.patch.object(JobServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(JobServiceAsyncClient))
def test_job_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(JobServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(JobServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(JobServiceClient, transports.JobServiceGrpcTransport, 'grpc', 'true'), (JobServiceAsyncClient, transports.JobServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (JobServiceClient, transports.JobServiceGrpcTransport, 'grpc', 'false'), (JobServiceAsyncClient, transports.JobServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (JobServiceClient, transports.JobServiceRestTransport, 'rest', 'true'), (JobServiceClient, transports.JobServiceRestTransport, 'rest', 'false')])
@mock.patch.object(JobServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(JobServiceClient))
@mock.patch.object(JobServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(JobServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_job_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [JobServiceClient, JobServiceAsyncClient])
@mock.patch.object(JobServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(JobServiceClient))
@mock.patch.object(JobServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(JobServiceAsyncClient))
def test_job_service_client_get_mtls_endpoint_and_cert_source(client_class):
    if False:
        for i in range(10):
            print('nop')
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(JobServiceClient, transports.JobServiceGrpcTransport, 'grpc'), (JobServiceAsyncClient, transports.JobServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (JobServiceClient, transports.JobServiceRestTransport, 'rest')])
def test_job_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(JobServiceClient, transports.JobServiceGrpcTransport, 'grpc', grpc_helpers), (JobServiceAsyncClient, transports.JobServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (JobServiceClient, transports.JobServiceRestTransport, 'rest', None)])
def test_job_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_job_service_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.talent_v4beta1.services.job_service.transports.JobServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = JobServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(JobServiceClient, transports.JobServiceGrpcTransport, 'grpc', grpc_helpers), (JobServiceAsyncClient, transports.JobServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_job_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('jobs.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/jobs'), scopes=None, default_host='jobs.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [job_service.CreateJobRequest, dict])
def test_create_job(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = gct_job.Job(name='name_value', company='company_value', requisition_id='requisition_id_value', title='title_value', description='description_value', addresses=['addresses_value'], job_benefits=[common.JobBenefit.CHILD_CARE], degree_types=[common.DegreeType.PRIMARY_EDUCATION], department='department_value', employment_types=[common.EmploymentType.FULL_TIME], incentives='incentives_value', language_code='language_code_value', job_level=common.JobLevel.ENTRY_LEVEL, promotion_value=1635, qualifications='qualifications_value', responsibilities='responsibilities_value', posting_region=common.PostingRegion.ADMINISTRATIVE_AREA, visibility=common.Visibility.ACCOUNT_ONLY, company_display_name='company_display_name_value')
        response = client.create_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.CreateJobRequest()
    assert isinstance(response, gct_job.Job)
    assert response.name == 'name_value'
    assert response.company == 'company_value'
    assert response.requisition_id == 'requisition_id_value'
    assert response.title == 'title_value'
    assert response.description == 'description_value'
    assert response.addresses == ['addresses_value']
    assert response.job_benefits == [common.JobBenefit.CHILD_CARE]
    assert response.degree_types == [common.DegreeType.PRIMARY_EDUCATION]
    assert response.department == 'department_value'
    assert response.employment_types == [common.EmploymentType.FULL_TIME]
    assert response.incentives == 'incentives_value'
    assert response.language_code == 'language_code_value'
    assert response.job_level == common.JobLevel.ENTRY_LEVEL
    assert response.promotion_value == 1635
    assert response.qualifications == 'qualifications_value'
    assert response.responsibilities == 'responsibilities_value'
    assert response.posting_region == common.PostingRegion.ADMINISTRATIVE_AREA
    assert response.visibility == common.Visibility.ACCOUNT_ONLY
    assert response.company_display_name == 'company_display_name_value'

def test_create_job_empty_call():
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        client.create_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.CreateJobRequest()

@pytest.mark.asyncio
async def test_create_job_async(transport: str='grpc_asyncio', request_type=job_service.CreateJobRequest):
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_job.Job(name='name_value', company='company_value', requisition_id='requisition_id_value', title='title_value', description='description_value', addresses=['addresses_value'], job_benefits=[common.JobBenefit.CHILD_CARE], degree_types=[common.DegreeType.PRIMARY_EDUCATION], department='department_value', employment_types=[common.EmploymentType.FULL_TIME], incentives='incentives_value', language_code='language_code_value', job_level=common.JobLevel.ENTRY_LEVEL, promotion_value=1635, qualifications='qualifications_value', responsibilities='responsibilities_value', posting_region=common.PostingRegion.ADMINISTRATIVE_AREA, visibility=common.Visibility.ACCOUNT_ONLY, company_display_name='company_display_name_value'))
        response = await client.create_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.CreateJobRequest()
    assert isinstance(response, gct_job.Job)
    assert response.name == 'name_value'
    assert response.company == 'company_value'
    assert response.requisition_id == 'requisition_id_value'
    assert response.title == 'title_value'
    assert response.description == 'description_value'
    assert response.addresses == ['addresses_value']
    assert response.job_benefits == [common.JobBenefit.CHILD_CARE]
    assert response.degree_types == [common.DegreeType.PRIMARY_EDUCATION]
    assert response.department == 'department_value'
    assert response.employment_types == [common.EmploymentType.FULL_TIME]
    assert response.incentives == 'incentives_value'
    assert response.language_code == 'language_code_value'
    assert response.job_level == common.JobLevel.ENTRY_LEVEL
    assert response.promotion_value == 1635
    assert response.qualifications == 'qualifications_value'
    assert response.responsibilities == 'responsibilities_value'
    assert response.posting_region == common.PostingRegion.ADMINISTRATIVE_AREA
    assert response.visibility == common.Visibility.ACCOUNT_ONLY
    assert response.company_display_name == 'company_display_name_value'

@pytest.mark.asyncio
async def test_create_job_async_from_dict():
    await test_create_job_async(request_type=dict)

def test_create_job_field_headers():
    if False:
        i = 10
        return i + 15
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.CreateJobRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = gct_job.Job()
        client.create_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_job_field_headers_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.CreateJobRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_job.Job())
        await client.create_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_job_flattened():
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = gct_job.Job()
        client.create_job(parent='parent_value', job=gct_job.Job(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].job
        mock_val = gct_job.Job(name='name_value')
        assert arg == mock_val

def test_create_job_flattened_error():
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_job(job_service.CreateJobRequest(), parent='parent_value', job=gct_job.Job(name='name_value'))

@pytest.mark.asyncio
async def test_create_job_flattened_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = gct_job.Job()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_job.Job())
        response = await client.create_job(parent='parent_value', job=gct_job.Job(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].job
        mock_val = gct_job.Job(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_job_flattened_error_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_job(job_service.CreateJobRequest(), parent='parent_value', job=gct_job.Job(name='name_value'))

@pytest.mark.parametrize('request_type', [job_service.BatchCreateJobsRequest, dict])
def test_batch_create_jobs(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_create_jobs), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.batch_create_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.BatchCreateJobsRequest()
    assert isinstance(response, future.Future)

def test_batch_create_jobs_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_create_jobs), '__call__') as call:
        client.batch_create_jobs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.BatchCreateJobsRequest()

@pytest.mark.asyncio
async def test_batch_create_jobs_async(transport: str='grpc_asyncio', request_type=job_service.BatchCreateJobsRequest):
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_create_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_create_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.BatchCreateJobsRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_batch_create_jobs_async_from_dict():
    await test_batch_create_jobs_async(request_type=dict)

def test_batch_create_jobs_field_headers():
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.BatchCreateJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_create_jobs), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_create_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_create_jobs_field_headers_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.BatchCreateJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_create_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.batch_create_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_batch_create_jobs_flattened():
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_create_jobs), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_create_jobs(parent='parent_value', jobs=[job.Job(name='name_value')])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].jobs
        mock_val = [job.Job(name='name_value')]
        assert arg == mock_val

def test_batch_create_jobs_flattened_error():
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.batch_create_jobs(job_service.BatchCreateJobsRequest(), parent='parent_value', jobs=[job.Job(name='name_value')])

@pytest.mark.asyncio
async def test_batch_create_jobs_flattened_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_create_jobs), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_create_jobs(parent='parent_value', jobs=[job.Job(name='name_value')])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].jobs
        mock_val = [job.Job(name='name_value')]
        assert arg == mock_val

@pytest.mark.asyncio
async def test_batch_create_jobs_flattened_error_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.batch_create_jobs(job_service.BatchCreateJobsRequest(), parent='parent_value', jobs=[job.Job(name='name_value')])

@pytest.mark.parametrize('request_type', [job_service.GetJobRequest, dict])
def test_get_job(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = job.Job(name='name_value', company='company_value', requisition_id='requisition_id_value', title='title_value', description='description_value', addresses=['addresses_value'], job_benefits=[common.JobBenefit.CHILD_CARE], degree_types=[common.DegreeType.PRIMARY_EDUCATION], department='department_value', employment_types=[common.EmploymentType.FULL_TIME], incentives='incentives_value', language_code='language_code_value', job_level=common.JobLevel.ENTRY_LEVEL, promotion_value=1635, qualifications='qualifications_value', responsibilities='responsibilities_value', posting_region=common.PostingRegion.ADMINISTRATIVE_AREA, visibility=common.Visibility.ACCOUNT_ONLY, company_display_name='company_display_name_value')
        response = client.get_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.GetJobRequest()
    assert isinstance(response, job.Job)
    assert response.name == 'name_value'
    assert response.company == 'company_value'
    assert response.requisition_id == 'requisition_id_value'
    assert response.title == 'title_value'
    assert response.description == 'description_value'
    assert response.addresses == ['addresses_value']
    assert response.job_benefits == [common.JobBenefit.CHILD_CARE]
    assert response.degree_types == [common.DegreeType.PRIMARY_EDUCATION]
    assert response.department == 'department_value'
    assert response.employment_types == [common.EmploymentType.FULL_TIME]
    assert response.incentives == 'incentives_value'
    assert response.language_code == 'language_code_value'
    assert response.job_level == common.JobLevel.ENTRY_LEVEL
    assert response.promotion_value == 1635
    assert response.qualifications == 'qualifications_value'
    assert response.responsibilities == 'responsibilities_value'
    assert response.posting_region == common.PostingRegion.ADMINISTRATIVE_AREA
    assert response.visibility == common.Visibility.ACCOUNT_ONLY
    assert response.company_display_name == 'company_display_name_value'

def test_get_job_empty_call():
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        client.get_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.GetJobRequest()

@pytest.mark.asyncio
async def test_get_job_async(transport: str='grpc_asyncio', request_type=job_service.GetJobRequest):
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job.Job(name='name_value', company='company_value', requisition_id='requisition_id_value', title='title_value', description='description_value', addresses=['addresses_value'], job_benefits=[common.JobBenefit.CHILD_CARE], degree_types=[common.DegreeType.PRIMARY_EDUCATION], department='department_value', employment_types=[common.EmploymentType.FULL_TIME], incentives='incentives_value', language_code='language_code_value', job_level=common.JobLevel.ENTRY_LEVEL, promotion_value=1635, qualifications='qualifications_value', responsibilities='responsibilities_value', posting_region=common.PostingRegion.ADMINISTRATIVE_AREA, visibility=common.Visibility.ACCOUNT_ONLY, company_display_name='company_display_name_value'))
        response = await client.get_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.GetJobRequest()
    assert isinstance(response, job.Job)
    assert response.name == 'name_value'
    assert response.company == 'company_value'
    assert response.requisition_id == 'requisition_id_value'
    assert response.title == 'title_value'
    assert response.description == 'description_value'
    assert response.addresses == ['addresses_value']
    assert response.job_benefits == [common.JobBenefit.CHILD_CARE]
    assert response.degree_types == [common.DegreeType.PRIMARY_EDUCATION]
    assert response.department == 'department_value'
    assert response.employment_types == [common.EmploymentType.FULL_TIME]
    assert response.incentives == 'incentives_value'
    assert response.language_code == 'language_code_value'
    assert response.job_level == common.JobLevel.ENTRY_LEVEL
    assert response.promotion_value == 1635
    assert response.qualifications == 'qualifications_value'
    assert response.responsibilities == 'responsibilities_value'
    assert response.posting_region == common.PostingRegion.ADMINISTRATIVE_AREA
    assert response.visibility == common.Visibility.ACCOUNT_ONLY
    assert response.company_display_name == 'company_display_name_value'

@pytest.mark.asyncio
async def test_get_job_async_from_dict():
    await test_get_job_async(request_type=dict)

def test_get_job_field_headers():
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.GetJobRequest()
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
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.GetJobRequest()
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
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_job(job_service.GetJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_job_flattened_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_job(job_service.GetJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [job_service.UpdateJobRequest, dict])
def test_update_job(request_type, transport: str='grpc'):
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = gct_job.Job(name='name_value', company='company_value', requisition_id='requisition_id_value', title='title_value', description='description_value', addresses=['addresses_value'], job_benefits=[common.JobBenefit.CHILD_CARE], degree_types=[common.DegreeType.PRIMARY_EDUCATION], department='department_value', employment_types=[common.EmploymentType.FULL_TIME], incentives='incentives_value', language_code='language_code_value', job_level=common.JobLevel.ENTRY_LEVEL, promotion_value=1635, qualifications='qualifications_value', responsibilities='responsibilities_value', posting_region=common.PostingRegion.ADMINISTRATIVE_AREA, visibility=common.Visibility.ACCOUNT_ONLY, company_display_name='company_display_name_value')
        response = client.update_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.UpdateJobRequest()
    assert isinstance(response, gct_job.Job)
    assert response.name == 'name_value'
    assert response.company == 'company_value'
    assert response.requisition_id == 'requisition_id_value'
    assert response.title == 'title_value'
    assert response.description == 'description_value'
    assert response.addresses == ['addresses_value']
    assert response.job_benefits == [common.JobBenefit.CHILD_CARE]
    assert response.degree_types == [common.DegreeType.PRIMARY_EDUCATION]
    assert response.department == 'department_value'
    assert response.employment_types == [common.EmploymentType.FULL_TIME]
    assert response.incentives == 'incentives_value'
    assert response.language_code == 'language_code_value'
    assert response.job_level == common.JobLevel.ENTRY_LEVEL
    assert response.promotion_value == 1635
    assert response.qualifications == 'qualifications_value'
    assert response.responsibilities == 'responsibilities_value'
    assert response.posting_region == common.PostingRegion.ADMINISTRATIVE_AREA
    assert response.visibility == common.Visibility.ACCOUNT_ONLY
    assert response.company_display_name == 'company_display_name_value'

def test_update_job_empty_call():
    if False:
        i = 10
        return i + 15
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        client.update_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.UpdateJobRequest()

@pytest.mark.asyncio
async def test_update_job_async(transport: str='grpc_asyncio', request_type=job_service.UpdateJobRequest):
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_job.Job(name='name_value', company='company_value', requisition_id='requisition_id_value', title='title_value', description='description_value', addresses=['addresses_value'], job_benefits=[common.JobBenefit.CHILD_CARE], degree_types=[common.DegreeType.PRIMARY_EDUCATION], department='department_value', employment_types=[common.EmploymentType.FULL_TIME], incentives='incentives_value', language_code='language_code_value', job_level=common.JobLevel.ENTRY_LEVEL, promotion_value=1635, qualifications='qualifications_value', responsibilities='responsibilities_value', posting_region=common.PostingRegion.ADMINISTRATIVE_AREA, visibility=common.Visibility.ACCOUNT_ONLY, company_display_name='company_display_name_value'))
        response = await client.update_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.UpdateJobRequest()
    assert isinstance(response, gct_job.Job)
    assert response.name == 'name_value'
    assert response.company == 'company_value'
    assert response.requisition_id == 'requisition_id_value'
    assert response.title == 'title_value'
    assert response.description == 'description_value'
    assert response.addresses == ['addresses_value']
    assert response.job_benefits == [common.JobBenefit.CHILD_CARE]
    assert response.degree_types == [common.DegreeType.PRIMARY_EDUCATION]
    assert response.department == 'department_value'
    assert response.employment_types == [common.EmploymentType.FULL_TIME]
    assert response.incentives == 'incentives_value'
    assert response.language_code == 'language_code_value'
    assert response.job_level == common.JobLevel.ENTRY_LEVEL
    assert response.promotion_value == 1635
    assert response.qualifications == 'qualifications_value'
    assert response.responsibilities == 'responsibilities_value'
    assert response.posting_region == common.PostingRegion.ADMINISTRATIVE_AREA
    assert response.visibility == common.Visibility.ACCOUNT_ONLY
    assert response.company_display_name == 'company_display_name_value'

@pytest.mark.asyncio
async def test_update_job_async_from_dict():
    await test_update_job_async(request_type=dict)

def test_update_job_field_headers():
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.UpdateJobRequest()
    request.job.name = 'name_value'
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = gct_job.Job()
        client.update_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'job.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_job_field_headers_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.UpdateJobRequest()
    request.job.name = 'name_value'
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_job.Job())
        await client.update_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'job.name=name_value') in kw['metadata']

def test_update_job_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = gct_job.Job()
        client.update_job(job=gct_job.Job(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].job
        mock_val = gct_job.Job(name='name_value')
        assert arg == mock_val

def test_update_job_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_job(job_service.UpdateJobRequest(), job=gct_job.Job(name='name_value'))

@pytest.mark.asyncio
async def test_update_job_flattened_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_job), '__call__') as call:
        call.return_value = gct_job.Job()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_job.Job())
        response = await client.update_job(job=gct_job.Job(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].job
        mock_val = gct_job.Job(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_job_flattened_error_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_job(job_service.UpdateJobRequest(), job=gct_job.Job(name='name_value'))

@pytest.mark.parametrize('request_type', [job_service.BatchUpdateJobsRequest, dict])
def test_batch_update_jobs(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_update_jobs), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.batch_update_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.BatchUpdateJobsRequest()
    assert isinstance(response, future.Future)

def test_batch_update_jobs_empty_call():
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_update_jobs), '__call__') as call:
        client.batch_update_jobs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.BatchUpdateJobsRequest()

@pytest.mark.asyncio
async def test_batch_update_jobs_async(transport: str='grpc_asyncio', request_type=job_service.BatchUpdateJobsRequest):
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_update_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_update_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.BatchUpdateJobsRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_batch_update_jobs_async_from_dict():
    await test_batch_update_jobs_async(request_type=dict)

def test_batch_update_jobs_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.BatchUpdateJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_update_jobs), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_update_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_update_jobs_field_headers_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.BatchUpdateJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_update_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.batch_update_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_batch_update_jobs_flattened():
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_update_jobs), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_update_jobs(parent='parent_value', jobs=[job.Job(name='name_value')])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].jobs
        mock_val = [job.Job(name='name_value')]
        assert arg == mock_val

def test_batch_update_jobs_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.batch_update_jobs(job_service.BatchUpdateJobsRequest(), parent='parent_value', jobs=[job.Job(name='name_value')])

@pytest.mark.asyncio
async def test_batch_update_jobs_flattened_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_update_jobs), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_update_jobs(parent='parent_value', jobs=[job.Job(name='name_value')])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].jobs
        mock_val = [job.Job(name='name_value')]
        assert arg == mock_val

@pytest.mark.asyncio
async def test_batch_update_jobs_flattened_error_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.batch_update_jobs(job_service.BatchUpdateJobsRequest(), parent='parent_value', jobs=[job.Job(name='name_value')])

@pytest.mark.parametrize('request_type', [job_service.DeleteJobRequest, dict])
def test_delete_job(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        call.return_value = None
        response = client.delete_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.DeleteJobRequest()
    assert response is None

def test_delete_job_empty_call():
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        client.delete_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.DeleteJobRequest()

@pytest.mark.asyncio
async def test_delete_job_async(transport: str='grpc_asyncio', request_type=job_service.DeleteJobRequest):
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.DeleteJobRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_job_async_from_dict():
    await test_delete_job_async(request_type=dict)

def test_delete_job_field_headers():
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.DeleteJobRequest()
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
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.DeleteJobRequest()
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
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_job(job_service.DeleteJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_job_flattened_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_job(job_service.DeleteJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [job_service.BatchDeleteJobsRequest, dict])
def test_batch_delete_jobs(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_delete_jobs), '__call__') as call:
        call.return_value = None
        response = client.batch_delete_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.BatchDeleteJobsRequest()
    assert response is None

def test_batch_delete_jobs_empty_call():
    if False:
        i = 10
        return i + 15
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_delete_jobs), '__call__') as call:
        client.batch_delete_jobs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.BatchDeleteJobsRequest()

@pytest.mark.asyncio
async def test_batch_delete_jobs_async(transport: str='grpc_asyncio', request_type=job_service.BatchDeleteJobsRequest):
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_delete_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.batch_delete_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.BatchDeleteJobsRequest()
    assert response is None

@pytest.mark.asyncio
async def test_batch_delete_jobs_async_from_dict():
    await test_batch_delete_jobs_async(request_type=dict)

def test_batch_delete_jobs_field_headers():
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.BatchDeleteJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_delete_jobs), '__call__') as call:
        call.return_value = None
        client.batch_delete_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_delete_jobs_field_headers_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.BatchDeleteJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_delete_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.batch_delete_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_batch_delete_jobs_flattened():
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_delete_jobs), '__call__') as call:
        call.return_value = None
        client.batch_delete_jobs(parent='parent_value', filter='filter_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

def test_batch_delete_jobs_flattened_error():
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.batch_delete_jobs(job_service.BatchDeleteJobsRequest(), parent='parent_value', filter='filter_value')

@pytest.mark.asyncio
async def test_batch_delete_jobs_flattened_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_delete_jobs), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.batch_delete_jobs(parent='parent_value', filter='filter_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_batch_delete_jobs_flattened_error_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.batch_delete_jobs(job_service.BatchDeleteJobsRequest(), parent='parent_value', filter='filter_value')

@pytest.mark.parametrize('request_type', [job_service.ListJobsRequest, dict])
def test_list_jobs(request_type, transport: str='grpc'):
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = job_service.ListJobsResponse(next_page_token='next_page_token_value')
        response = client.list_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.ListJobsRequest()
    assert isinstance(response, pagers.ListJobsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_jobs_empty_call():
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        client.list_jobs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.ListJobsRequest()

@pytest.mark.asyncio
async def test_list_jobs_async(transport: str='grpc_asyncio', request_type=job_service.ListJobsRequest):
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job_service.ListJobsResponse(next_page_token='next_page_token_value'))
        response = await client.list_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.ListJobsRequest()
    assert isinstance(response, pagers.ListJobsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_jobs_async_from_dict():
    await test_list_jobs_async(request_type=dict)

def test_list_jobs_field_headers():
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.ListJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = job_service.ListJobsResponse()
        client.list_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_jobs_field_headers_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.ListJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job_service.ListJobsResponse())
        await client.list_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_jobs_flattened():
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = job_service.ListJobsResponse()
        client.list_jobs(parent='parent_value', filter='filter_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

def test_list_jobs_flattened_error():
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_jobs(job_service.ListJobsRequest(), parent='parent_value', filter='filter_value')

@pytest.mark.asyncio
async def test_list_jobs_flattened_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = job_service.ListJobsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job_service.ListJobsResponse())
        response = await client.list_jobs(parent='parent_value', filter='filter_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_jobs_flattened_error_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_jobs(job_service.ListJobsRequest(), parent='parent_value', filter='filter_value')

def test_list_jobs_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.side_effect = (job_service.ListJobsResponse(jobs=[job.Job(), job.Job(), job.Job()], next_page_token='abc'), job_service.ListJobsResponse(jobs=[], next_page_token='def'), job_service.ListJobsResponse(jobs=[job.Job()], next_page_token='ghi'), job_service.ListJobsResponse(jobs=[job.Job(), job.Job()]), RuntimeError)
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
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.side_effect = (job_service.ListJobsResponse(jobs=[job.Job(), job.Job(), job.Job()], next_page_token='abc'), job_service.ListJobsResponse(jobs=[], next_page_token='def'), job_service.ListJobsResponse(jobs=[job.Job()], next_page_token='ghi'), job_service.ListJobsResponse(jobs=[job.Job(), job.Job()]), RuntimeError)
        pages = list(client.list_jobs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_jobs_async_pager():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (job_service.ListJobsResponse(jobs=[job.Job(), job.Job(), job.Job()], next_page_token='abc'), job_service.ListJobsResponse(jobs=[], next_page_token='def'), job_service.ListJobsResponse(jobs=[job.Job()], next_page_token='ghi'), job_service.ListJobsResponse(jobs=[job.Job(), job.Job()]), RuntimeError)
        async_pager = await client.list_jobs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, job.Job) for i in responses))

@pytest.mark.asyncio
async def test_list_jobs_async_pages():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (job_service.ListJobsResponse(jobs=[job.Job(), job.Job(), job.Job()], next_page_token='abc'), job_service.ListJobsResponse(jobs=[], next_page_token='def'), job_service.ListJobsResponse(jobs=[job.Job()], next_page_token='ghi'), job_service.ListJobsResponse(jobs=[job.Job(), job.Job()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_jobs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [job_service.SearchJobsRequest, dict])
def test_search_jobs(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_jobs), '__call__') as call:
        call.return_value = job_service.SearchJobsResponse(next_page_token='next_page_token_value', estimated_total_size=2141, total_size=1086, broadened_query_jobs_count=2766)
        response = client.search_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.SearchJobsRequest()
    assert isinstance(response, pagers.SearchJobsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.estimated_total_size == 2141
    assert response.total_size == 1086
    assert response.broadened_query_jobs_count == 2766

def test_search_jobs_empty_call():
    if False:
        i = 10
        return i + 15
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search_jobs), '__call__') as call:
        client.search_jobs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.SearchJobsRequest()

@pytest.mark.asyncio
async def test_search_jobs_async(transport: str='grpc_asyncio', request_type=job_service.SearchJobsRequest):
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job_service.SearchJobsResponse(next_page_token='next_page_token_value', estimated_total_size=2141, total_size=1086, broadened_query_jobs_count=2766))
        response = await client.search_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.SearchJobsRequest()
    assert isinstance(response, pagers.SearchJobsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.estimated_total_size == 2141
    assert response.total_size == 1086
    assert response.broadened_query_jobs_count == 2766

@pytest.mark.asyncio
async def test_search_jobs_async_from_dict():
    await test_search_jobs_async(request_type=dict)

def test_search_jobs_field_headers():
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.SearchJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.search_jobs), '__call__') as call:
        call.return_value = job_service.SearchJobsResponse()
        client.search_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_search_jobs_field_headers_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.SearchJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.search_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job_service.SearchJobsResponse())
        await client.search_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_search_jobs_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_jobs), '__call__') as call:
        call.side_effect = (job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()], next_page_token='abc'), job_service.SearchJobsResponse(matching_jobs=[], next_page_token='def'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob()], next_page_token='ghi'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.search_jobs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, job_service.SearchJobsResponse.MatchingJob) for i in results))

def test_search_jobs_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_jobs), '__call__') as call:
        call.side_effect = (job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()], next_page_token='abc'), job_service.SearchJobsResponse(matching_jobs=[], next_page_token='def'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob()], next_page_token='ghi'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()]), RuntimeError)
        pages = list(client.search_jobs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_search_jobs_async_pager():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()], next_page_token='abc'), job_service.SearchJobsResponse(matching_jobs=[], next_page_token='def'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob()], next_page_token='ghi'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()]), RuntimeError)
        async_pager = await client.search_jobs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, job_service.SearchJobsResponse.MatchingJob) for i in responses))

@pytest.mark.asyncio
async def test_search_jobs_async_pages():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()], next_page_token='abc'), job_service.SearchJobsResponse(matching_jobs=[], next_page_token='def'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob()], next_page_token='ghi'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()]), RuntimeError)
        pages = []
        async for page_ in (await client.search_jobs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [job_service.SearchJobsRequest, dict])
def test_search_jobs_for_alert(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_jobs_for_alert), '__call__') as call:
        call.return_value = job_service.SearchJobsResponse(next_page_token='next_page_token_value', estimated_total_size=2141, total_size=1086, broadened_query_jobs_count=2766)
        response = client.search_jobs_for_alert(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.SearchJobsRequest()
    assert isinstance(response, pagers.SearchJobsForAlertPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.estimated_total_size == 2141
    assert response.total_size == 1086
    assert response.broadened_query_jobs_count == 2766

def test_search_jobs_for_alert_empty_call():
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search_jobs_for_alert), '__call__') as call:
        client.search_jobs_for_alert()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.SearchJobsRequest()

@pytest.mark.asyncio
async def test_search_jobs_for_alert_async(transport: str='grpc_asyncio', request_type=job_service.SearchJobsRequest):
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_jobs_for_alert), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job_service.SearchJobsResponse(next_page_token='next_page_token_value', estimated_total_size=2141, total_size=1086, broadened_query_jobs_count=2766))
        response = await client.search_jobs_for_alert(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == job_service.SearchJobsRequest()
    assert isinstance(response, pagers.SearchJobsForAlertAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.estimated_total_size == 2141
    assert response.total_size == 1086
    assert response.broadened_query_jobs_count == 2766

@pytest.mark.asyncio
async def test_search_jobs_for_alert_async_from_dict():
    await test_search_jobs_for_alert_async(request_type=dict)

def test_search_jobs_for_alert_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.SearchJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.search_jobs_for_alert), '__call__') as call:
        call.return_value = job_service.SearchJobsResponse()
        client.search_jobs_for_alert(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_search_jobs_for_alert_field_headers_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = job_service.SearchJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.search_jobs_for_alert), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(job_service.SearchJobsResponse())
        await client.search_jobs_for_alert(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_search_jobs_for_alert_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_jobs_for_alert), '__call__') as call:
        call.side_effect = (job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()], next_page_token='abc'), job_service.SearchJobsResponse(matching_jobs=[], next_page_token='def'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob()], next_page_token='ghi'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.search_jobs_for_alert(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, job_service.SearchJobsResponse.MatchingJob) for i in results))

def test_search_jobs_for_alert_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_jobs_for_alert), '__call__') as call:
        call.side_effect = (job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()], next_page_token='abc'), job_service.SearchJobsResponse(matching_jobs=[], next_page_token='def'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob()], next_page_token='ghi'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()]), RuntimeError)
        pages = list(client.search_jobs_for_alert(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_search_jobs_for_alert_async_pager():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_jobs_for_alert), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()], next_page_token='abc'), job_service.SearchJobsResponse(matching_jobs=[], next_page_token='def'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob()], next_page_token='ghi'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()]), RuntimeError)
        async_pager = await client.search_jobs_for_alert(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, job_service.SearchJobsResponse.MatchingJob) for i in responses))

@pytest.mark.asyncio
async def test_search_jobs_for_alert_async_pages():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_jobs_for_alert), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()], next_page_token='abc'), job_service.SearchJobsResponse(matching_jobs=[], next_page_token='def'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob()], next_page_token='ghi'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()]), RuntimeError)
        pages = []
        async for page_ in (await client.search_jobs_for_alert(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [job_service.CreateJobRequest, dict])
def test_create_job_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/tenants/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gct_job.Job(name='name_value', company='company_value', requisition_id='requisition_id_value', title='title_value', description='description_value', addresses=['addresses_value'], job_benefits=[common.JobBenefit.CHILD_CARE], degree_types=[common.DegreeType.PRIMARY_EDUCATION], department='department_value', employment_types=[common.EmploymentType.FULL_TIME], incentives='incentives_value', language_code='language_code_value', job_level=common.JobLevel.ENTRY_LEVEL, promotion_value=1635, qualifications='qualifications_value', responsibilities='responsibilities_value', posting_region=common.PostingRegion.ADMINISTRATIVE_AREA, visibility=common.Visibility.ACCOUNT_ONLY, company_display_name='company_display_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gct_job.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_job(request)
    assert isinstance(response, gct_job.Job)
    assert response.name == 'name_value'
    assert response.company == 'company_value'
    assert response.requisition_id == 'requisition_id_value'
    assert response.title == 'title_value'
    assert response.description == 'description_value'
    assert response.addresses == ['addresses_value']
    assert response.job_benefits == [common.JobBenefit.CHILD_CARE]
    assert response.degree_types == [common.DegreeType.PRIMARY_EDUCATION]
    assert response.department == 'department_value'
    assert response.employment_types == [common.EmploymentType.FULL_TIME]
    assert response.incentives == 'incentives_value'
    assert response.language_code == 'language_code_value'
    assert response.job_level == common.JobLevel.ENTRY_LEVEL
    assert response.promotion_value == 1635
    assert response.qualifications == 'qualifications_value'
    assert response.responsibilities == 'responsibilities_value'
    assert response.posting_region == common.PostingRegion.ADMINISTRATIVE_AREA
    assert response.visibility == common.Visibility.ACCOUNT_ONLY
    assert response.company_display_name == 'company_display_name_value'

def test_create_job_rest_required_fields(request_type=job_service.CreateJobRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.JobServiceRestTransport
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
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gct_job.Job()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gct_job.Job.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_job_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'job'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_job_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobServiceRestInterceptor())
    client = JobServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobServiceRestInterceptor, 'post_create_job') as post, mock.patch.object(transports.JobServiceRestInterceptor, 'pre_create_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = job_service.CreateJobRequest.pb(job_service.CreateJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gct_job.Job.to_json(gct_job.Job())
        request = job_service.CreateJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gct_job.Job()
        client.create_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_job_rest_bad_request(transport: str='rest', request_type=job_service.CreateJobRequest):
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/tenants/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_job(request)

def test_create_job_rest_flattened():
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gct_job.Job()
        sample_request = {'parent': 'projects/sample1/tenants/sample2'}
        mock_args = dict(parent='parent_value', job=gct_job.Job(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gct_job.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v4beta1/{parent=projects/*/tenants/*}/jobs' % client.transport._host, args[1])

def test_create_job_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_job(job_service.CreateJobRequest(), parent='parent_value', job=gct_job.Job(name='name_value'))

def test_create_job_rest_error():
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [job_service.BatchCreateJobsRequest, dict])
def test_batch_create_jobs_rest(request_type):
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/tenants/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_create_jobs(request)
    assert response.operation.name == 'operations/spam'

def test_batch_create_jobs_rest_required_fields(request_type=job_service.BatchCreateJobsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.JobServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_create_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_create_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.batch_create_jobs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_create_jobs_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_create_jobs._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'jobs'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_create_jobs_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobServiceRestInterceptor())
    client = JobServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.JobServiceRestInterceptor, 'post_batch_create_jobs') as post, mock.patch.object(transports.JobServiceRestInterceptor, 'pre_batch_create_jobs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = job_service.BatchCreateJobsRequest.pb(job_service.BatchCreateJobsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = job_service.BatchCreateJobsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.batch_create_jobs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_create_jobs_rest_bad_request(transport: str='rest', request_type=job_service.BatchCreateJobsRequest):
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/tenants/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_create_jobs(request)

def test_batch_create_jobs_rest_flattened():
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/tenants/sample2'}
        mock_args = dict(parent='parent_value', jobs=[job.Job(name='name_value')])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.batch_create_jobs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v4beta1/{parent=projects/*/tenants/*}/jobs:batchCreate' % client.transport._host, args[1])

def test_batch_create_jobs_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.batch_create_jobs(job_service.BatchCreateJobsRequest(), parent='parent_value', jobs=[job.Job(name='name_value')])

def test_batch_create_jobs_rest_error():
    if False:
        i = 10
        return i + 15
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [job_service.GetJobRequest, dict])
def test_get_job_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/tenants/sample2/jobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = job.Job(name='name_value', company='company_value', requisition_id='requisition_id_value', title='title_value', description='description_value', addresses=['addresses_value'], job_benefits=[common.JobBenefit.CHILD_CARE], degree_types=[common.DegreeType.PRIMARY_EDUCATION], department='department_value', employment_types=[common.EmploymentType.FULL_TIME], incentives='incentives_value', language_code='language_code_value', job_level=common.JobLevel.ENTRY_LEVEL, promotion_value=1635, qualifications='qualifications_value', responsibilities='responsibilities_value', posting_region=common.PostingRegion.ADMINISTRATIVE_AREA, visibility=common.Visibility.ACCOUNT_ONLY, company_display_name='company_display_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = job.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_job(request)
    assert isinstance(response, job.Job)
    assert response.name == 'name_value'
    assert response.company == 'company_value'
    assert response.requisition_id == 'requisition_id_value'
    assert response.title == 'title_value'
    assert response.description == 'description_value'
    assert response.addresses == ['addresses_value']
    assert response.job_benefits == [common.JobBenefit.CHILD_CARE]
    assert response.degree_types == [common.DegreeType.PRIMARY_EDUCATION]
    assert response.department == 'department_value'
    assert response.employment_types == [common.EmploymentType.FULL_TIME]
    assert response.incentives == 'incentives_value'
    assert response.language_code == 'language_code_value'
    assert response.job_level == common.JobLevel.ENTRY_LEVEL
    assert response.promotion_value == 1635
    assert response.qualifications == 'qualifications_value'
    assert response.responsibilities == 'responsibilities_value'
    assert response.posting_region == common.PostingRegion.ADMINISTRATIVE_AREA
    assert response.visibility == common.Visibility.ACCOUNT_ONLY
    assert response.company_display_name == 'company_display_name_value'

def test_get_job_rest_required_fields(request_type=job_service.GetJobRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.JobServiceRestTransport
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
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_job_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobServiceRestInterceptor())
    client = JobServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobServiceRestInterceptor, 'post_get_job') as post, mock.patch.object(transports.JobServiceRestInterceptor, 'pre_get_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = job_service.GetJobRequest.pb(job_service.GetJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = job.Job.to_json(job.Job())
        request = job_service.GetJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = job.Job()
        client.get_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_job_rest_bad_request(transport: str='rest', request_type=job_service.GetJobRequest):
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/tenants/sample2/jobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_job(request)

def test_get_job_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = job.Job()
        sample_request = {'name': 'projects/sample1/tenants/sample2/jobs/sample3'}
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
        assert path_template.validate('%s/v4beta1/{name=projects/*/tenants/*/jobs/*}' % client.transport._host, args[1])

def test_get_job_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_job(job_service.GetJobRequest(), name='name_value')

def test_get_job_rest_error():
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [job_service.UpdateJobRequest, dict])
def test_update_job_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'job': {'name': 'projects/sample1/tenants/sample2/jobs/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gct_job.Job(name='name_value', company='company_value', requisition_id='requisition_id_value', title='title_value', description='description_value', addresses=['addresses_value'], job_benefits=[common.JobBenefit.CHILD_CARE], degree_types=[common.DegreeType.PRIMARY_EDUCATION], department='department_value', employment_types=[common.EmploymentType.FULL_TIME], incentives='incentives_value', language_code='language_code_value', job_level=common.JobLevel.ENTRY_LEVEL, promotion_value=1635, qualifications='qualifications_value', responsibilities='responsibilities_value', posting_region=common.PostingRegion.ADMINISTRATIVE_AREA, visibility=common.Visibility.ACCOUNT_ONLY, company_display_name='company_display_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gct_job.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_job(request)
    assert isinstance(response, gct_job.Job)
    assert response.name == 'name_value'
    assert response.company == 'company_value'
    assert response.requisition_id == 'requisition_id_value'
    assert response.title == 'title_value'
    assert response.description == 'description_value'
    assert response.addresses == ['addresses_value']
    assert response.job_benefits == [common.JobBenefit.CHILD_CARE]
    assert response.degree_types == [common.DegreeType.PRIMARY_EDUCATION]
    assert response.department == 'department_value'
    assert response.employment_types == [common.EmploymentType.FULL_TIME]
    assert response.incentives == 'incentives_value'
    assert response.language_code == 'language_code_value'
    assert response.job_level == common.JobLevel.ENTRY_LEVEL
    assert response.promotion_value == 1635
    assert response.qualifications == 'qualifications_value'
    assert response.responsibilities == 'responsibilities_value'
    assert response.posting_region == common.PostingRegion.ADMINISTRATIVE_AREA
    assert response.visibility == common.Visibility.ACCOUNT_ONLY
    assert response.company_display_name == 'company_display_name_value'

def test_update_job_rest_required_fields(request_type=job_service.UpdateJobRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.JobServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gct_job.Job()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gct_job.Job.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_job_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('job',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_job_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobServiceRestInterceptor())
    client = JobServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobServiceRestInterceptor, 'post_update_job') as post, mock.patch.object(transports.JobServiceRestInterceptor, 'pre_update_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = job_service.UpdateJobRequest.pb(job_service.UpdateJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gct_job.Job.to_json(gct_job.Job())
        request = job_service.UpdateJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gct_job.Job()
        client.update_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_job_rest_bad_request(transport: str='rest', request_type=job_service.UpdateJobRequest):
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'job': {'name': 'projects/sample1/tenants/sample2/jobs/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_job(request)

def test_update_job_rest_flattened():
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gct_job.Job()
        sample_request = {'job': {'name': 'projects/sample1/tenants/sample2/jobs/sample3'}}
        mock_args = dict(job=gct_job.Job(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gct_job.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v4beta1/{job.name=projects/*/tenants/*/jobs/*}' % client.transport._host, args[1])

def test_update_job_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_job(job_service.UpdateJobRequest(), job=gct_job.Job(name='name_value'))

def test_update_job_rest_error():
    if False:
        i = 10
        return i + 15
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [job_service.BatchUpdateJobsRequest, dict])
def test_batch_update_jobs_rest(request_type):
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/tenants/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_update_jobs(request)
    assert response.operation.name == 'operations/spam'

def test_batch_update_jobs_rest_required_fields(request_type=job_service.BatchUpdateJobsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.JobServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_update_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_update_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.batch_update_jobs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_update_jobs_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_update_jobs._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'jobs'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_update_jobs_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobServiceRestInterceptor())
    client = JobServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.JobServiceRestInterceptor, 'post_batch_update_jobs') as post, mock.patch.object(transports.JobServiceRestInterceptor, 'pre_batch_update_jobs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = job_service.BatchUpdateJobsRequest.pb(job_service.BatchUpdateJobsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = job_service.BatchUpdateJobsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.batch_update_jobs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_update_jobs_rest_bad_request(transport: str='rest', request_type=job_service.BatchUpdateJobsRequest):
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/tenants/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_update_jobs(request)

def test_batch_update_jobs_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/tenants/sample2'}
        mock_args = dict(parent='parent_value', jobs=[job.Job(name='name_value')])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.batch_update_jobs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v4beta1/{parent=projects/*/tenants/*}/jobs:batchUpdate' % client.transport._host, args[1])

def test_batch_update_jobs_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.batch_update_jobs(job_service.BatchUpdateJobsRequest(), parent='parent_value', jobs=[job.Job(name='name_value')])

def test_batch_update_jobs_rest_error():
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [job_service.DeleteJobRequest, dict])
def test_delete_job_rest(request_type):
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/tenants/sample2/jobs/sample3'}
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

def test_delete_job_rest_required_fields(request_type=job_service.DeleteJobRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.JobServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_job_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobServiceRestInterceptor())
    client = JobServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobServiceRestInterceptor, 'pre_delete_job') as pre:
        pre.assert_not_called()
        pb_message = job_service.DeleteJobRequest.pb(job_service.DeleteJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = job_service.DeleteJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_job_rest_bad_request(transport: str='rest', request_type=job_service.DeleteJobRequest):
    if False:
        i = 10
        return i + 15
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/tenants/sample2/jobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_job(request)

def test_delete_job_rest_flattened():
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/tenants/sample2/jobs/sample3'}
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
        assert path_template.validate('%s/v4beta1/{name=projects/*/tenants/*/jobs/*}' % client.transport._host, args[1])

def test_delete_job_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_job(job_service.DeleteJobRequest(), name='name_value')

def test_delete_job_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [job_service.BatchDeleteJobsRequest, dict])
def test_batch_delete_jobs_rest(request_type):
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/tenants/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_delete_jobs(request)
    assert response is None

def test_batch_delete_jobs_rest_required_fields(request_type=job_service.BatchDeleteJobsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.JobServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['filter'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_delete_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['filter'] = 'filter_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_delete_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'filter' in jsonified_request
    assert jsonified_request['filter'] == 'filter_value'
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = None
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = ''
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.batch_delete_jobs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_delete_jobs_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_delete_jobs._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'filter'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_delete_jobs_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobServiceRestInterceptor())
    client = JobServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobServiceRestInterceptor, 'pre_batch_delete_jobs') as pre:
        pre.assert_not_called()
        pb_message = job_service.BatchDeleteJobsRequest.pb(job_service.BatchDeleteJobsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = job_service.BatchDeleteJobsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.batch_delete_jobs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_batch_delete_jobs_rest_bad_request(transport: str='rest', request_type=job_service.BatchDeleteJobsRequest):
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/tenants/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_delete_jobs(request)

def test_batch_delete_jobs_rest_flattened():
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'parent': 'projects/sample1/tenants/sample2'}
        mock_args = dict(parent='parent_value', filter='filter_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.batch_delete_jobs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v4beta1/{parent=projects/*/tenants/*}/jobs:batchDelete' % client.transport._host, args[1])

def test_batch_delete_jobs_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.batch_delete_jobs(job_service.BatchDeleteJobsRequest(), parent='parent_value', filter='filter_value')

def test_batch_delete_jobs_rest_error():
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [job_service.ListJobsRequest, dict])
def test_list_jobs_rest(request_type):
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/tenants/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = job_service.ListJobsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = job_service.ListJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_jobs(request)
    assert isinstance(response, pagers.ListJobsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_jobs_rest_required_fields(request_type=job_service.ListJobsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.JobServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['filter'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'filter' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'filter' in jsonified_request
    assert jsonified_request['filter'] == request_init['filter']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['filter'] = 'filter_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_jobs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'job_view', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'filter' in jsonified_request
    assert jsonified_request['filter'] == 'filter_value'
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = job_service.ListJobsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = job_service.ListJobsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_jobs(request)
            expected_params = [('filter', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_jobs_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_jobs._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'jobView', 'pageSize', 'pageToken')) & set(('parent', 'filter'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_jobs_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobServiceRestInterceptor())
    client = JobServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobServiceRestInterceptor, 'post_list_jobs') as post, mock.patch.object(transports.JobServiceRestInterceptor, 'pre_list_jobs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = job_service.ListJobsRequest.pb(job_service.ListJobsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = job_service.ListJobsResponse.to_json(job_service.ListJobsResponse())
        request = job_service.ListJobsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = job_service.ListJobsResponse()
        client.list_jobs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_jobs_rest_bad_request(transport: str='rest', request_type=job_service.ListJobsRequest):
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/tenants/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_jobs(request)

def test_list_jobs_rest_flattened():
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = job_service.ListJobsResponse()
        sample_request = {'parent': 'projects/sample1/tenants/sample2'}
        mock_args = dict(parent='parent_value', filter='filter_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = job_service.ListJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_jobs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v4beta1/{parent=projects/*/tenants/*}/jobs' % client.transport._host, args[1])

def test_list_jobs_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_jobs(job_service.ListJobsRequest(), parent='parent_value', filter='filter_value')

def test_list_jobs_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (job_service.ListJobsResponse(jobs=[job.Job(), job.Job(), job.Job()], next_page_token='abc'), job_service.ListJobsResponse(jobs=[], next_page_token='def'), job_service.ListJobsResponse(jobs=[job.Job()], next_page_token='ghi'), job_service.ListJobsResponse(jobs=[job.Job(), job.Job()]))
        response = response + response
        response = tuple((job_service.ListJobsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/tenants/sample2'}
        pager = client.list_jobs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, job.Job) for i in results))
        pages = list(client.list_jobs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [job_service.SearchJobsRequest, dict])
def test_search_jobs_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/tenants/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = job_service.SearchJobsResponse(next_page_token='next_page_token_value', estimated_total_size=2141, total_size=1086, broadened_query_jobs_count=2766)
        response_value = Response()
        response_value.status_code = 200
        return_value = job_service.SearchJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.search_jobs(request)
    assert isinstance(response, pagers.SearchJobsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.estimated_total_size == 2141
    assert response.total_size == 1086
    assert response.broadened_query_jobs_count == 2766

def test_search_jobs_rest_required_fields(request_type=job_service.SearchJobsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.JobServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = job_service.SearchJobsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = job_service.SearchJobsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.search_jobs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_search_jobs_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.search_jobs._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'requestMetadata'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_search_jobs_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobServiceRestInterceptor())
    client = JobServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobServiceRestInterceptor, 'post_search_jobs') as post, mock.patch.object(transports.JobServiceRestInterceptor, 'pre_search_jobs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = job_service.SearchJobsRequest.pb(job_service.SearchJobsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = job_service.SearchJobsResponse.to_json(job_service.SearchJobsResponse())
        request = job_service.SearchJobsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = job_service.SearchJobsResponse()
        client.search_jobs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_search_jobs_rest_bad_request(transport: str='rest', request_type=job_service.SearchJobsRequest):
    if False:
        i = 10
        return i + 15
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/tenants/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.search_jobs(request)

def test_search_jobs_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()], next_page_token='abc'), job_service.SearchJobsResponse(matching_jobs=[], next_page_token='def'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob()], next_page_token='ghi'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()]))
        response = response + response
        response = tuple((job_service.SearchJobsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/tenants/sample2'}
        pager = client.search_jobs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, job_service.SearchJobsResponse.MatchingJob) for i in results))
        pages = list(client.search_jobs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [job_service.SearchJobsRequest, dict])
def test_search_jobs_for_alert_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/tenants/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = job_service.SearchJobsResponse(next_page_token='next_page_token_value', estimated_total_size=2141, total_size=1086, broadened_query_jobs_count=2766)
        response_value = Response()
        response_value.status_code = 200
        return_value = job_service.SearchJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.search_jobs_for_alert(request)
    assert isinstance(response, pagers.SearchJobsForAlertPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.estimated_total_size == 2141
    assert response.total_size == 1086
    assert response.broadened_query_jobs_count == 2766

def test_search_jobs_for_alert_rest_required_fields(request_type=job_service.SearchJobsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.JobServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_jobs_for_alert._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_jobs_for_alert._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = job_service.SearchJobsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = job_service.SearchJobsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.search_jobs_for_alert(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_search_jobs_for_alert_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.search_jobs_for_alert._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'requestMetadata'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_search_jobs_for_alert_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.JobServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.JobServiceRestInterceptor())
    client = JobServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.JobServiceRestInterceptor, 'post_search_jobs_for_alert') as post, mock.patch.object(transports.JobServiceRestInterceptor, 'pre_search_jobs_for_alert') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = job_service.SearchJobsRequest.pb(job_service.SearchJobsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = job_service.SearchJobsResponse.to_json(job_service.SearchJobsResponse())
        request = job_service.SearchJobsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = job_service.SearchJobsResponse()
        client.search_jobs_for_alert(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_search_jobs_for_alert_rest_bad_request(transport: str='rest', request_type=job_service.SearchJobsRequest):
    if False:
        i = 10
        return i + 15
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/tenants/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.search_jobs_for_alert(request)

def test_search_jobs_for_alert_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()], next_page_token='abc'), job_service.SearchJobsResponse(matching_jobs=[], next_page_token='def'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob()], next_page_token='ghi'), job_service.SearchJobsResponse(matching_jobs=[job_service.SearchJobsResponse.MatchingJob(), job_service.SearchJobsResponse.MatchingJob()]))
        response = response + response
        response = tuple((job_service.SearchJobsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/tenants/sample2'}
        pager = client.search_jobs_for_alert(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, job_service.SearchJobsResponse.MatchingJob) for i in results))
        pages = list(client.search_jobs_for_alert(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.JobServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.JobServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = JobServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.JobServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = JobServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = JobServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.JobServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = JobServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.JobServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = JobServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        print('Hello World!')
    transport = transports.JobServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.JobServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.JobServiceGrpcTransport, transports.JobServiceGrpcAsyncIOTransport, transports.JobServiceRestTransport])
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
    transport = JobServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.JobServiceGrpcTransport)

def test_job_service_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.JobServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_job_service_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.talent_v4beta1.services.job_service.transports.JobServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.JobServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_job', 'batch_create_jobs', 'get_job', 'update_job', 'batch_update_jobs', 'delete_job', 'batch_delete_jobs', 'list_jobs', 'search_jobs', 'search_jobs_for_alert', 'get_operation')
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

def test_job_service_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.talent_v4beta1.services.job_service.transports.JobServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.JobServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/jobs'), quota_project_id='octopus')

def test_job_service_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.talent_v4beta1.services.job_service.transports.JobServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.JobServiceTransport()
        adc.assert_called_once()

def test_job_service_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        JobServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/jobs'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.JobServiceGrpcTransport, transports.JobServiceGrpcAsyncIOTransport])
def test_job_service_transport_auth_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/jobs'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.JobServiceGrpcTransport, transports.JobServiceGrpcAsyncIOTransport, transports.JobServiceRestTransport])
def test_job_service_transport_auth_gdch_credentials(transport_class):
    if False:
        i = 10
        return i + 15
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.JobServiceGrpcTransport, grpc_helpers), (transports.JobServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_job_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('jobs.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/jobs'), scopes=['1', '2'], default_host='jobs.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.JobServiceGrpcTransport, transports.JobServiceGrpcAsyncIOTransport])
def test_job_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_job_service_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.JobServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_job_service_rest_lro_client():
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_job_service_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='jobs.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('jobs.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://jobs.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_job_service_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='jobs.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('jobs.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://jobs.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_job_service_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = JobServiceClient(credentials=creds1, transport=transport_name)
    client2 = JobServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_job._session
    session2 = client2.transport.create_job._session
    assert session1 != session2
    session1 = client1.transport.batch_create_jobs._session
    session2 = client2.transport.batch_create_jobs._session
    assert session1 != session2
    session1 = client1.transport.get_job._session
    session2 = client2.transport.get_job._session
    assert session1 != session2
    session1 = client1.transport.update_job._session
    session2 = client2.transport.update_job._session
    assert session1 != session2
    session1 = client1.transport.batch_update_jobs._session
    session2 = client2.transport.batch_update_jobs._session
    assert session1 != session2
    session1 = client1.transport.delete_job._session
    session2 = client2.transport.delete_job._session
    assert session1 != session2
    session1 = client1.transport.batch_delete_jobs._session
    session2 = client2.transport.batch_delete_jobs._session
    assert session1 != session2
    session1 = client1.transport.list_jobs._session
    session2 = client2.transport.list_jobs._session
    assert session1 != session2
    session1 = client1.transport.search_jobs._session
    session2 = client2.transport.search_jobs._session
    assert session1 != session2
    session1 = client1.transport.search_jobs_for_alert._session
    session2 = client2.transport.search_jobs_for_alert._session
    assert session1 != session2

def test_job_service_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.JobServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_job_service_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.JobServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.JobServiceGrpcTransport, transports.JobServiceGrpcAsyncIOTransport])
def test_job_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.JobServiceGrpcTransport, transports.JobServiceGrpcAsyncIOTransport])
def test_job_service_transport_channel_mtls_with_adc(transport_class):
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

def test_job_service_grpc_lro_client():
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_job_service_grpc_lro_async_client():
    if False:
        return 10
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_company_path():
    if False:
        return 10
    project = 'squid'
    tenant = 'clam'
    company = 'whelk'
    expected = 'projects/{project}/tenants/{tenant}/companies/{company}'.format(project=project, tenant=tenant, company=company)
    actual = JobServiceClient.company_path(project, tenant, company)
    assert expected == actual

def test_parse_company_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'octopus', 'tenant': 'oyster', 'company': 'nudibranch'}
    path = JobServiceClient.company_path(**expected)
    actual = JobServiceClient.parse_company_path(path)
    assert expected == actual

def test_job_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    tenant = 'mussel'
    job = 'winkle'
    expected = 'projects/{project}/tenants/{tenant}/jobs/{job}'.format(project=project, tenant=tenant, job=job)
    actual = JobServiceClient.job_path(project, tenant, job)
    assert expected == actual

def test_parse_job_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nautilus', 'tenant': 'scallop', 'job': 'abalone'}
    path = JobServiceClient.job_path(**expected)
    actual = JobServiceClient.parse_job_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = JobServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'clam'}
    path = JobServiceClient.common_billing_account_path(**expected)
    actual = JobServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = JobServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'octopus'}
    path = JobServiceClient.common_folder_path(**expected)
    actual = JobServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = JobServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'nudibranch'}
    path = JobServiceClient.common_organization_path(**expected)
    actual = JobServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = JobServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'mussel'}
    path = JobServiceClient.common_project_path(**expected)
    actual = JobServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = JobServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = JobServiceClient.common_location_path(**expected)
    actual = JobServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.JobServiceTransport, '_prep_wrapped_messages') as prep:
        client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.JobServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = JobServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        print('Hello World!')
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/operations/sample2'}, request)
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
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/operations/sample2'}
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

def test_get_operation(transport: str='grpc'):
    if False:
        return 10
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = JobServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = JobServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(JobServiceClient, transports.JobServiceGrpcTransport), (JobServiceAsyncClient, transports.JobServiceGrpcAsyncIOTransport)])
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