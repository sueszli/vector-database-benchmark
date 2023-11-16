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
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.type import datetime_pb2
from google.type import dayofweek_pb2
from google.type import timeofday_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.osconfig_v1.services.os_config_service import OsConfigServiceAsyncClient, OsConfigServiceClient, pagers, transports
from google.cloud.osconfig_v1.types import osconfig_common, patch_deployments, patch_jobs

def client_cert_source_callback():
    if False:
        return 10
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        for i in range(10):
            print('nop')
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        for i in range(10):
            print('nop')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert OsConfigServiceClient._get_default_mtls_endpoint(None) is None
    assert OsConfigServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert OsConfigServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert OsConfigServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert OsConfigServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert OsConfigServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(OsConfigServiceClient, 'grpc'), (OsConfigServiceAsyncClient, 'grpc_asyncio'), (OsConfigServiceClient, 'rest')])
def test_os_config_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('osconfig.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://osconfig.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.OsConfigServiceGrpcTransport, 'grpc'), (transports.OsConfigServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.OsConfigServiceRestTransport, 'rest')])
def test_os_config_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(OsConfigServiceClient, 'grpc'), (OsConfigServiceAsyncClient, 'grpc_asyncio'), (OsConfigServiceClient, 'rest')])
def test_os_config_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('osconfig.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://osconfig.googleapis.com')

def test_os_config_service_client_get_transport_class():
    if False:
        for i in range(10):
            print('nop')
    transport = OsConfigServiceClient.get_transport_class()
    available_transports = [transports.OsConfigServiceGrpcTransport, transports.OsConfigServiceRestTransport]
    assert transport in available_transports
    transport = OsConfigServiceClient.get_transport_class('grpc')
    assert transport == transports.OsConfigServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(OsConfigServiceClient, transports.OsConfigServiceGrpcTransport, 'grpc'), (OsConfigServiceAsyncClient, transports.OsConfigServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (OsConfigServiceClient, transports.OsConfigServiceRestTransport, 'rest')])
@mock.patch.object(OsConfigServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(OsConfigServiceClient))
@mock.patch.object(OsConfigServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(OsConfigServiceAsyncClient))
def test_os_config_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(OsConfigServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(OsConfigServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(OsConfigServiceClient, transports.OsConfigServiceGrpcTransport, 'grpc', 'true'), (OsConfigServiceAsyncClient, transports.OsConfigServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (OsConfigServiceClient, transports.OsConfigServiceGrpcTransport, 'grpc', 'false'), (OsConfigServiceAsyncClient, transports.OsConfigServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (OsConfigServiceClient, transports.OsConfigServiceRestTransport, 'rest', 'true'), (OsConfigServiceClient, transports.OsConfigServiceRestTransport, 'rest', 'false')])
@mock.patch.object(OsConfigServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(OsConfigServiceClient))
@mock.patch.object(OsConfigServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(OsConfigServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_os_config_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [OsConfigServiceClient, OsConfigServiceAsyncClient])
@mock.patch.object(OsConfigServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(OsConfigServiceClient))
@mock.patch.object(OsConfigServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(OsConfigServiceAsyncClient))
def test_os_config_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(OsConfigServiceClient, transports.OsConfigServiceGrpcTransport, 'grpc'), (OsConfigServiceAsyncClient, transports.OsConfigServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (OsConfigServiceClient, transports.OsConfigServiceRestTransport, 'rest')])
def test_os_config_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(OsConfigServiceClient, transports.OsConfigServiceGrpcTransport, 'grpc', grpc_helpers), (OsConfigServiceAsyncClient, transports.OsConfigServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (OsConfigServiceClient, transports.OsConfigServiceRestTransport, 'rest', None)])
def test_os_config_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_os_config_service_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.osconfig_v1.services.os_config_service.transports.OsConfigServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = OsConfigServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(OsConfigServiceClient, transports.OsConfigServiceGrpcTransport, 'grpc', grpc_helpers), (OsConfigServiceAsyncClient, transports.OsConfigServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_os_config_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('osconfig.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='osconfig.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [patch_jobs.ExecutePatchJobRequest, dict])
def test_execute_patch_job(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.execute_patch_job), '__call__') as call:
        call.return_value = patch_jobs.PatchJob(name='name_value', display_name='display_name_value', description='description_value', state=patch_jobs.PatchJob.State.STARTED, dry_run=True, error_message='error_message_value', percent_complete=0.1705, patch_deployment='patch_deployment_value')
        response = client.execute_patch_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_jobs.ExecutePatchJobRequest()
    assert isinstance(response, patch_jobs.PatchJob)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.state == patch_jobs.PatchJob.State.STARTED
    assert response.dry_run is True
    assert response.error_message == 'error_message_value'
    assert math.isclose(response.percent_complete, 0.1705, rel_tol=1e-06)
    assert response.patch_deployment == 'patch_deployment_value'

def test_execute_patch_job_empty_call():
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.execute_patch_job), '__call__') as call:
        client.execute_patch_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_jobs.ExecutePatchJobRequest()

@pytest.mark.asyncio
async def test_execute_patch_job_async(transport: str='grpc_asyncio', request_type=patch_jobs.ExecutePatchJobRequest):
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.execute_patch_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_jobs.PatchJob(name='name_value', display_name='display_name_value', description='description_value', state=patch_jobs.PatchJob.State.STARTED, dry_run=True, error_message='error_message_value', percent_complete=0.1705, patch_deployment='patch_deployment_value'))
        response = await client.execute_patch_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_jobs.ExecutePatchJobRequest()
    assert isinstance(response, patch_jobs.PatchJob)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.state == patch_jobs.PatchJob.State.STARTED
    assert response.dry_run is True
    assert response.error_message == 'error_message_value'
    assert math.isclose(response.percent_complete, 0.1705, rel_tol=1e-06)
    assert response.patch_deployment == 'patch_deployment_value'

@pytest.mark.asyncio
async def test_execute_patch_job_async_from_dict():
    await test_execute_patch_job_async(request_type=dict)

def test_execute_patch_job_field_headers():
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_jobs.ExecutePatchJobRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.execute_patch_job), '__call__') as call:
        call.return_value = patch_jobs.PatchJob()
        client.execute_patch_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_execute_patch_job_field_headers_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_jobs.ExecutePatchJobRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.execute_patch_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_jobs.PatchJob())
        await client.execute_patch_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [patch_jobs.GetPatchJobRequest, dict])
def test_get_patch_job(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_patch_job), '__call__') as call:
        call.return_value = patch_jobs.PatchJob(name='name_value', display_name='display_name_value', description='description_value', state=patch_jobs.PatchJob.State.STARTED, dry_run=True, error_message='error_message_value', percent_complete=0.1705, patch_deployment='patch_deployment_value')
        response = client.get_patch_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_jobs.GetPatchJobRequest()
    assert isinstance(response, patch_jobs.PatchJob)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.state == patch_jobs.PatchJob.State.STARTED
    assert response.dry_run is True
    assert response.error_message == 'error_message_value'
    assert math.isclose(response.percent_complete, 0.1705, rel_tol=1e-06)
    assert response.patch_deployment == 'patch_deployment_value'

def test_get_patch_job_empty_call():
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_patch_job), '__call__') as call:
        client.get_patch_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_jobs.GetPatchJobRequest()

@pytest.mark.asyncio
async def test_get_patch_job_async(transport: str='grpc_asyncio', request_type=patch_jobs.GetPatchJobRequest):
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_patch_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_jobs.PatchJob(name='name_value', display_name='display_name_value', description='description_value', state=patch_jobs.PatchJob.State.STARTED, dry_run=True, error_message='error_message_value', percent_complete=0.1705, patch_deployment='patch_deployment_value'))
        response = await client.get_patch_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_jobs.GetPatchJobRequest()
    assert isinstance(response, patch_jobs.PatchJob)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.state == patch_jobs.PatchJob.State.STARTED
    assert response.dry_run is True
    assert response.error_message == 'error_message_value'
    assert math.isclose(response.percent_complete, 0.1705, rel_tol=1e-06)
    assert response.patch_deployment == 'patch_deployment_value'

@pytest.mark.asyncio
async def test_get_patch_job_async_from_dict():
    await test_get_patch_job_async(request_type=dict)

def test_get_patch_job_field_headers():
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_jobs.GetPatchJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_patch_job), '__call__') as call:
        call.return_value = patch_jobs.PatchJob()
        client.get_patch_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_patch_job_field_headers_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_jobs.GetPatchJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_patch_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_jobs.PatchJob())
        await client.get_patch_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_patch_job_flattened():
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_patch_job), '__call__') as call:
        call.return_value = patch_jobs.PatchJob()
        client.get_patch_job(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_patch_job_flattened_error():
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_patch_job(patch_jobs.GetPatchJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_patch_job_flattened_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_patch_job), '__call__') as call:
        call.return_value = patch_jobs.PatchJob()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_jobs.PatchJob())
        response = await client.get_patch_job(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_patch_job_flattened_error_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_patch_job(patch_jobs.GetPatchJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [patch_jobs.CancelPatchJobRequest, dict])
def test_cancel_patch_job(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.cancel_patch_job), '__call__') as call:
        call.return_value = patch_jobs.PatchJob(name='name_value', display_name='display_name_value', description='description_value', state=patch_jobs.PatchJob.State.STARTED, dry_run=True, error_message='error_message_value', percent_complete=0.1705, patch_deployment='patch_deployment_value')
        response = client.cancel_patch_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_jobs.CancelPatchJobRequest()
    assert isinstance(response, patch_jobs.PatchJob)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.state == patch_jobs.PatchJob.State.STARTED
    assert response.dry_run is True
    assert response.error_message == 'error_message_value'
    assert math.isclose(response.percent_complete, 0.1705, rel_tol=1e-06)
    assert response.patch_deployment == 'patch_deployment_value'

def test_cancel_patch_job_empty_call():
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.cancel_patch_job), '__call__') as call:
        client.cancel_patch_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_jobs.CancelPatchJobRequest()

@pytest.mark.asyncio
async def test_cancel_patch_job_async(transport: str='grpc_asyncio', request_type=patch_jobs.CancelPatchJobRequest):
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.cancel_patch_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_jobs.PatchJob(name='name_value', display_name='display_name_value', description='description_value', state=patch_jobs.PatchJob.State.STARTED, dry_run=True, error_message='error_message_value', percent_complete=0.1705, patch_deployment='patch_deployment_value'))
        response = await client.cancel_patch_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_jobs.CancelPatchJobRequest()
    assert isinstance(response, patch_jobs.PatchJob)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.state == patch_jobs.PatchJob.State.STARTED
    assert response.dry_run is True
    assert response.error_message == 'error_message_value'
    assert math.isclose(response.percent_complete, 0.1705, rel_tol=1e-06)
    assert response.patch_deployment == 'patch_deployment_value'

@pytest.mark.asyncio
async def test_cancel_patch_job_async_from_dict():
    await test_cancel_patch_job_async(request_type=dict)

def test_cancel_patch_job_field_headers():
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_jobs.CancelPatchJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.cancel_patch_job), '__call__') as call:
        call.return_value = patch_jobs.PatchJob()
        client.cancel_patch_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_cancel_patch_job_field_headers_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_jobs.CancelPatchJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.cancel_patch_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_jobs.PatchJob())
        await client.cancel_patch_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [patch_jobs.ListPatchJobsRequest, dict])
def test_list_patch_jobs(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_patch_jobs), '__call__') as call:
        call.return_value = patch_jobs.ListPatchJobsResponse(next_page_token='next_page_token_value')
        response = client.list_patch_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_jobs.ListPatchJobsRequest()
    assert isinstance(response, pagers.ListPatchJobsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_patch_jobs_empty_call():
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_patch_jobs), '__call__') as call:
        client.list_patch_jobs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_jobs.ListPatchJobsRequest()

@pytest.mark.asyncio
async def test_list_patch_jobs_async(transport: str='grpc_asyncio', request_type=patch_jobs.ListPatchJobsRequest):
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_patch_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_jobs.ListPatchJobsResponse(next_page_token='next_page_token_value'))
        response = await client.list_patch_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_jobs.ListPatchJobsRequest()
    assert isinstance(response, pagers.ListPatchJobsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_patch_jobs_async_from_dict():
    await test_list_patch_jobs_async(request_type=dict)

def test_list_patch_jobs_field_headers():
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_jobs.ListPatchJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_patch_jobs), '__call__') as call:
        call.return_value = patch_jobs.ListPatchJobsResponse()
        client.list_patch_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_patch_jobs_field_headers_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_jobs.ListPatchJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_patch_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_jobs.ListPatchJobsResponse())
        await client.list_patch_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_patch_jobs_flattened():
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_patch_jobs), '__call__') as call:
        call.return_value = patch_jobs.ListPatchJobsResponse()
        client.list_patch_jobs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_patch_jobs_flattened_error():
    if False:
        return 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_patch_jobs(patch_jobs.ListPatchJobsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_patch_jobs_flattened_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_patch_jobs), '__call__') as call:
        call.return_value = patch_jobs.ListPatchJobsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_jobs.ListPatchJobsResponse())
        response = await client.list_patch_jobs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_patch_jobs_flattened_error_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_patch_jobs(patch_jobs.ListPatchJobsRequest(), parent='parent_value')

def test_list_patch_jobs_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_patch_jobs), '__call__') as call:
        call.side_effect = (patch_jobs.ListPatchJobsResponse(patch_jobs=[patch_jobs.PatchJob(), patch_jobs.PatchJob(), patch_jobs.PatchJob()], next_page_token='abc'), patch_jobs.ListPatchJobsResponse(patch_jobs=[], next_page_token='def'), patch_jobs.ListPatchJobsResponse(patch_jobs=[patch_jobs.PatchJob()], next_page_token='ghi'), patch_jobs.ListPatchJobsResponse(patch_jobs=[patch_jobs.PatchJob(), patch_jobs.PatchJob()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_patch_jobs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, patch_jobs.PatchJob) for i in results))

def test_list_patch_jobs_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_patch_jobs), '__call__') as call:
        call.side_effect = (patch_jobs.ListPatchJobsResponse(patch_jobs=[patch_jobs.PatchJob(), patch_jobs.PatchJob(), patch_jobs.PatchJob()], next_page_token='abc'), patch_jobs.ListPatchJobsResponse(patch_jobs=[], next_page_token='def'), patch_jobs.ListPatchJobsResponse(patch_jobs=[patch_jobs.PatchJob()], next_page_token='ghi'), patch_jobs.ListPatchJobsResponse(patch_jobs=[patch_jobs.PatchJob(), patch_jobs.PatchJob()]), RuntimeError)
        pages = list(client.list_patch_jobs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_patch_jobs_async_pager():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_patch_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (patch_jobs.ListPatchJobsResponse(patch_jobs=[patch_jobs.PatchJob(), patch_jobs.PatchJob(), patch_jobs.PatchJob()], next_page_token='abc'), patch_jobs.ListPatchJobsResponse(patch_jobs=[], next_page_token='def'), patch_jobs.ListPatchJobsResponse(patch_jobs=[patch_jobs.PatchJob()], next_page_token='ghi'), patch_jobs.ListPatchJobsResponse(patch_jobs=[patch_jobs.PatchJob(), patch_jobs.PatchJob()]), RuntimeError)
        async_pager = await client.list_patch_jobs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, patch_jobs.PatchJob) for i in responses))

@pytest.mark.asyncio
async def test_list_patch_jobs_async_pages():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_patch_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (patch_jobs.ListPatchJobsResponse(patch_jobs=[patch_jobs.PatchJob(), patch_jobs.PatchJob(), patch_jobs.PatchJob()], next_page_token='abc'), patch_jobs.ListPatchJobsResponse(patch_jobs=[], next_page_token='def'), patch_jobs.ListPatchJobsResponse(patch_jobs=[patch_jobs.PatchJob()], next_page_token='ghi'), patch_jobs.ListPatchJobsResponse(patch_jobs=[patch_jobs.PatchJob(), patch_jobs.PatchJob()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_patch_jobs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [patch_jobs.ListPatchJobInstanceDetailsRequest, dict])
def test_list_patch_job_instance_details(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_patch_job_instance_details), '__call__') as call:
        call.return_value = patch_jobs.ListPatchJobInstanceDetailsResponse(next_page_token='next_page_token_value')
        response = client.list_patch_job_instance_details(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_jobs.ListPatchJobInstanceDetailsRequest()
    assert isinstance(response, pagers.ListPatchJobInstanceDetailsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_patch_job_instance_details_empty_call():
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_patch_job_instance_details), '__call__') as call:
        client.list_patch_job_instance_details()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_jobs.ListPatchJobInstanceDetailsRequest()

@pytest.mark.asyncio
async def test_list_patch_job_instance_details_async(transport: str='grpc_asyncio', request_type=patch_jobs.ListPatchJobInstanceDetailsRequest):
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_patch_job_instance_details), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_jobs.ListPatchJobInstanceDetailsResponse(next_page_token='next_page_token_value'))
        response = await client.list_patch_job_instance_details(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_jobs.ListPatchJobInstanceDetailsRequest()
    assert isinstance(response, pagers.ListPatchJobInstanceDetailsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_patch_job_instance_details_async_from_dict():
    await test_list_patch_job_instance_details_async(request_type=dict)

def test_list_patch_job_instance_details_field_headers():
    if False:
        return 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_jobs.ListPatchJobInstanceDetailsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_patch_job_instance_details), '__call__') as call:
        call.return_value = patch_jobs.ListPatchJobInstanceDetailsResponse()
        client.list_patch_job_instance_details(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_patch_job_instance_details_field_headers_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_jobs.ListPatchJobInstanceDetailsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_patch_job_instance_details), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_jobs.ListPatchJobInstanceDetailsResponse())
        await client.list_patch_job_instance_details(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_patch_job_instance_details_flattened():
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_patch_job_instance_details), '__call__') as call:
        call.return_value = patch_jobs.ListPatchJobInstanceDetailsResponse()
        client.list_patch_job_instance_details(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_patch_job_instance_details_flattened_error():
    if False:
        return 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_patch_job_instance_details(patch_jobs.ListPatchJobInstanceDetailsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_patch_job_instance_details_flattened_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_patch_job_instance_details), '__call__') as call:
        call.return_value = patch_jobs.ListPatchJobInstanceDetailsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_jobs.ListPatchJobInstanceDetailsResponse())
        response = await client.list_patch_job_instance_details(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_patch_job_instance_details_flattened_error_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_patch_job_instance_details(patch_jobs.ListPatchJobInstanceDetailsRequest(), parent='parent_value')

def test_list_patch_job_instance_details_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_patch_job_instance_details), '__call__') as call:
        call.side_effect = (patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[patch_jobs.PatchJobInstanceDetails(), patch_jobs.PatchJobInstanceDetails(), patch_jobs.PatchJobInstanceDetails()], next_page_token='abc'), patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[], next_page_token='def'), patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[patch_jobs.PatchJobInstanceDetails()], next_page_token='ghi'), patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[patch_jobs.PatchJobInstanceDetails(), patch_jobs.PatchJobInstanceDetails()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_patch_job_instance_details(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, patch_jobs.PatchJobInstanceDetails) for i in results))

def test_list_patch_job_instance_details_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_patch_job_instance_details), '__call__') as call:
        call.side_effect = (patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[patch_jobs.PatchJobInstanceDetails(), patch_jobs.PatchJobInstanceDetails(), patch_jobs.PatchJobInstanceDetails()], next_page_token='abc'), patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[], next_page_token='def'), patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[patch_jobs.PatchJobInstanceDetails()], next_page_token='ghi'), patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[patch_jobs.PatchJobInstanceDetails(), patch_jobs.PatchJobInstanceDetails()]), RuntimeError)
        pages = list(client.list_patch_job_instance_details(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_patch_job_instance_details_async_pager():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_patch_job_instance_details), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[patch_jobs.PatchJobInstanceDetails(), patch_jobs.PatchJobInstanceDetails(), patch_jobs.PatchJobInstanceDetails()], next_page_token='abc'), patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[], next_page_token='def'), patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[patch_jobs.PatchJobInstanceDetails()], next_page_token='ghi'), patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[patch_jobs.PatchJobInstanceDetails(), patch_jobs.PatchJobInstanceDetails()]), RuntimeError)
        async_pager = await client.list_patch_job_instance_details(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, patch_jobs.PatchJobInstanceDetails) for i in responses))

@pytest.mark.asyncio
async def test_list_patch_job_instance_details_async_pages():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_patch_job_instance_details), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[patch_jobs.PatchJobInstanceDetails(), patch_jobs.PatchJobInstanceDetails(), patch_jobs.PatchJobInstanceDetails()], next_page_token='abc'), patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[], next_page_token='def'), patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[patch_jobs.PatchJobInstanceDetails()], next_page_token='ghi'), patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[patch_jobs.PatchJobInstanceDetails(), patch_jobs.PatchJobInstanceDetails()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_patch_job_instance_details(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [patch_deployments.CreatePatchDeploymentRequest, dict])
def test_create_patch_deployment(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment(name='name_value', description='description_value', state=patch_deployments.PatchDeployment.State.ACTIVE)
        response = client.create_patch_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.CreatePatchDeploymentRequest()
    assert isinstance(response, patch_deployments.PatchDeployment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == patch_deployments.PatchDeployment.State.ACTIVE

def test_create_patch_deployment_empty_call():
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_patch_deployment), '__call__') as call:
        client.create_patch_deployment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.CreatePatchDeploymentRequest()

@pytest.mark.asyncio
async def test_create_patch_deployment_async(transport: str='grpc_asyncio', request_type=patch_deployments.CreatePatchDeploymentRequest):
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_patch_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_deployments.PatchDeployment(name='name_value', description='description_value', state=patch_deployments.PatchDeployment.State.ACTIVE))
        response = await client.create_patch_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.CreatePatchDeploymentRequest()
    assert isinstance(response, patch_deployments.PatchDeployment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == patch_deployments.PatchDeployment.State.ACTIVE

@pytest.mark.asyncio
async def test_create_patch_deployment_async_from_dict():
    await test_create_patch_deployment_async(request_type=dict)

def test_create_patch_deployment_field_headers():
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_deployments.CreatePatchDeploymentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment()
        client.create_patch_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_patch_deployment_field_headers_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_deployments.CreatePatchDeploymentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_patch_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_deployments.PatchDeployment())
        await client.create_patch_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_patch_deployment_flattened():
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment()
        client.create_patch_deployment(parent='parent_value', patch_deployment=patch_deployments.PatchDeployment(name='name_value'), patch_deployment_id='patch_deployment_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].patch_deployment
        mock_val = patch_deployments.PatchDeployment(name='name_value')
        assert arg == mock_val
        arg = args[0].patch_deployment_id
        mock_val = 'patch_deployment_id_value'
        assert arg == mock_val

def test_create_patch_deployment_flattened_error():
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_patch_deployment(patch_deployments.CreatePatchDeploymentRequest(), parent='parent_value', patch_deployment=patch_deployments.PatchDeployment(name='name_value'), patch_deployment_id='patch_deployment_id_value')

@pytest.mark.asyncio
async def test_create_patch_deployment_flattened_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_deployments.PatchDeployment())
        response = await client.create_patch_deployment(parent='parent_value', patch_deployment=patch_deployments.PatchDeployment(name='name_value'), patch_deployment_id='patch_deployment_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].patch_deployment
        mock_val = patch_deployments.PatchDeployment(name='name_value')
        assert arg == mock_val
        arg = args[0].patch_deployment_id
        mock_val = 'patch_deployment_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_patch_deployment_flattened_error_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_patch_deployment(patch_deployments.CreatePatchDeploymentRequest(), parent='parent_value', patch_deployment=patch_deployments.PatchDeployment(name='name_value'), patch_deployment_id='patch_deployment_id_value')

@pytest.mark.parametrize('request_type', [patch_deployments.GetPatchDeploymentRequest, dict])
def test_get_patch_deployment(request_type, transport: str='grpc'):
    if False:
        return 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment(name='name_value', description='description_value', state=patch_deployments.PatchDeployment.State.ACTIVE)
        response = client.get_patch_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.GetPatchDeploymentRequest()
    assert isinstance(response, patch_deployments.PatchDeployment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == patch_deployments.PatchDeployment.State.ACTIVE

def test_get_patch_deployment_empty_call():
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_patch_deployment), '__call__') as call:
        client.get_patch_deployment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.GetPatchDeploymentRequest()

@pytest.mark.asyncio
async def test_get_patch_deployment_async(transport: str='grpc_asyncio', request_type=patch_deployments.GetPatchDeploymentRequest):
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_patch_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_deployments.PatchDeployment(name='name_value', description='description_value', state=patch_deployments.PatchDeployment.State.ACTIVE))
        response = await client.get_patch_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.GetPatchDeploymentRequest()
    assert isinstance(response, patch_deployments.PatchDeployment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == patch_deployments.PatchDeployment.State.ACTIVE

@pytest.mark.asyncio
async def test_get_patch_deployment_async_from_dict():
    await test_get_patch_deployment_async(request_type=dict)

def test_get_patch_deployment_field_headers():
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_deployments.GetPatchDeploymentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment()
        client.get_patch_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_patch_deployment_field_headers_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_deployments.GetPatchDeploymentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_patch_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_deployments.PatchDeployment())
        await client.get_patch_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_patch_deployment_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment()
        client.get_patch_deployment(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_patch_deployment_flattened_error():
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_patch_deployment(patch_deployments.GetPatchDeploymentRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_patch_deployment_flattened_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_deployments.PatchDeployment())
        response = await client.get_patch_deployment(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_patch_deployment_flattened_error_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_patch_deployment(patch_deployments.GetPatchDeploymentRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [patch_deployments.ListPatchDeploymentsRequest, dict])
def test_list_patch_deployments(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_patch_deployments), '__call__') as call:
        call.return_value = patch_deployments.ListPatchDeploymentsResponse(next_page_token='next_page_token_value')
        response = client.list_patch_deployments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.ListPatchDeploymentsRequest()
    assert isinstance(response, pagers.ListPatchDeploymentsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_patch_deployments_empty_call():
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_patch_deployments), '__call__') as call:
        client.list_patch_deployments()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.ListPatchDeploymentsRequest()

@pytest.mark.asyncio
async def test_list_patch_deployments_async(transport: str='grpc_asyncio', request_type=patch_deployments.ListPatchDeploymentsRequest):
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_patch_deployments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_deployments.ListPatchDeploymentsResponse(next_page_token='next_page_token_value'))
        response = await client.list_patch_deployments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.ListPatchDeploymentsRequest()
    assert isinstance(response, pagers.ListPatchDeploymentsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_patch_deployments_async_from_dict():
    await test_list_patch_deployments_async(request_type=dict)

def test_list_patch_deployments_field_headers():
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_deployments.ListPatchDeploymentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_patch_deployments), '__call__') as call:
        call.return_value = patch_deployments.ListPatchDeploymentsResponse()
        client.list_patch_deployments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_patch_deployments_field_headers_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_deployments.ListPatchDeploymentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_patch_deployments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_deployments.ListPatchDeploymentsResponse())
        await client.list_patch_deployments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_patch_deployments_flattened():
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_patch_deployments), '__call__') as call:
        call.return_value = patch_deployments.ListPatchDeploymentsResponse()
        client.list_patch_deployments(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_patch_deployments_flattened_error():
    if False:
        return 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_patch_deployments(patch_deployments.ListPatchDeploymentsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_patch_deployments_flattened_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_patch_deployments), '__call__') as call:
        call.return_value = patch_deployments.ListPatchDeploymentsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_deployments.ListPatchDeploymentsResponse())
        response = await client.list_patch_deployments(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_patch_deployments_flattened_error_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_patch_deployments(patch_deployments.ListPatchDeploymentsRequest(), parent='parent_value')

def test_list_patch_deployments_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_patch_deployments), '__call__') as call:
        call.side_effect = (patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[patch_deployments.PatchDeployment(), patch_deployments.PatchDeployment(), patch_deployments.PatchDeployment()], next_page_token='abc'), patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[], next_page_token='def'), patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[patch_deployments.PatchDeployment()], next_page_token='ghi'), patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[patch_deployments.PatchDeployment(), patch_deployments.PatchDeployment()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_patch_deployments(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, patch_deployments.PatchDeployment) for i in results))

def test_list_patch_deployments_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_patch_deployments), '__call__') as call:
        call.side_effect = (patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[patch_deployments.PatchDeployment(), patch_deployments.PatchDeployment(), patch_deployments.PatchDeployment()], next_page_token='abc'), patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[], next_page_token='def'), patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[patch_deployments.PatchDeployment()], next_page_token='ghi'), patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[patch_deployments.PatchDeployment(), patch_deployments.PatchDeployment()]), RuntimeError)
        pages = list(client.list_patch_deployments(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_patch_deployments_async_pager():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_patch_deployments), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[patch_deployments.PatchDeployment(), patch_deployments.PatchDeployment(), patch_deployments.PatchDeployment()], next_page_token='abc'), patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[], next_page_token='def'), patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[patch_deployments.PatchDeployment()], next_page_token='ghi'), patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[patch_deployments.PatchDeployment(), patch_deployments.PatchDeployment()]), RuntimeError)
        async_pager = await client.list_patch_deployments(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, patch_deployments.PatchDeployment) for i in responses))

@pytest.mark.asyncio
async def test_list_patch_deployments_async_pages():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_patch_deployments), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[patch_deployments.PatchDeployment(), patch_deployments.PatchDeployment(), patch_deployments.PatchDeployment()], next_page_token='abc'), patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[], next_page_token='def'), patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[patch_deployments.PatchDeployment()], next_page_token='ghi'), patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[patch_deployments.PatchDeployment(), patch_deployments.PatchDeployment()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_patch_deployments(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [patch_deployments.DeletePatchDeploymentRequest, dict])
def test_delete_patch_deployment(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_patch_deployment), '__call__') as call:
        call.return_value = None
        response = client.delete_patch_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.DeletePatchDeploymentRequest()
    assert response is None

def test_delete_patch_deployment_empty_call():
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_patch_deployment), '__call__') as call:
        client.delete_patch_deployment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.DeletePatchDeploymentRequest()

@pytest.mark.asyncio
async def test_delete_patch_deployment_async(transport: str='grpc_asyncio', request_type=patch_deployments.DeletePatchDeploymentRequest):
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_patch_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_patch_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.DeletePatchDeploymentRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_patch_deployment_async_from_dict():
    await test_delete_patch_deployment_async(request_type=dict)

def test_delete_patch_deployment_field_headers():
    if False:
        return 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_deployments.DeletePatchDeploymentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_patch_deployment), '__call__') as call:
        call.return_value = None
        client.delete_patch_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_patch_deployment_field_headers_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_deployments.DeletePatchDeploymentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_patch_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_patch_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_patch_deployment_flattened():
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_patch_deployment), '__call__') as call:
        call.return_value = None
        client.delete_patch_deployment(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_patch_deployment_flattened_error():
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_patch_deployment(patch_deployments.DeletePatchDeploymentRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_patch_deployment_flattened_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_patch_deployment), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_patch_deployment(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_patch_deployment_flattened_error_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_patch_deployment(patch_deployments.DeletePatchDeploymentRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [patch_deployments.UpdatePatchDeploymentRequest, dict])
def test_update_patch_deployment(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment(name='name_value', description='description_value', state=patch_deployments.PatchDeployment.State.ACTIVE)
        response = client.update_patch_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.UpdatePatchDeploymentRequest()
    assert isinstance(response, patch_deployments.PatchDeployment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == patch_deployments.PatchDeployment.State.ACTIVE

def test_update_patch_deployment_empty_call():
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_patch_deployment), '__call__') as call:
        client.update_patch_deployment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.UpdatePatchDeploymentRequest()

@pytest.mark.asyncio
async def test_update_patch_deployment_async(transport: str='grpc_asyncio', request_type=patch_deployments.UpdatePatchDeploymentRequest):
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_patch_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_deployments.PatchDeployment(name='name_value', description='description_value', state=patch_deployments.PatchDeployment.State.ACTIVE))
        response = await client.update_patch_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.UpdatePatchDeploymentRequest()
    assert isinstance(response, patch_deployments.PatchDeployment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == patch_deployments.PatchDeployment.State.ACTIVE

@pytest.mark.asyncio
async def test_update_patch_deployment_async_from_dict():
    await test_update_patch_deployment_async(request_type=dict)

def test_update_patch_deployment_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_deployments.UpdatePatchDeploymentRequest()
    request.patch_deployment.name = 'name_value'
    with mock.patch.object(type(client.transport.update_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment()
        client.update_patch_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'patch_deployment.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_patch_deployment_field_headers_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_deployments.UpdatePatchDeploymentRequest()
    request.patch_deployment.name = 'name_value'
    with mock.patch.object(type(client.transport.update_patch_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_deployments.PatchDeployment())
        await client.update_patch_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'patch_deployment.name=name_value') in kw['metadata']

def test_update_patch_deployment_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment()
        client.update_patch_deployment(patch_deployment=patch_deployments.PatchDeployment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].patch_deployment
        mock_val = patch_deployments.PatchDeployment(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_patch_deployment_flattened_error():
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_patch_deployment(patch_deployments.UpdatePatchDeploymentRequest(), patch_deployment=patch_deployments.PatchDeployment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_patch_deployment_flattened_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_deployments.PatchDeployment())
        response = await client.update_patch_deployment(patch_deployment=patch_deployments.PatchDeployment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].patch_deployment
        mock_val = patch_deployments.PatchDeployment(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_patch_deployment_flattened_error_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_patch_deployment(patch_deployments.UpdatePatchDeploymentRequest(), patch_deployment=patch_deployments.PatchDeployment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [patch_deployments.PausePatchDeploymentRequest, dict])
def test_pause_patch_deployment(request_type, transport: str='grpc'):
    if False:
        return 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.pause_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment(name='name_value', description='description_value', state=patch_deployments.PatchDeployment.State.ACTIVE)
        response = client.pause_patch_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.PausePatchDeploymentRequest()
    assert isinstance(response, patch_deployments.PatchDeployment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == patch_deployments.PatchDeployment.State.ACTIVE

def test_pause_patch_deployment_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.pause_patch_deployment), '__call__') as call:
        client.pause_patch_deployment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.PausePatchDeploymentRequest()

@pytest.mark.asyncio
async def test_pause_patch_deployment_async(transport: str='grpc_asyncio', request_type=patch_deployments.PausePatchDeploymentRequest):
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.pause_patch_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_deployments.PatchDeployment(name='name_value', description='description_value', state=patch_deployments.PatchDeployment.State.ACTIVE))
        response = await client.pause_patch_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.PausePatchDeploymentRequest()
    assert isinstance(response, patch_deployments.PatchDeployment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == patch_deployments.PatchDeployment.State.ACTIVE

@pytest.mark.asyncio
async def test_pause_patch_deployment_async_from_dict():
    await test_pause_patch_deployment_async(request_type=dict)

def test_pause_patch_deployment_field_headers():
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_deployments.PausePatchDeploymentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.pause_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment()
        client.pause_patch_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_pause_patch_deployment_field_headers_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_deployments.PausePatchDeploymentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.pause_patch_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_deployments.PatchDeployment())
        await client.pause_patch_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_pause_patch_deployment_flattened():
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.pause_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment()
        client.pause_patch_deployment(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_pause_patch_deployment_flattened_error():
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.pause_patch_deployment(patch_deployments.PausePatchDeploymentRequest(), name='name_value')

@pytest.mark.asyncio
async def test_pause_patch_deployment_flattened_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.pause_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_deployments.PatchDeployment())
        response = await client.pause_patch_deployment(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_pause_patch_deployment_flattened_error_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.pause_patch_deployment(patch_deployments.PausePatchDeploymentRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [patch_deployments.ResumePatchDeploymentRequest, dict])
def test_resume_patch_deployment(request_type, transport: str='grpc'):
    if False:
        return 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resume_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment(name='name_value', description='description_value', state=patch_deployments.PatchDeployment.State.ACTIVE)
        response = client.resume_patch_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.ResumePatchDeploymentRequest()
    assert isinstance(response, patch_deployments.PatchDeployment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == patch_deployments.PatchDeployment.State.ACTIVE

def test_resume_patch_deployment_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.resume_patch_deployment), '__call__') as call:
        client.resume_patch_deployment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.ResumePatchDeploymentRequest()

@pytest.mark.asyncio
async def test_resume_patch_deployment_async(transport: str='grpc_asyncio', request_type=patch_deployments.ResumePatchDeploymentRequest):
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resume_patch_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_deployments.PatchDeployment(name='name_value', description='description_value', state=patch_deployments.PatchDeployment.State.ACTIVE))
        response = await client.resume_patch_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == patch_deployments.ResumePatchDeploymentRequest()
    assert isinstance(response, patch_deployments.PatchDeployment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == patch_deployments.PatchDeployment.State.ACTIVE

@pytest.mark.asyncio
async def test_resume_patch_deployment_async_from_dict():
    await test_resume_patch_deployment_async(request_type=dict)

def test_resume_patch_deployment_field_headers():
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_deployments.ResumePatchDeploymentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.resume_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment()
        client.resume_patch_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_resume_patch_deployment_field_headers_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = patch_deployments.ResumePatchDeploymentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.resume_patch_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_deployments.PatchDeployment())
        await client.resume_patch_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_resume_patch_deployment_flattened():
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.resume_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment()
        client.resume_patch_deployment(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_resume_patch_deployment_flattened_error():
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.resume_patch_deployment(patch_deployments.ResumePatchDeploymentRequest(), name='name_value')

@pytest.mark.asyncio
async def test_resume_patch_deployment_flattened_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.resume_patch_deployment), '__call__') as call:
        call.return_value = patch_deployments.PatchDeployment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(patch_deployments.PatchDeployment())
        response = await client.resume_patch_deployment(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_resume_patch_deployment_flattened_error_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.resume_patch_deployment(patch_deployments.ResumePatchDeploymentRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [patch_jobs.ExecutePatchJobRequest, dict])
def test_execute_patch_job_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_jobs.PatchJob(name='name_value', display_name='display_name_value', description='description_value', state=patch_jobs.PatchJob.State.STARTED, dry_run=True, error_message='error_message_value', percent_complete=0.1705, patch_deployment='patch_deployment_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_jobs.PatchJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.execute_patch_job(request)
    assert isinstance(response, patch_jobs.PatchJob)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.state == patch_jobs.PatchJob.State.STARTED
    assert response.dry_run is True
    assert response.error_message == 'error_message_value'
    assert math.isclose(response.percent_complete, 0.1705, rel_tol=1e-06)
    assert response.patch_deployment == 'patch_deployment_value'

def test_execute_patch_job_rest_required_fields(request_type=patch_jobs.ExecutePatchJobRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.OsConfigServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).execute_patch_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).execute_patch_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = patch_jobs.PatchJob()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = patch_jobs.PatchJob.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.execute_patch_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_execute_patch_job_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.execute_patch_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'instanceFilter'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_execute_patch_job_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigServiceRestInterceptor())
    client = OsConfigServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'post_execute_patch_job') as post, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'pre_execute_patch_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = patch_jobs.ExecutePatchJobRequest.pb(patch_jobs.ExecutePatchJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = patch_jobs.PatchJob.to_json(patch_jobs.PatchJob())
        request = patch_jobs.ExecutePatchJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = patch_jobs.PatchJob()
        client.execute_patch_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_execute_patch_job_rest_bad_request(transport: str='rest', request_type=patch_jobs.ExecutePatchJobRequest):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.execute_patch_job(request)

def test_execute_patch_job_rest_error():
    if False:
        return 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [patch_jobs.GetPatchJobRequest, dict])
def test_get_patch_job_rest(request_type):
    if False:
        return 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/patchJobs/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_jobs.PatchJob(name='name_value', display_name='display_name_value', description='description_value', state=patch_jobs.PatchJob.State.STARTED, dry_run=True, error_message='error_message_value', percent_complete=0.1705, patch_deployment='patch_deployment_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_jobs.PatchJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_patch_job(request)
    assert isinstance(response, patch_jobs.PatchJob)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.state == patch_jobs.PatchJob.State.STARTED
    assert response.dry_run is True
    assert response.error_message == 'error_message_value'
    assert math.isclose(response.percent_complete, 0.1705, rel_tol=1e-06)
    assert response.patch_deployment == 'patch_deployment_value'

def test_get_patch_job_rest_required_fields(request_type=patch_jobs.GetPatchJobRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.OsConfigServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_patch_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_patch_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = patch_jobs.PatchJob()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = patch_jobs.PatchJob.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_patch_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_patch_job_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_patch_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_patch_job_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigServiceRestInterceptor())
    client = OsConfigServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'post_get_patch_job') as post, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'pre_get_patch_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = patch_jobs.GetPatchJobRequest.pb(patch_jobs.GetPatchJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = patch_jobs.PatchJob.to_json(patch_jobs.PatchJob())
        request = patch_jobs.GetPatchJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = patch_jobs.PatchJob()
        client.get_patch_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_patch_job_rest_bad_request(transport: str='rest', request_type=patch_jobs.GetPatchJobRequest):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/patchJobs/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_patch_job(request)

def test_get_patch_job_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_jobs.PatchJob()
        sample_request = {'name': 'projects/sample1/patchJobs/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_jobs.PatchJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_patch_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/patchJobs/*}' % client.transport._host, args[1])

def test_get_patch_job_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_patch_job(patch_jobs.GetPatchJobRequest(), name='name_value')

def test_get_patch_job_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [patch_jobs.CancelPatchJobRequest, dict])
def test_cancel_patch_job_rest(request_type):
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/patchJobs/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_jobs.PatchJob(name='name_value', display_name='display_name_value', description='description_value', state=patch_jobs.PatchJob.State.STARTED, dry_run=True, error_message='error_message_value', percent_complete=0.1705, patch_deployment='patch_deployment_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_jobs.PatchJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.cancel_patch_job(request)
    assert isinstance(response, patch_jobs.PatchJob)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.state == patch_jobs.PatchJob.State.STARTED
    assert response.dry_run is True
    assert response.error_message == 'error_message_value'
    assert math.isclose(response.percent_complete, 0.1705, rel_tol=1e-06)
    assert response.patch_deployment == 'patch_deployment_value'

def test_cancel_patch_job_rest_required_fields(request_type=patch_jobs.CancelPatchJobRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.OsConfigServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).cancel_patch_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).cancel_patch_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = patch_jobs.PatchJob()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = patch_jobs.PatchJob.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.cancel_patch_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_cancel_patch_job_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.cancel_patch_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_cancel_patch_job_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigServiceRestInterceptor())
    client = OsConfigServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'post_cancel_patch_job') as post, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'pre_cancel_patch_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = patch_jobs.CancelPatchJobRequest.pb(patch_jobs.CancelPatchJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = patch_jobs.PatchJob.to_json(patch_jobs.PatchJob())
        request = patch_jobs.CancelPatchJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = patch_jobs.PatchJob()
        client.cancel_patch_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_cancel_patch_job_rest_bad_request(transport: str='rest', request_type=patch_jobs.CancelPatchJobRequest):
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/patchJobs/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.cancel_patch_job(request)

def test_cancel_patch_job_rest_error():
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [patch_jobs.ListPatchJobsRequest, dict])
def test_list_patch_jobs_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_jobs.ListPatchJobsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_jobs.ListPatchJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_patch_jobs(request)
    assert isinstance(response, pagers.ListPatchJobsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_patch_jobs_rest_required_fields(request_type=patch_jobs.ListPatchJobsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.OsConfigServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_patch_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_patch_jobs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = patch_jobs.ListPatchJobsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = patch_jobs.ListPatchJobsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_patch_jobs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_patch_jobs_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_patch_jobs._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_patch_jobs_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigServiceRestInterceptor())
    client = OsConfigServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'post_list_patch_jobs') as post, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'pre_list_patch_jobs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = patch_jobs.ListPatchJobsRequest.pb(patch_jobs.ListPatchJobsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = patch_jobs.ListPatchJobsResponse.to_json(patch_jobs.ListPatchJobsResponse())
        request = patch_jobs.ListPatchJobsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = patch_jobs.ListPatchJobsResponse()
        client.list_patch_jobs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_patch_jobs_rest_bad_request(transport: str='rest', request_type=patch_jobs.ListPatchJobsRequest):
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_patch_jobs(request)

def test_list_patch_jobs_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_jobs.ListPatchJobsResponse()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_jobs.ListPatchJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_patch_jobs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*}/patchJobs' % client.transport._host, args[1])

def test_list_patch_jobs_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_patch_jobs(patch_jobs.ListPatchJobsRequest(), parent='parent_value')

def test_list_patch_jobs_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (patch_jobs.ListPatchJobsResponse(patch_jobs=[patch_jobs.PatchJob(), patch_jobs.PatchJob(), patch_jobs.PatchJob()], next_page_token='abc'), patch_jobs.ListPatchJobsResponse(patch_jobs=[], next_page_token='def'), patch_jobs.ListPatchJobsResponse(patch_jobs=[patch_jobs.PatchJob()], next_page_token='ghi'), patch_jobs.ListPatchJobsResponse(patch_jobs=[patch_jobs.PatchJob(), patch_jobs.PatchJob()]))
        response = response + response
        response = tuple((patch_jobs.ListPatchJobsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1'}
        pager = client.list_patch_jobs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, patch_jobs.PatchJob) for i in results))
        pages = list(client.list_patch_jobs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [patch_jobs.ListPatchJobInstanceDetailsRequest, dict])
def test_list_patch_job_instance_details_rest(request_type):
    if False:
        return 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/patchJobs/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_jobs.ListPatchJobInstanceDetailsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_jobs.ListPatchJobInstanceDetailsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_patch_job_instance_details(request)
    assert isinstance(response, pagers.ListPatchJobInstanceDetailsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_patch_job_instance_details_rest_required_fields(request_type=patch_jobs.ListPatchJobInstanceDetailsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.OsConfigServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_patch_job_instance_details._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_patch_job_instance_details._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = patch_jobs.ListPatchJobInstanceDetailsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = patch_jobs.ListPatchJobInstanceDetailsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_patch_job_instance_details(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_patch_job_instance_details_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_patch_job_instance_details._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_patch_job_instance_details_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigServiceRestInterceptor())
    client = OsConfigServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'post_list_patch_job_instance_details') as post, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'pre_list_patch_job_instance_details') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = patch_jobs.ListPatchJobInstanceDetailsRequest.pb(patch_jobs.ListPatchJobInstanceDetailsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = patch_jobs.ListPatchJobInstanceDetailsResponse.to_json(patch_jobs.ListPatchJobInstanceDetailsResponse())
        request = patch_jobs.ListPatchJobInstanceDetailsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = patch_jobs.ListPatchJobInstanceDetailsResponse()
        client.list_patch_job_instance_details(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_patch_job_instance_details_rest_bad_request(transport: str='rest', request_type=patch_jobs.ListPatchJobInstanceDetailsRequest):
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/patchJobs/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_patch_job_instance_details(request)

def test_list_patch_job_instance_details_rest_flattened():
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_jobs.ListPatchJobInstanceDetailsResponse()
        sample_request = {'parent': 'projects/sample1/patchJobs/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_jobs.ListPatchJobInstanceDetailsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_patch_job_instance_details(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/patchJobs/*}/instanceDetails' % client.transport._host, args[1])

def test_list_patch_job_instance_details_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_patch_job_instance_details(patch_jobs.ListPatchJobInstanceDetailsRequest(), parent='parent_value')

def test_list_patch_job_instance_details_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[patch_jobs.PatchJobInstanceDetails(), patch_jobs.PatchJobInstanceDetails(), patch_jobs.PatchJobInstanceDetails()], next_page_token='abc'), patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[], next_page_token='def'), patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[patch_jobs.PatchJobInstanceDetails()], next_page_token='ghi'), patch_jobs.ListPatchJobInstanceDetailsResponse(patch_job_instance_details=[patch_jobs.PatchJobInstanceDetails(), patch_jobs.PatchJobInstanceDetails()]))
        response = response + response
        response = tuple((patch_jobs.ListPatchJobInstanceDetailsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/patchJobs/sample2'}
        pager = client.list_patch_job_instance_details(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, patch_jobs.PatchJobInstanceDetails) for i in results))
        pages = list(client.list_patch_job_instance_details(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [patch_deployments.CreatePatchDeploymentRequest, dict])
def test_create_patch_deployment_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request_init['patch_deployment'] = {'name': 'name_value', 'description': 'description_value', 'instance_filter': {'all_': True, 'group_labels': [{'labels': {}}], 'zones': ['zones_value1', 'zones_value2'], 'instances': ['instances_value1', 'instances_value2'], 'instance_name_prefixes': ['instance_name_prefixes_value1', 'instance_name_prefixes_value2']}, 'patch_config': {'reboot_config': 1, 'apt': {'type_': 1, 'excludes': ['excludes_value1', 'excludes_value2'], 'exclusive_packages': ['exclusive_packages_value1', 'exclusive_packages_value2']}, 'yum': {'security': True, 'minimal': True, 'excludes': ['excludes_value1', 'excludes_value2'], 'exclusive_packages': ['exclusive_packages_value1', 'exclusive_packages_value2']}, 'goo': {}, 'zypper': {'with_optional': True, 'with_update': True, 'categories': ['categories_value1', 'categories_value2'], 'severities': ['severities_value1', 'severities_value2'], 'excludes': ['excludes_value1', 'excludes_value2'], 'exclusive_patches': ['exclusive_patches_value1', 'exclusive_patches_value2']}, 'windows_update': {'classifications': [1], 'excludes': ['excludes_value1', 'excludes_value2'], 'exclusive_patches': ['exclusive_patches_value1', 'exclusive_patches_value2']}, 'pre_step': {'linux_exec_step_config': {'local_path': 'local_path_value', 'gcs_object': {'bucket': 'bucket_value', 'object_': 'object__value', 'generation_number': 1812}, 'allowed_success_codes': [2222, 2223], 'interpreter': 1}, 'windows_exec_step_config': {}}, 'post_step': {}, 'mig_instances_allowed': True}, 'duration': {'seconds': 751, 'nanos': 543}, 'one_time_schedule': {'execute_time': {'seconds': 751, 'nanos': 543}}, 'recurring_schedule': {'time_zone': {'id': 'id_value', 'version': 'version_value'}, 'start_time': {}, 'end_time': {}, 'time_of_day': {'hours': 561, 'minutes': 773, 'seconds': 751, 'nanos': 543}, 'frequency': 1, 'weekly': {'day_of_week': 1}, 'monthly': {'week_day_of_month': {'week_ordinal': 1268, 'day_of_week': 1, 'day_offset': 1060}, 'month_day': 963}, 'last_execute_time': {}, 'next_execute_time': {}}, 'create_time': {}, 'update_time': {}, 'last_execute_time': {}, 'rollout': {'mode': 1, 'disruption_budget': {'fixed': 528, 'percent': 753}}, 'state': 1}
    test_field = patch_deployments.CreatePatchDeploymentRequest.meta.fields['patch_deployment']

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
    for (field, value) in request_init['patch_deployment'].items():
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
                for i in range(0, len(request_init['patch_deployment'][field])):
                    del request_init['patch_deployment'][field][i][subfield]
            else:
                del request_init['patch_deployment'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_deployments.PatchDeployment(name='name_value', description='description_value', state=patch_deployments.PatchDeployment.State.ACTIVE)
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_deployments.PatchDeployment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_patch_deployment(request)
    assert isinstance(response, patch_deployments.PatchDeployment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == patch_deployments.PatchDeployment.State.ACTIVE

def test_create_patch_deployment_rest_required_fields(request_type=patch_deployments.CreatePatchDeploymentRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.OsConfigServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['patch_deployment_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'patchDeploymentId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_patch_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'patchDeploymentId' in jsonified_request
    assert jsonified_request['patchDeploymentId'] == request_init['patch_deployment_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['patchDeploymentId'] = 'patch_deployment_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_patch_deployment._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('patch_deployment_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'patchDeploymentId' in jsonified_request
    assert jsonified_request['patchDeploymentId'] == 'patch_deployment_id_value'
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = patch_deployments.PatchDeployment()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = patch_deployments.PatchDeployment.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_patch_deployment(request)
            expected_params = [('patchDeploymentId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_patch_deployment_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_patch_deployment._get_unset_required_fields({})
    assert set(unset_fields) == set(('patchDeploymentId',)) & set(('parent', 'patchDeploymentId', 'patchDeployment'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_patch_deployment_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigServiceRestInterceptor())
    client = OsConfigServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'post_create_patch_deployment') as post, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'pre_create_patch_deployment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = patch_deployments.CreatePatchDeploymentRequest.pb(patch_deployments.CreatePatchDeploymentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = patch_deployments.PatchDeployment.to_json(patch_deployments.PatchDeployment())
        request = patch_deployments.CreatePatchDeploymentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = patch_deployments.PatchDeployment()
        client.create_patch_deployment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_patch_deployment_rest_bad_request(transport: str='rest', request_type=patch_deployments.CreatePatchDeploymentRequest):
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_patch_deployment(request)

def test_create_patch_deployment_rest_flattened():
    if False:
        return 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_deployments.PatchDeployment()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value', patch_deployment=patch_deployments.PatchDeployment(name='name_value'), patch_deployment_id='patch_deployment_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_deployments.PatchDeployment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_patch_deployment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*}/patchDeployments' % client.transport._host, args[1])

def test_create_patch_deployment_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_patch_deployment(patch_deployments.CreatePatchDeploymentRequest(), parent='parent_value', patch_deployment=patch_deployments.PatchDeployment(name='name_value'), patch_deployment_id='patch_deployment_id_value')

def test_create_patch_deployment_rest_error():
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [patch_deployments.GetPatchDeploymentRequest, dict])
def test_get_patch_deployment_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/patchDeployments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_deployments.PatchDeployment(name='name_value', description='description_value', state=patch_deployments.PatchDeployment.State.ACTIVE)
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_deployments.PatchDeployment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_patch_deployment(request)
    assert isinstance(response, patch_deployments.PatchDeployment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == patch_deployments.PatchDeployment.State.ACTIVE

def test_get_patch_deployment_rest_required_fields(request_type=patch_deployments.GetPatchDeploymentRequest):
    if False:
        return 10
    transport_class = transports.OsConfigServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_patch_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_patch_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = patch_deployments.PatchDeployment()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = patch_deployments.PatchDeployment.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_patch_deployment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_patch_deployment_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_patch_deployment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_patch_deployment_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigServiceRestInterceptor())
    client = OsConfigServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'post_get_patch_deployment') as post, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'pre_get_patch_deployment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = patch_deployments.GetPatchDeploymentRequest.pb(patch_deployments.GetPatchDeploymentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = patch_deployments.PatchDeployment.to_json(patch_deployments.PatchDeployment())
        request = patch_deployments.GetPatchDeploymentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = patch_deployments.PatchDeployment()
        client.get_patch_deployment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_patch_deployment_rest_bad_request(transport: str='rest', request_type=patch_deployments.GetPatchDeploymentRequest):
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/patchDeployments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_patch_deployment(request)

def test_get_patch_deployment_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_deployments.PatchDeployment()
        sample_request = {'name': 'projects/sample1/patchDeployments/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_deployments.PatchDeployment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_patch_deployment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/patchDeployments/*}' % client.transport._host, args[1])

def test_get_patch_deployment_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_patch_deployment(patch_deployments.GetPatchDeploymentRequest(), name='name_value')

def test_get_patch_deployment_rest_error():
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [patch_deployments.ListPatchDeploymentsRequest, dict])
def test_list_patch_deployments_rest(request_type):
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_deployments.ListPatchDeploymentsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_deployments.ListPatchDeploymentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_patch_deployments(request)
    assert isinstance(response, pagers.ListPatchDeploymentsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_patch_deployments_rest_required_fields(request_type=patch_deployments.ListPatchDeploymentsRequest):
    if False:
        return 10
    transport_class = transports.OsConfigServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_patch_deployments._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_patch_deployments._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = patch_deployments.ListPatchDeploymentsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = patch_deployments.ListPatchDeploymentsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_patch_deployments(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_patch_deployments_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_patch_deployments._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_patch_deployments_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigServiceRestInterceptor())
    client = OsConfigServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'post_list_patch_deployments') as post, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'pre_list_patch_deployments') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = patch_deployments.ListPatchDeploymentsRequest.pb(patch_deployments.ListPatchDeploymentsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = patch_deployments.ListPatchDeploymentsResponse.to_json(patch_deployments.ListPatchDeploymentsResponse())
        request = patch_deployments.ListPatchDeploymentsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = patch_deployments.ListPatchDeploymentsResponse()
        client.list_patch_deployments(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_patch_deployments_rest_bad_request(transport: str='rest', request_type=patch_deployments.ListPatchDeploymentsRequest):
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_patch_deployments(request)

def test_list_patch_deployments_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_deployments.ListPatchDeploymentsResponse()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_deployments.ListPatchDeploymentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_patch_deployments(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*}/patchDeployments' % client.transport._host, args[1])

def test_list_patch_deployments_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_patch_deployments(patch_deployments.ListPatchDeploymentsRequest(), parent='parent_value')

def test_list_patch_deployments_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[patch_deployments.PatchDeployment(), patch_deployments.PatchDeployment(), patch_deployments.PatchDeployment()], next_page_token='abc'), patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[], next_page_token='def'), patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[patch_deployments.PatchDeployment()], next_page_token='ghi'), patch_deployments.ListPatchDeploymentsResponse(patch_deployments=[patch_deployments.PatchDeployment(), patch_deployments.PatchDeployment()]))
        response = response + response
        response = tuple((patch_deployments.ListPatchDeploymentsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1'}
        pager = client.list_patch_deployments(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, patch_deployments.PatchDeployment) for i in results))
        pages = list(client.list_patch_deployments(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [patch_deployments.DeletePatchDeploymentRequest, dict])
def test_delete_patch_deployment_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/patchDeployments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_patch_deployment(request)
    assert response is None

def test_delete_patch_deployment_rest_required_fields(request_type=patch_deployments.DeletePatchDeploymentRequest):
    if False:
        return 10
    transport_class = transports.OsConfigServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_patch_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_patch_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_patch_deployment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_patch_deployment_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_patch_deployment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_patch_deployment_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigServiceRestInterceptor())
    client = OsConfigServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'pre_delete_patch_deployment') as pre:
        pre.assert_not_called()
        pb_message = patch_deployments.DeletePatchDeploymentRequest.pb(patch_deployments.DeletePatchDeploymentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = patch_deployments.DeletePatchDeploymentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_patch_deployment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_patch_deployment_rest_bad_request(transport: str='rest', request_type=patch_deployments.DeletePatchDeploymentRequest):
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/patchDeployments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_patch_deployment(request)

def test_delete_patch_deployment_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/patchDeployments/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_patch_deployment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/patchDeployments/*}' % client.transport._host, args[1])

def test_delete_patch_deployment_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_patch_deployment(patch_deployments.DeletePatchDeploymentRequest(), name='name_value')

def test_delete_patch_deployment_rest_error():
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [patch_deployments.UpdatePatchDeploymentRequest, dict])
def test_update_patch_deployment_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'patch_deployment': {'name': 'projects/sample1/patchDeployments/sample2'}}
    request_init['patch_deployment'] = {'name': 'projects/sample1/patchDeployments/sample2', 'description': 'description_value', 'instance_filter': {'all_': True, 'group_labels': [{'labels': {}}], 'zones': ['zones_value1', 'zones_value2'], 'instances': ['instances_value1', 'instances_value2'], 'instance_name_prefixes': ['instance_name_prefixes_value1', 'instance_name_prefixes_value2']}, 'patch_config': {'reboot_config': 1, 'apt': {'type_': 1, 'excludes': ['excludes_value1', 'excludes_value2'], 'exclusive_packages': ['exclusive_packages_value1', 'exclusive_packages_value2']}, 'yum': {'security': True, 'minimal': True, 'excludes': ['excludes_value1', 'excludes_value2'], 'exclusive_packages': ['exclusive_packages_value1', 'exclusive_packages_value2']}, 'goo': {}, 'zypper': {'with_optional': True, 'with_update': True, 'categories': ['categories_value1', 'categories_value2'], 'severities': ['severities_value1', 'severities_value2'], 'excludes': ['excludes_value1', 'excludes_value2'], 'exclusive_patches': ['exclusive_patches_value1', 'exclusive_patches_value2']}, 'windows_update': {'classifications': [1], 'excludes': ['excludes_value1', 'excludes_value2'], 'exclusive_patches': ['exclusive_patches_value1', 'exclusive_patches_value2']}, 'pre_step': {'linux_exec_step_config': {'local_path': 'local_path_value', 'gcs_object': {'bucket': 'bucket_value', 'object_': 'object__value', 'generation_number': 1812}, 'allowed_success_codes': [2222, 2223], 'interpreter': 1}, 'windows_exec_step_config': {}}, 'post_step': {}, 'mig_instances_allowed': True}, 'duration': {'seconds': 751, 'nanos': 543}, 'one_time_schedule': {'execute_time': {'seconds': 751, 'nanos': 543}}, 'recurring_schedule': {'time_zone': {'id': 'id_value', 'version': 'version_value'}, 'start_time': {}, 'end_time': {}, 'time_of_day': {'hours': 561, 'minutes': 773, 'seconds': 751, 'nanos': 543}, 'frequency': 1, 'weekly': {'day_of_week': 1}, 'monthly': {'week_day_of_month': {'week_ordinal': 1268, 'day_of_week': 1, 'day_offset': 1060}, 'month_day': 963}, 'last_execute_time': {}, 'next_execute_time': {}}, 'create_time': {}, 'update_time': {}, 'last_execute_time': {}, 'rollout': {'mode': 1, 'disruption_budget': {'fixed': 528, 'percent': 753}}, 'state': 1}
    test_field = patch_deployments.UpdatePatchDeploymentRequest.meta.fields['patch_deployment']

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
    for (field, value) in request_init['patch_deployment'].items():
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
                for i in range(0, len(request_init['patch_deployment'][field])):
                    del request_init['patch_deployment'][field][i][subfield]
            else:
                del request_init['patch_deployment'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_deployments.PatchDeployment(name='name_value', description='description_value', state=patch_deployments.PatchDeployment.State.ACTIVE)
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_deployments.PatchDeployment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_patch_deployment(request)
    assert isinstance(response, patch_deployments.PatchDeployment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == patch_deployments.PatchDeployment.State.ACTIVE

def test_update_patch_deployment_rest_required_fields(request_type=patch_deployments.UpdatePatchDeploymentRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.OsConfigServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_patch_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_patch_deployment._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = patch_deployments.PatchDeployment()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = patch_deployments.PatchDeployment.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_patch_deployment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_patch_deployment_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_patch_deployment._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('patchDeployment',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_patch_deployment_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigServiceRestInterceptor())
    client = OsConfigServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'post_update_patch_deployment') as post, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'pre_update_patch_deployment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = patch_deployments.UpdatePatchDeploymentRequest.pb(patch_deployments.UpdatePatchDeploymentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = patch_deployments.PatchDeployment.to_json(patch_deployments.PatchDeployment())
        request = patch_deployments.UpdatePatchDeploymentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = patch_deployments.PatchDeployment()
        client.update_patch_deployment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_patch_deployment_rest_bad_request(transport: str='rest', request_type=patch_deployments.UpdatePatchDeploymentRequest):
    if False:
        return 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'patch_deployment': {'name': 'projects/sample1/patchDeployments/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_patch_deployment(request)

def test_update_patch_deployment_rest_flattened():
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_deployments.PatchDeployment()
        sample_request = {'patch_deployment': {'name': 'projects/sample1/patchDeployments/sample2'}}
        mock_args = dict(patch_deployment=patch_deployments.PatchDeployment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_deployments.PatchDeployment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_patch_deployment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{patch_deployment.name=projects/*/patchDeployments/*}' % client.transport._host, args[1])

def test_update_patch_deployment_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_patch_deployment(patch_deployments.UpdatePatchDeploymentRequest(), patch_deployment=patch_deployments.PatchDeployment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_patch_deployment_rest_error():
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [patch_deployments.PausePatchDeploymentRequest, dict])
def test_pause_patch_deployment_rest(request_type):
    if False:
        return 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/patchDeployments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_deployments.PatchDeployment(name='name_value', description='description_value', state=patch_deployments.PatchDeployment.State.ACTIVE)
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_deployments.PatchDeployment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.pause_patch_deployment(request)
    assert isinstance(response, patch_deployments.PatchDeployment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == patch_deployments.PatchDeployment.State.ACTIVE

def test_pause_patch_deployment_rest_required_fields(request_type=patch_deployments.PausePatchDeploymentRequest):
    if False:
        return 10
    transport_class = transports.OsConfigServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).pause_patch_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).pause_patch_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = patch_deployments.PatchDeployment()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = patch_deployments.PatchDeployment.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.pause_patch_deployment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_pause_patch_deployment_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.pause_patch_deployment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_pause_patch_deployment_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigServiceRestInterceptor())
    client = OsConfigServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'post_pause_patch_deployment') as post, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'pre_pause_patch_deployment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = patch_deployments.PausePatchDeploymentRequest.pb(patch_deployments.PausePatchDeploymentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = patch_deployments.PatchDeployment.to_json(patch_deployments.PatchDeployment())
        request = patch_deployments.PausePatchDeploymentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = patch_deployments.PatchDeployment()
        client.pause_patch_deployment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_pause_patch_deployment_rest_bad_request(transport: str='rest', request_type=patch_deployments.PausePatchDeploymentRequest):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/patchDeployments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.pause_patch_deployment(request)

def test_pause_patch_deployment_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_deployments.PatchDeployment()
        sample_request = {'name': 'projects/sample1/patchDeployments/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_deployments.PatchDeployment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.pause_patch_deployment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/patchDeployments/*}:pause' % client.transport._host, args[1])

def test_pause_patch_deployment_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.pause_patch_deployment(patch_deployments.PausePatchDeploymentRequest(), name='name_value')

def test_pause_patch_deployment_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [patch_deployments.ResumePatchDeploymentRequest, dict])
def test_resume_patch_deployment_rest(request_type):
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/patchDeployments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_deployments.PatchDeployment(name='name_value', description='description_value', state=patch_deployments.PatchDeployment.State.ACTIVE)
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_deployments.PatchDeployment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.resume_patch_deployment(request)
    assert isinstance(response, patch_deployments.PatchDeployment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == patch_deployments.PatchDeployment.State.ACTIVE

def test_resume_patch_deployment_rest_required_fields(request_type=patch_deployments.ResumePatchDeploymentRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.OsConfigServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resume_patch_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resume_patch_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = patch_deployments.PatchDeployment()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = patch_deployments.PatchDeployment.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.resume_patch_deployment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_resume_patch_deployment_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.resume_patch_deployment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_resume_patch_deployment_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.OsConfigServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigServiceRestInterceptor())
    client = OsConfigServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'post_resume_patch_deployment') as post, mock.patch.object(transports.OsConfigServiceRestInterceptor, 'pre_resume_patch_deployment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = patch_deployments.ResumePatchDeploymentRequest.pb(patch_deployments.ResumePatchDeploymentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = patch_deployments.PatchDeployment.to_json(patch_deployments.PatchDeployment())
        request = patch_deployments.ResumePatchDeploymentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = patch_deployments.PatchDeployment()
        client.resume_patch_deployment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_resume_patch_deployment_rest_bad_request(transport: str='rest', request_type=patch_deployments.ResumePatchDeploymentRequest):
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/patchDeployments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.resume_patch_deployment(request)

def test_resume_patch_deployment_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = patch_deployments.PatchDeployment()
        sample_request = {'name': 'projects/sample1/patchDeployments/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = patch_deployments.PatchDeployment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.resume_patch_deployment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/patchDeployments/*}:resume' % client.transport._host, args[1])

def test_resume_patch_deployment_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.resume_patch_deployment(patch_deployments.ResumePatchDeploymentRequest(), name='name_value')

def test_resume_patch_deployment_rest_error():
    if False:
        return 10
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.OsConfigServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.OsConfigServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = OsConfigServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.OsConfigServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = OsConfigServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = OsConfigServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.OsConfigServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = OsConfigServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.OsConfigServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = OsConfigServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.OsConfigServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.OsConfigServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.OsConfigServiceGrpcTransport, transports.OsConfigServiceGrpcAsyncIOTransport, transports.OsConfigServiceRestTransport])
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
        for i in range(10):
            print('nop')
    transport = OsConfigServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        i = 10
        return i + 15
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.OsConfigServiceGrpcTransport)

def test_os_config_service_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.OsConfigServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_os_config_service_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.osconfig_v1.services.os_config_service.transports.OsConfigServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.OsConfigServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('execute_patch_job', 'get_patch_job', 'cancel_patch_job', 'list_patch_jobs', 'list_patch_job_instance_details', 'create_patch_deployment', 'get_patch_deployment', 'list_patch_deployments', 'delete_patch_deployment', 'update_patch_deployment', 'pause_patch_deployment', 'resume_patch_deployment')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_os_config_service_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.osconfig_v1.services.os_config_service.transports.OsConfigServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.OsConfigServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_os_config_service_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.osconfig_v1.services.os_config_service.transports.OsConfigServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.OsConfigServiceTransport()
        adc.assert_called_once()

def test_os_config_service_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        OsConfigServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.OsConfigServiceGrpcTransport, transports.OsConfigServiceGrpcAsyncIOTransport])
def test_os_config_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.OsConfigServiceGrpcTransport, transports.OsConfigServiceGrpcAsyncIOTransport, transports.OsConfigServiceRestTransport])
def test_os_config_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.OsConfigServiceGrpcTransport, grpc_helpers), (transports.OsConfigServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_os_config_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('osconfig.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='osconfig.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.OsConfigServiceGrpcTransport, transports.OsConfigServiceGrpcAsyncIOTransport])
def test_os_config_service_grpc_transport_client_cert_source_for_mtls(transport_class):
    if False:
        return 10
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

def test_os_config_service_http_transport_client_cert_source_for_mtls():
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.OsConfigServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_os_config_service_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='osconfig.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('osconfig.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://osconfig.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_os_config_service_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='osconfig.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('osconfig.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://osconfig.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_os_config_service_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = OsConfigServiceClient(credentials=creds1, transport=transport_name)
    client2 = OsConfigServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.execute_patch_job._session
    session2 = client2.transport.execute_patch_job._session
    assert session1 != session2
    session1 = client1.transport.get_patch_job._session
    session2 = client2.transport.get_patch_job._session
    assert session1 != session2
    session1 = client1.transport.cancel_patch_job._session
    session2 = client2.transport.cancel_patch_job._session
    assert session1 != session2
    session1 = client1.transport.list_patch_jobs._session
    session2 = client2.transport.list_patch_jobs._session
    assert session1 != session2
    session1 = client1.transport.list_patch_job_instance_details._session
    session2 = client2.transport.list_patch_job_instance_details._session
    assert session1 != session2
    session1 = client1.transport.create_patch_deployment._session
    session2 = client2.transport.create_patch_deployment._session
    assert session1 != session2
    session1 = client1.transport.get_patch_deployment._session
    session2 = client2.transport.get_patch_deployment._session
    assert session1 != session2
    session1 = client1.transport.list_patch_deployments._session
    session2 = client2.transport.list_patch_deployments._session
    assert session1 != session2
    session1 = client1.transport.delete_patch_deployment._session
    session2 = client2.transport.delete_patch_deployment._session
    assert session1 != session2
    session1 = client1.transport.update_patch_deployment._session
    session2 = client2.transport.update_patch_deployment._session
    assert session1 != session2
    session1 = client1.transport.pause_patch_deployment._session
    session2 = client2.transport.pause_patch_deployment._session
    assert session1 != session2
    session1 = client1.transport.resume_patch_deployment._session
    session2 = client2.transport.resume_patch_deployment._session
    assert session1 != session2

def test_os_config_service_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.OsConfigServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_os_config_service_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.OsConfigServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.OsConfigServiceGrpcTransport, transports.OsConfigServiceGrpcAsyncIOTransport])
def test_os_config_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.OsConfigServiceGrpcTransport, transports.OsConfigServiceGrpcAsyncIOTransport])
def test_os_config_service_transport_channel_mtls_with_adc(transport_class):
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

def test_instance_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    zone = 'clam'
    instance = 'whelk'
    expected = 'projects/{project}/zones/{zone}/instances/{instance}'.format(project=project, zone=zone, instance=instance)
    actual = OsConfigServiceClient.instance_path(project, zone, instance)
    assert expected == actual

def test_parse_instance_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'octopus', 'zone': 'oyster', 'instance': 'nudibranch'}
    path = OsConfigServiceClient.instance_path(**expected)
    actual = OsConfigServiceClient.parse_instance_path(path)
    assert expected == actual

def test_patch_deployment_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    patch_deployment = 'mussel'
    expected = 'projects/{project}/patchDeployments/{patch_deployment}'.format(project=project, patch_deployment=patch_deployment)
    actual = OsConfigServiceClient.patch_deployment_path(project, patch_deployment)
    assert expected == actual

def test_parse_patch_deployment_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'winkle', 'patch_deployment': 'nautilus'}
    path = OsConfigServiceClient.patch_deployment_path(**expected)
    actual = OsConfigServiceClient.parse_patch_deployment_path(path)
    assert expected == actual

def test_patch_job_path():
    if False:
        while True:
            i = 10
    project = 'scallop'
    patch_job = 'abalone'
    expected = 'projects/{project}/patchJobs/{patch_job}'.format(project=project, patch_job=patch_job)
    actual = OsConfigServiceClient.patch_job_path(project, patch_job)
    assert expected == actual

def test_parse_patch_job_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'squid', 'patch_job': 'clam'}
    path = OsConfigServiceClient.patch_job_path(**expected)
    actual = OsConfigServiceClient.parse_patch_job_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = OsConfigServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'octopus'}
    path = OsConfigServiceClient.common_billing_account_path(**expected)
    actual = OsConfigServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = OsConfigServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'nudibranch'}
    path = OsConfigServiceClient.common_folder_path(**expected)
    actual = OsConfigServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = OsConfigServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'mussel'}
    path = OsConfigServiceClient.common_organization_path(**expected)
    actual = OsConfigServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = OsConfigServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nautilus'}
    path = OsConfigServiceClient.common_project_path(**expected)
    actual = OsConfigServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = OsConfigServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'squid', 'location': 'clam'}
    path = OsConfigServiceClient.common_location_path(**expected)
    actual = OsConfigServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.OsConfigServiceTransport, '_prep_wrapped_messages') as prep:
        client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.OsConfigServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = OsConfigServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = OsConfigServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        while True:
            i = 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = OsConfigServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(OsConfigServiceClient, transports.OsConfigServiceGrpcTransport), (OsConfigServiceAsyncClient, transports.OsConfigServiceGrpcAsyncIOTransport)])
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