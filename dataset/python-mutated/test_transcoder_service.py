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
from google.protobuf import timestamp_pb2
from google.rpc import status_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.video.transcoder_v1.services.transcoder_service import TranscoderServiceAsyncClient, TranscoderServiceClient, pagers, transports
from google.cloud.video.transcoder_v1.types import resources, services

def client_cert_source_callback():
    if False:
        print('Hello World!')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        print('Hello World!')
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
    assert TranscoderServiceClient._get_default_mtls_endpoint(None) is None
    assert TranscoderServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert TranscoderServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert TranscoderServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert TranscoderServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert TranscoderServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(TranscoderServiceClient, 'grpc'), (TranscoderServiceAsyncClient, 'grpc_asyncio'), (TranscoderServiceClient, 'rest')])
def test_transcoder_service_client_from_service_account_info(client_class, transport_name):
    if False:
        while True:
            i = 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('transcoder.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://transcoder.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.TranscoderServiceGrpcTransport, 'grpc'), (transports.TranscoderServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.TranscoderServiceRestTransport, 'rest')])
def test_transcoder_service_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(TranscoderServiceClient, 'grpc'), (TranscoderServiceAsyncClient, 'grpc_asyncio'), (TranscoderServiceClient, 'rest')])
def test_transcoder_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('transcoder.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://transcoder.googleapis.com')

def test_transcoder_service_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = TranscoderServiceClient.get_transport_class()
    available_transports = [transports.TranscoderServiceGrpcTransport, transports.TranscoderServiceRestTransport]
    assert transport in available_transports
    transport = TranscoderServiceClient.get_transport_class('grpc')
    assert transport == transports.TranscoderServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(TranscoderServiceClient, transports.TranscoderServiceGrpcTransport, 'grpc'), (TranscoderServiceAsyncClient, transports.TranscoderServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (TranscoderServiceClient, transports.TranscoderServiceRestTransport, 'rest')])
@mock.patch.object(TranscoderServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TranscoderServiceClient))
@mock.patch.object(TranscoderServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TranscoderServiceAsyncClient))
def test_transcoder_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(TranscoderServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(TranscoderServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(TranscoderServiceClient, transports.TranscoderServiceGrpcTransport, 'grpc', 'true'), (TranscoderServiceAsyncClient, transports.TranscoderServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (TranscoderServiceClient, transports.TranscoderServiceGrpcTransport, 'grpc', 'false'), (TranscoderServiceAsyncClient, transports.TranscoderServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (TranscoderServiceClient, transports.TranscoderServiceRestTransport, 'rest', 'true'), (TranscoderServiceClient, transports.TranscoderServiceRestTransport, 'rest', 'false')])
@mock.patch.object(TranscoderServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TranscoderServiceClient))
@mock.patch.object(TranscoderServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TranscoderServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_transcoder_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [TranscoderServiceClient, TranscoderServiceAsyncClient])
@mock.patch.object(TranscoderServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TranscoderServiceClient))
@mock.patch.object(TranscoderServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TranscoderServiceAsyncClient))
def test_transcoder_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(TranscoderServiceClient, transports.TranscoderServiceGrpcTransport, 'grpc'), (TranscoderServiceAsyncClient, transports.TranscoderServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (TranscoderServiceClient, transports.TranscoderServiceRestTransport, 'rest')])
def test_transcoder_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(TranscoderServiceClient, transports.TranscoderServiceGrpcTransport, 'grpc', grpc_helpers), (TranscoderServiceAsyncClient, transports.TranscoderServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (TranscoderServiceClient, transports.TranscoderServiceRestTransport, 'rest', None)])
def test_transcoder_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_transcoder_service_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.video.transcoder_v1.services.transcoder_service.transports.TranscoderServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = TranscoderServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(TranscoderServiceClient, transports.TranscoderServiceGrpcTransport, 'grpc', grpc_helpers), (TranscoderServiceAsyncClient, transports.TranscoderServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_transcoder_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('transcoder.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='transcoder.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [services.CreateJobRequest, dict])
def test_create_job(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = resources.Job(name='name_value', input_uri='input_uri_value', output_uri='output_uri_value', state=resources.Job.ProcessingState.PENDING, ttl_after_completion_days=2670, mode=resources.Job.ProcessingMode.PROCESSING_MODE_INTERACTIVE, batch_mode_priority=2023, optimization=resources.Job.OptimizationStrategy.AUTODETECT, template_id='template_id_value')
        response = client.create_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.CreateJobRequest()
    assert isinstance(response, resources.Job)
    assert response.name == 'name_value'
    assert response.input_uri == 'input_uri_value'
    assert response.output_uri == 'output_uri_value'
    assert response.state == resources.Job.ProcessingState.PENDING
    assert response.ttl_after_completion_days == 2670
    assert response.mode == resources.Job.ProcessingMode.PROCESSING_MODE_INTERACTIVE
    assert response.batch_mode_priority == 2023
    assert response.optimization == resources.Job.OptimizationStrategy.AUTODETECT

def test_create_job_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        client.create_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.CreateJobRequest()

@pytest.mark.asyncio
async def test_create_job_async(transport: str='grpc_asyncio', request_type=services.CreateJobRequest):
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Job(name='name_value', input_uri='input_uri_value', output_uri='output_uri_value', state=resources.Job.ProcessingState.PENDING, ttl_after_completion_days=2670, mode=resources.Job.ProcessingMode.PROCESSING_MODE_INTERACTIVE, batch_mode_priority=2023, optimization=resources.Job.OptimizationStrategy.AUTODETECT))
        response = await client.create_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.CreateJobRequest()
    assert isinstance(response, resources.Job)
    assert response.name == 'name_value'
    assert response.input_uri == 'input_uri_value'
    assert response.output_uri == 'output_uri_value'
    assert response.state == resources.Job.ProcessingState.PENDING
    assert response.ttl_after_completion_days == 2670
    assert response.mode == resources.Job.ProcessingMode.PROCESSING_MODE_INTERACTIVE
    assert response.batch_mode_priority == 2023
    assert response.optimization == resources.Job.OptimizationStrategy.AUTODETECT

@pytest.mark.asyncio
async def test_create_job_async_from_dict():
    await test_create_job_async(request_type=dict)

def test_create_job_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = services.CreateJobRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = resources.Job()
        client.create_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_job_field_headers_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = services.CreateJobRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Job())
        await client.create_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_job_flattened():
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = resources.Job()
        client.create_job(parent='parent_value', job=resources.Job(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].job
        mock_val = resources.Job(name='name_value')
        assert arg == mock_val

def test_create_job_flattened_error():
    if False:
        i = 10
        return i + 15
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_job(services.CreateJobRequest(), parent='parent_value', job=resources.Job(name='name_value'))

@pytest.mark.asyncio
async def test_create_job_flattened_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_job), '__call__') as call:
        call.return_value = resources.Job()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Job())
        response = await client.create_job(parent='parent_value', job=resources.Job(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].job
        mock_val = resources.Job(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_job_flattened_error_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_job(services.CreateJobRequest(), parent='parent_value', job=resources.Job(name='name_value'))

@pytest.mark.parametrize('request_type', [services.ListJobsRequest, dict])
def test_list_jobs(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = services.ListJobsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.ListJobsRequest()
    assert isinstance(response, pagers.ListJobsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_jobs_empty_call():
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        client.list_jobs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.ListJobsRequest()

@pytest.mark.asyncio
async def test_list_jobs_async(transport: str='grpc_asyncio', request_type=services.ListJobsRequest):
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(services.ListJobsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.ListJobsRequest()
    assert isinstance(response, pagers.ListJobsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_jobs_async_from_dict():
    await test_list_jobs_async(request_type=dict)

def test_list_jobs_field_headers():
    if False:
        while True:
            i = 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = services.ListJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = services.ListJobsResponse()
        client.list_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_jobs_field_headers_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = services.ListJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(services.ListJobsResponse())
        await client.list_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_jobs_flattened():
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = services.ListJobsResponse()
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
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_jobs(services.ListJobsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_jobs_flattened_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.return_value = services.ListJobsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(services.ListJobsResponse())
        response = await client.list_jobs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_jobs_flattened_error_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_jobs(services.ListJobsRequest(), parent='parent_value')

def test_list_jobs_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.side_effect = (services.ListJobsResponse(jobs=[resources.Job(), resources.Job(), resources.Job()], next_page_token='abc'), services.ListJobsResponse(jobs=[], next_page_token='def'), services.ListJobsResponse(jobs=[resources.Job()], next_page_token='ghi'), services.ListJobsResponse(jobs=[resources.Job(), resources.Job()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_jobs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Job) for i in results))

def test_list_jobs_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_jobs), '__call__') as call:
        call.side_effect = (services.ListJobsResponse(jobs=[resources.Job(), resources.Job(), resources.Job()], next_page_token='abc'), services.ListJobsResponse(jobs=[], next_page_token='def'), services.ListJobsResponse(jobs=[resources.Job()], next_page_token='ghi'), services.ListJobsResponse(jobs=[resources.Job(), resources.Job()]), RuntimeError)
        pages = list(client.list_jobs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_jobs_async_pager():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (services.ListJobsResponse(jobs=[resources.Job(), resources.Job(), resources.Job()], next_page_token='abc'), services.ListJobsResponse(jobs=[], next_page_token='def'), services.ListJobsResponse(jobs=[resources.Job()], next_page_token='ghi'), services.ListJobsResponse(jobs=[resources.Job(), resources.Job()]), RuntimeError)
        async_pager = await client.list_jobs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Job) for i in responses))

@pytest.mark.asyncio
async def test_list_jobs_async_pages():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (services.ListJobsResponse(jobs=[resources.Job(), resources.Job(), resources.Job()], next_page_token='abc'), services.ListJobsResponse(jobs=[], next_page_token='def'), services.ListJobsResponse(jobs=[resources.Job()], next_page_token='ghi'), services.ListJobsResponse(jobs=[resources.Job(), resources.Job()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_jobs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [services.GetJobRequest, dict])
def test_get_job(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = resources.Job(name='name_value', input_uri='input_uri_value', output_uri='output_uri_value', state=resources.Job.ProcessingState.PENDING, ttl_after_completion_days=2670, mode=resources.Job.ProcessingMode.PROCESSING_MODE_INTERACTIVE, batch_mode_priority=2023, optimization=resources.Job.OptimizationStrategy.AUTODETECT, template_id='template_id_value')
        response = client.get_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.GetJobRequest()
    assert isinstance(response, resources.Job)
    assert response.name == 'name_value'
    assert response.input_uri == 'input_uri_value'
    assert response.output_uri == 'output_uri_value'
    assert response.state == resources.Job.ProcessingState.PENDING
    assert response.ttl_after_completion_days == 2670
    assert response.mode == resources.Job.ProcessingMode.PROCESSING_MODE_INTERACTIVE
    assert response.batch_mode_priority == 2023
    assert response.optimization == resources.Job.OptimizationStrategy.AUTODETECT

def test_get_job_empty_call():
    if False:
        while True:
            i = 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        client.get_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.GetJobRequest()

@pytest.mark.asyncio
async def test_get_job_async(transport: str='grpc_asyncio', request_type=services.GetJobRequest):
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Job(name='name_value', input_uri='input_uri_value', output_uri='output_uri_value', state=resources.Job.ProcessingState.PENDING, ttl_after_completion_days=2670, mode=resources.Job.ProcessingMode.PROCESSING_MODE_INTERACTIVE, batch_mode_priority=2023, optimization=resources.Job.OptimizationStrategy.AUTODETECT))
        response = await client.get_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.GetJobRequest()
    assert isinstance(response, resources.Job)
    assert response.name == 'name_value'
    assert response.input_uri == 'input_uri_value'
    assert response.output_uri == 'output_uri_value'
    assert response.state == resources.Job.ProcessingState.PENDING
    assert response.ttl_after_completion_days == 2670
    assert response.mode == resources.Job.ProcessingMode.PROCESSING_MODE_INTERACTIVE
    assert response.batch_mode_priority == 2023
    assert response.optimization == resources.Job.OptimizationStrategy.AUTODETECT

@pytest.mark.asyncio
async def test_get_job_async_from_dict():
    await test_get_job_async(request_type=dict)

def test_get_job_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = services.GetJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = resources.Job()
        client.get_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_job_field_headers_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = services.GetJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Job())
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
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = resources.Job()
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
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_job(services.GetJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_job_flattened_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_job), '__call__') as call:
        call.return_value = resources.Job()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Job())
        response = await client.get_job(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_job_flattened_error_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_job(services.GetJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [services.DeleteJobRequest, dict])
def test_delete_job(request_type, transport: str='grpc'):
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        call.return_value = None
        response = client.delete_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.DeleteJobRequest()
    assert response is None

def test_delete_job_empty_call():
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        client.delete_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.DeleteJobRequest()

@pytest.mark.asyncio
async def test_delete_job_async(transport: str='grpc_asyncio', request_type=services.DeleteJobRequest):
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.DeleteJobRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_job_async_from_dict():
    await test_delete_job_async(request_type=dict)

def test_delete_job_field_headers():
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = services.DeleteJobRequest()
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
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = services.DeleteJobRequest()
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
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_job(services.DeleteJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_job_flattened_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_job(services.DeleteJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [services.CreateJobTemplateRequest, dict])
def test_create_job_template(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_job_template), '__call__') as call:
        call.return_value = resources.JobTemplate(name='name_value')
        response = client.create_job_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.CreateJobTemplateRequest()
    assert isinstance(response, resources.JobTemplate)
    assert response.name == 'name_value'

def test_create_job_template_empty_call():
    if False:
        while True:
            i = 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_job_template), '__call__') as call:
        client.create_job_template()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.CreateJobTemplateRequest()

@pytest.mark.asyncio
async def test_create_job_template_async(transport: str='grpc_asyncio', request_type=services.CreateJobTemplateRequest):
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_job_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.JobTemplate(name='name_value'))
        response = await client.create_job_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.CreateJobTemplateRequest()
    assert isinstance(response, resources.JobTemplate)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_create_job_template_async_from_dict():
    await test_create_job_template_async(request_type=dict)

def test_create_job_template_field_headers():
    if False:
        while True:
            i = 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = services.CreateJobTemplateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_job_template), '__call__') as call:
        call.return_value = resources.JobTemplate()
        client.create_job_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_job_template_field_headers_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = services.CreateJobTemplateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_job_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.JobTemplate())
        await client.create_job_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_job_template_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_job_template), '__call__') as call:
        call.return_value = resources.JobTemplate()
        client.create_job_template(parent='parent_value', job_template=resources.JobTemplate(name='name_value'), job_template_id='job_template_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].job_template
        mock_val = resources.JobTemplate(name='name_value')
        assert arg == mock_val
        arg = args[0].job_template_id
        mock_val = 'job_template_id_value'
        assert arg == mock_val

def test_create_job_template_flattened_error():
    if False:
        while True:
            i = 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_job_template(services.CreateJobTemplateRequest(), parent='parent_value', job_template=resources.JobTemplate(name='name_value'), job_template_id='job_template_id_value')

@pytest.mark.asyncio
async def test_create_job_template_flattened_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_job_template), '__call__') as call:
        call.return_value = resources.JobTemplate()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.JobTemplate())
        response = await client.create_job_template(parent='parent_value', job_template=resources.JobTemplate(name='name_value'), job_template_id='job_template_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].job_template
        mock_val = resources.JobTemplate(name='name_value')
        assert arg == mock_val
        arg = args[0].job_template_id
        mock_val = 'job_template_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_job_template_flattened_error_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_job_template(services.CreateJobTemplateRequest(), parent='parent_value', job_template=resources.JobTemplate(name='name_value'), job_template_id='job_template_id_value')

@pytest.mark.parametrize('request_type', [services.ListJobTemplatesRequest, dict])
def test_list_job_templates(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_job_templates), '__call__') as call:
        call.return_value = services.ListJobTemplatesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_job_templates(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.ListJobTemplatesRequest()
    assert isinstance(response, pagers.ListJobTemplatesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_job_templates_empty_call():
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_job_templates), '__call__') as call:
        client.list_job_templates()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.ListJobTemplatesRequest()

@pytest.mark.asyncio
async def test_list_job_templates_async(transport: str='grpc_asyncio', request_type=services.ListJobTemplatesRequest):
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_job_templates), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(services.ListJobTemplatesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_job_templates(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.ListJobTemplatesRequest()
    assert isinstance(response, pagers.ListJobTemplatesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_job_templates_async_from_dict():
    await test_list_job_templates_async(request_type=dict)

def test_list_job_templates_field_headers():
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = services.ListJobTemplatesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_job_templates), '__call__') as call:
        call.return_value = services.ListJobTemplatesResponse()
        client.list_job_templates(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_job_templates_field_headers_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = services.ListJobTemplatesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_job_templates), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(services.ListJobTemplatesResponse())
        await client.list_job_templates(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_job_templates_flattened():
    if False:
        i = 10
        return i + 15
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_job_templates), '__call__') as call:
        call.return_value = services.ListJobTemplatesResponse()
        client.list_job_templates(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_job_templates_flattened_error():
    if False:
        i = 10
        return i + 15
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_job_templates(services.ListJobTemplatesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_job_templates_flattened_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_job_templates), '__call__') as call:
        call.return_value = services.ListJobTemplatesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(services.ListJobTemplatesResponse())
        response = await client.list_job_templates(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_job_templates_flattened_error_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_job_templates(services.ListJobTemplatesRequest(), parent='parent_value')

def test_list_job_templates_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_job_templates), '__call__') as call:
        call.side_effect = (services.ListJobTemplatesResponse(job_templates=[resources.JobTemplate(), resources.JobTemplate(), resources.JobTemplate()], next_page_token='abc'), services.ListJobTemplatesResponse(job_templates=[], next_page_token='def'), services.ListJobTemplatesResponse(job_templates=[resources.JobTemplate()], next_page_token='ghi'), services.ListJobTemplatesResponse(job_templates=[resources.JobTemplate(), resources.JobTemplate()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_job_templates(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.JobTemplate) for i in results))

def test_list_job_templates_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_job_templates), '__call__') as call:
        call.side_effect = (services.ListJobTemplatesResponse(job_templates=[resources.JobTemplate(), resources.JobTemplate(), resources.JobTemplate()], next_page_token='abc'), services.ListJobTemplatesResponse(job_templates=[], next_page_token='def'), services.ListJobTemplatesResponse(job_templates=[resources.JobTemplate()], next_page_token='ghi'), services.ListJobTemplatesResponse(job_templates=[resources.JobTemplate(), resources.JobTemplate()]), RuntimeError)
        pages = list(client.list_job_templates(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_job_templates_async_pager():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_job_templates), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (services.ListJobTemplatesResponse(job_templates=[resources.JobTemplate(), resources.JobTemplate(), resources.JobTemplate()], next_page_token='abc'), services.ListJobTemplatesResponse(job_templates=[], next_page_token='def'), services.ListJobTemplatesResponse(job_templates=[resources.JobTemplate()], next_page_token='ghi'), services.ListJobTemplatesResponse(job_templates=[resources.JobTemplate(), resources.JobTemplate()]), RuntimeError)
        async_pager = await client.list_job_templates(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.JobTemplate) for i in responses))

@pytest.mark.asyncio
async def test_list_job_templates_async_pages():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_job_templates), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (services.ListJobTemplatesResponse(job_templates=[resources.JobTemplate(), resources.JobTemplate(), resources.JobTemplate()], next_page_token='abc'), services.ListJobTemplatesResponse(job_templates=[], next_page_token='def'), services.ListJobTemplatesResponse(job_templates=[resources.JobTemplate()], next_page_token='ghi'), services.ListJobTemplatesResponse(job_templates=[resources.JobTemplate(), resources.JobTemplate()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_job_templates(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [services.GetJobTemplateRequest, dict])
def test_get_job_template(request_type, transport: str='grpc'):
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_job_template), '__call__') as call:
        call.return_value = resources.JobTemplate(name='name_value')
        response = client.get_job_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.GetJobTemplateRequest()
    assert isinstance(response, resources.JobTemplate)
    assert response.name == 'name_value'

def test_get_job_template_empty_call():
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_job_template), '__call__') as call:
        client.get_job_template()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.GetJobTemplateRequest()

@pytest.mark.asyncio
async def test_get_job_template_async(transport: str='grpc_asyncio', request_type=services.GetJobTemplateRequest):
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_job_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.JobTemplate(name='name_value'))
        response = await client.get_job_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.GetJobTemplateRequest()
    assert isinstance(response, resources.JobTemplate)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_job_template_async_from_dict():
    await test_get_job_template_async(request_type=dict)

def test_get_job_template_field_headers():
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = services.GetJobTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_job_template), '__call__') as call:
        call.return_value = resources.JobTemplate()
        client.get_job_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_job_template_field_headers_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = services.GetJobTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_job_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.JobTemplate())
        await client.get_job_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_job_template_flattened():
    if False:
        i = 10
        return i + 15
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_job_template), '__call__') as call:
        call.return_value = resources.JobTemplate()
        client.get_job_template(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_job_template_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_job_template(services.GetJobTemplateRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_job_template_flattened_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_job_template), '__call__') as call:
        call.return_value = resources.JobTemplate()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.JobTemplate())
        response = await client.get_job_template(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_job_template_flattened_error_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_job_template(services.GetJobTemplateRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [services.DeleteJobTemplateRequest, dict])
def test_delete_job_template(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_job_template), '__call__') as call:
        call.return_value = None
        response = client.delete_job_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.DeleteJobTemplateRequest()
    assert response is None

def test_delete_job_template_empty_call():
    if False:
        i = 10
        return i + 15
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_job_template), '__call__') as call:
        client.delete_job_template()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.DeleteJobTemplateRequest()

@pytest.mark.asyncio
async def test_delete_job_template_async(transport: str='grpc_asyncio', request_type=services.DeleteJobTemplateRequest):
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_job_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_job_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == services.DeleteJobTemplateRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_job_template_async_from_dict():
    await test_delete_job_template_async(request_type=dict)

def test_delete_job_template_field_headers():
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = services.DeleteJobTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_job_template), '__call__') as call:
        call.return_value = None
        client.delete_job_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_job_template_field_headers_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = services.DeleteJobTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_job_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_job_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_job_template_flattened():
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_job_template), '__call__') as call:
        call.return_value = None
        client.delete_job_template(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_job_template_flattened_error():
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_job_template(services.DeleteJobTemplateRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_job_template_flattened_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_job_template), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_job_template(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_job_template_flattened_error_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_job_template(services.DeleteJobTemplateRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [services.CreateJobRequest, dict])
def test_create_job_rest(request_type):
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['job'] = {'name': 'name_value', 'input_uri': 'input_uri_value', 'output_uri': 'output_uri_value', 'template_id': 'template_id_value', 'config': {'inputs': [{'key': 'key_value', 'uri': 'uri_value', 'preprocessing_config': {'color': {'saturation': 0.10980000000000001, 'contrast': 0.878, 'brightness': 0.1081}, 'denoise': {'strength': 0.879, 'tune': 'tune_value'}, 'deblock': {'strength': 0.879, 'enabled': True}, 'audio': {'lufs': 0.442, 'high_boost': True, 'low_boost': True}, 'crop': {'top_pixels': 1095, 'bottom_pixels': 1417, 'left_pixels': 1183, 'right_pixels': 1298}, 'pad': {'top_pixels': 1095, 'bottom_pixels': 1417, 'left_pixels': 1183, 'right_pixels': 1298}, 'deinterlace': {'yadif': {'mode': 'mode_value', 'disable_spatial_interlacing': True, 'parity': 'parity_value', 'deinterlace_all_frames': True}, 'bwdif': {'mode': 'mode_value', 'parity': 'parity_value', 'deinterlace_all_frames': True}}}}], 'edit_list': [{'key': 'key_value', 'inputs': ['inputs_value1', 'inputs_value2'], 'end_time_offset': {'seconds': 751, 'nanos': 543}, 'start_time_offset': {}}], 'elementary_streams': [{'key': 'key_value', 'video_stream': {'h264': {'width_pixels': 1300, 'height_pixels': 1389, 'frame_rate': 0.1046, 'bitrate_bps': 1167, 'pixel_format': 'pixel_format_value', 'rate_control_mode': 'rate_control_mode_value', 'crf_level': 946, 'allow_open_gop': True, 'gop_frame_count': 1592, 'gop_duration': {}, 'enable_two_pass': True, 'vbv_size_bits': 1401, 'vbv_fullness_bits': 1834, 'entropy_coder': 'entropy_coder_value', 'b_pyramid': True, 'b_frame_count': 1364, 'aq_strength': 0.1184, 'profile': 'profile_value', 'tune': 'tune_value', 'preset': 'preset_value'}, 'h265': {'width_pixels': 1300, 'height_pixels': 1389, 'frame_rate': 0.1046, 'bitrate_bps': 1167, 'pixel_format': 'pixel_format_value', 'rate_control_mode': 'rate_control_mode_value', 'crf_level': 946, 'allow_open_gop': True, 'gop_frame_count': 1592, 'gop_duration': {}, 'enable_two_pass': True, 'vbv_size_bits': 1401, 'vbv_fullness_bits': 1834, 'b_pyramid': True, 'b_frame_count': 1364, 'aq_strength': 0.1184, 'profile': 'profile_value', 'tune': 'tune_value', 'preset': 'preset_value'}, 'vp9': {'width_pixels': 1300, 'height_pixels': 1389, 'frame_rate': 0.1046, 'bitrate_bps': 1167, 'pixel_format': 'pixel_format_value', 'rate_control_mode': 'rate_control_mode_value', 'crf_level': 946, 'gop_frame_count': 1592, 'gop_duration': {}, 'profile': 'profile_value'}}, 'audio_stream': {'codec': 'codec_value', 'bitrate_bps': 1167, 'channel_count': 1377, 'channel_layout': ['channel_layout_value1', 'channel_layout_value2'], 'mapping_': [{'atom_key': 'atom_key_value', 'input_key': 'input_key_value', 'input_track': 1188, 'input_channel': 1384, 'output_channel': 1513, 'gain_db': 0.708}], 'sample_rate_hertz': 1817, 'language_code': 'language_code_value', 'display_name': 'display_name_value'}, 'text_stream': {'codec': 'codec_value', 'language_code': 'language_code_value', 'mapping_': [{'atom_key': 'atom_key_value', 'input_key': 'input_key_value', 'input_track': 1188}], 'display_name': 'display_name_value'}}], 'mux_streams': [{'key': 'key_value', 'file_name': 'file_name_value', 'container': 'container_value', 'elementary_streams': ['elementary_streams_value1', 'elementary_streams_value2'], 'segment_settings': {'segment_duration': {}, 'individual_segments': True}, 'encryption_id': 'encryption_id_value'}], 'manifests': [{'file_name': 'file_name_value', 'type_': 1, 'mux_streams': ['mux_streams_value1', 'mux_streams_value2'], 'dash': {'segment_reference_scheme': 1}}], 'output': {'uri': 'uri_value'}, 'ad_breaks': [{'start_time_offset': {}}], 'pubsub_destination': {'topic': 'topic_value'}, 'sprite_sheets': [{'format_': 'format__value', 'file_prefix': 'file_prefix_value', 'sprite_width_pixels': 2058, 'sprite_height_pixels': 2147, 'column_count': 1302, 'row_count': 992, 'start_time_offset': {}, 'end_time_offset': {}, 'total_count': 1196, 'interval': {}, 'quality': 777}], 'overlays': [{'image': {'uri': 'uri_value', 'resolution': {'x': 0.12, 'y': 0.121}, 'alpha': 0.518}, 'animations': [{'animation_static': {'xy': {}, 'start_time_offset': {}}, 'animation_fade': {'fade_type': 1, 'xy': {}, 'start_time_offset': {}, 'end_time_offset': {}}, 'animation_end': {'start_time_offset': {}}}]}], 'encryptions': [{'id': 'id_value', 'aes_128': {}, 'sample_aes': {}, 'mpeg_cenc': {'scheme': 'scheme_value'}, 'secret_manager_key_source': {'secret_version': 'secret_version_value'}, 'drm_systems': {'widevine': {}, 'fairplay': {}, 'playready': {}, 'clearkey': {}}}]}, 'state': 1, 'create_time': {'seconds': 751, 'nanos': 543}, 'start_time': {}, 'end_time': {}, 'ttl_after_completion_days': 2670, 'labels': {}, 'error': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}, 'mode': 1, 'batch_mode_priority': 2023, 'optimization': 1}
    test_field = services.CreateJobRequest.meta.fields['job']

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
        return_value = resources.Job(name='name_value', input_uri='input_uri_value', output_uri='output_uri_value', state=resources.Job.ProcessingState.PENDING, ttl_after_completion_days=2670, mode=resources.Job.ProcessingMode.PROCESSING_MODE_INTERACTIVE, batch_mode_priority=2023, optimization=resources.Job.OptimizationStrategy.AUTODETECT, template_id='template_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_job(request)
    assert isinstance(response, resources.Job)
    assert response.name == 'name_value'
    assert response.input_uri == 'input_uri_value'
    assert response.output_uri == 'output_uri_value'
    assert response.state == resources.Job.ProcessingState.PENDING
    assert response.ttl_after_completion_days == 2670
    assert response.mode == resources.Job.ProcessingMode.PROCESSING_MODE_INTERACTIVE
    assert response.batch_mode_priority == 2023
    assert response.optimization == resources.Job.OptimizationStrategy.AUTODETECT

def test_create_job_rest_required_fields(request_type=services.CreateJobRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.TranscoderServiceRestTransport
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
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Job()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Job.pb(return_value)
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
    transport = transports.TranscoderServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'job'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_job_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.TranscoderServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TranscoderServiceRestInterceptor())
    client = TranscoderServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TranscoderServiceRestInterceptor, 'post_create_job') as post, mock.patch.object(transports.TranscoderServiceRestInterceptor, 'pre_create_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = services.CreateJobRequest.pb(services.CreateJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Job.to_json(resources.Job())
        request = services.CreateJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Job()
        client.create_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_job_rest_bad_request(transport: str='rest', request_type=services.CreateJobRequest):
    if False:
        for i in range(10):
            print('nop')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Job()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', job=resources.Job(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/jobs' % client.transport._host, args[1])

def test_create_job_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_job(services.CreateJobRequest(), parent='parent_value', job=resources.Job(name='name_value'))

def test_create_job_rest_error():
    if False:
        while True:
            i = 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [services.ListJobsRequest, dict])
def test_list_jobs_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = services.ListJobsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = services.ListJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_jobs(request)
    assert isinstance(response, pagers.ListJobsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_jobs_rest_required_fields(request_type=services.ListJobsRequest):
    if False:
        return 10
    transport_class = transports.TranscoderServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_jobs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = services.ListJobsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = services.ListJobsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_jobs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_jobs_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TranscoderServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_jobs._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_jobs_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.TranscoderServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TranscoderServiceRestInterceptor())
    client = TranscoderServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TranscoderServiceRestInterceptor, 'post_list_jobs') as post, mock.patch.object(transports.TranscoderServiceRestInterceptor, 'pre_list_jobs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = services.ListJobsRequest.pb(services.ListJobsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = services.ListJobsResponse.to_json(services.ListJobsResponse())
        request = services.ListJobsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = services.ListJobsResponse()
        client.list_jobs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_jobs_rest_bad_request(transport: str='rest', request_type=services.ListJobsRequest):
    if False:
        while True:
            i = 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = services.ListJobsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = services.ListJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_jobs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/jobs' % client.transport._host, args[1])

def test_list_jobs_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_jobs(services.ListJobsRequest(), parent='parent_value')

def test_list_jobs_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (services.ListJobsResponse(jobs=[resources.Job(), resources.Job(), resources.Job()], next_page_token='abc'), services.ListJobsResponse(jobs=[], next_page_token='def'), services.ListJobsResponse(jobs=[resources.Job()], next_page_token='ghi'), services.ListJobsResponse(jobs=[resources.Job(), resources.Job()]))
        response = response + response
        response = tuple((services.ListJobsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_jobs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Job) for i in results))
        pages = list(client.list_jobs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [services.GetJobRequest, dict])
def test_get_job_rest(request_type):
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/jobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Job(name='name_value', input_uri='input_uri_value', output_uri='output_uri_value', state=resources.Job.ProcessingState.PENDING, ttl_after_completion_days=2670, mode=resources.Job.ProcessingMode.PROCESSING_MODE_INTERACTIVE, batch_mode_priority=2023, optimization=resources.Job.OptimizationStrategy.AUTODETECT, template_id='template_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_job(request)
    assert isinstance(response, resources.Job)
    assert response.name == 'name_value'
    assert response.input_uri == 'input_uri_value'
    assert response.output_uri == 'output_uri_value'
    assert response.state == resources.Job.ProcessingState.PENDING
    assert response.ttl_after_completion_days == 2670
    assert response.mode == resources.Job.ProcessingMode.PROCESSING_MODE_INTERACTIVE
    assert response.batch_mode_priority == 2023
    assert response.optimization == resources.Job.OptimizationStrategy.AUTODETECT

def test_get_job_rest_required_fields(request_type=services.GetJobRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.TranscoderServiceRestTransport
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
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Job()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Job.pb(return_value)
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
    transport = transports.TranscoderServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_job_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.TranscoderServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TranscoderServiceRestInterceptor())
    client = TranscoderServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TranscoderServiceRestInterceptor, 'post_get_job') as post, mock.patch.object(transports.TranscoderServiceRestInterceptor, 'pre_get_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = services.GetJobRequest.pb(services.GetJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Job.to_json(resources.Job())
        request = services.GetJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Job()
        client.get_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_job_rest_bad_request(transport: str='rest', request_type=services.GetJobRequest):
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Job()
        sample_request = {'name': 'projects/sample1/locations/sample2/jobs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Job.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/jobs/*}' % client.transport._host, args[1])

def test_get_job_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_job(services.GetJobRequest(), name='name_value')

def test_get_job_rest_error():
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [services.DeleteJobRequest, dict])
def test_delete_job_rest(request_type):
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_delete_job_rest_required_fields(request_type=services.DeleteJobRequest):
    if False:
        print('Hello World!')
    transport_class = transports.TranscoderServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_job._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.TranscoderServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_job._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_job_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TranscoderServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TranscoderServiceRestInterceptor())
    client = TranscoderServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TranscoderServiceRestInterceptor, 'pre_delete_job') as pre:
        pre.assert_not_called()
        pb_message = services.DeleteJobRequest.pb(services.DeleteJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = services.DeleteJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_job_rest_bad_request(transport: str='rest', request_type=services.DeleteJobRequest):
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/jobs/*}' % client.transport._host, args[1])

def test_delete_job_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_job(services.DeleteJobRequest(), name='name_value')

def test_delete_job_rest_error():
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [services.CreateJobTemplateRequest, dict])
def test_create_job_template_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['job_template'] = {'name': 'name_value', 'config': {'inputs': [{'key': 'key_value', 'uri': 'uri_value', 'preprocessing_config': {'color': {'saturation': 0.10980000000000001, 'contrast': 0.878, 'brightness': 0.1081}, 'denoise': {'strength': 0.879, 'tune': 'tune_value'}, 'deblock': {'strength': 0.879, 'enabled': True}, 'audio': {'lufs': 0.442, 'high_boost': True, 'low_boost': True}, 'crop': {'top_pixels': 1095, 'bottom_pixels': 1417, 'left_pixels': 1183, 'right_pixels': 1298}, 'pad': {'top_pixels': 1095, 'bottom_pixels': 1417, 'left_pixels': 1183, 'right_pixels': 1298}, 'deinterlace': {'yadif': {'mode': 'mode_value', 'disable_spatial_interlacing': True, 'parity': 'parity_value', 'deinterlace_all_frames': True}, 'bwdif': {'mode': 'mode_value', 'parity': 'parity_value', 'deinterlace_all_frames': True}}}}], 'edit_list': [{'key': 'key_value', 'inputs': ['inputs_value1', 'inputs_value2'], 'end_time_offset': {'seconds': 751, 'nanos': 543}, 'start_time_offset': {}}], 'elementary_streams': [{'key': 'key_value', 'video_stream': {'h264': {'width_pixels': 1300, 'height_pixels': 1389, 'frame_rate': 0.1046, 'bitrate_bps': 1167, 'pixel_format': 'pixel_format_value', 'rate_control_mode': 'rate_control_mode_value', 'crf_level': 946, 'allow_open_gop': True, 'gop_frame_count': 1592, 'gop_duration': {}, 'enable_two_pass': True, 'vbv_size_bits': 1401, 'vbv_fullness_bits': 1834, 'entropy_coder': 'entropy_coder_value', 'b_pyramid': True, 'b_frame_count': 1364, 'aq_strength': 0.1184, 'profile': 'profile_value', 'tune': 'tune_value', 'preset': 'preset_value'}, 'h265': {'width_pixels': 1300, 'height_pixels': 1389, 'frame_rate': 0.1046, 'bitrate_bps': 1167, 'pixel_format': 'pixel_format_value', 'rate_control_mode': 'rate_control_mode_value', 'crf_level': 946, 'allow_open_gop': True, 'gop_frame_count': 1592, 'gop_duration': {}, 'enable_two_pass': True, 'vbv_size_bits': 1401, 'vbv_fullness_bits': 1834, 'b_pyramid': True, 'b_frame_count': 1364, 'aq_strength': 0.1184, 'profile': 'profile_value', 'tune': 'tune_value', 'preset': 'preset_value'}, 'vp9': {'width_pixels': 1300, 'height_pixels': 1389, 'frame_rate': 0.1046, 'bitrate_bps': 1167, 'pixel_format': 'pixel_format_value', 'rate_control_mode': 'rate_control_mode_value', 'crf_level': 946, 'gop_frame_count': 1592, 'gop_duration': {}, 'profile': 'profile_value'}}, 'audio_stream': {'codec': 'codec_value', 'bitrate_bps': 1167, 'channel_count': 1377, 'channel_layout': ['channel_layout_value1', 'channel_layout_value2'], 'mapping_': [{'atom_key': 'atom_key_value', 'input_key': 'input_key_value', 'input_track': 1188, 'input_channel': 1384, 'output_channel': 1513, 'gain_db': 0.708}], 'sample_rate_hertz': 1817, 'language_code': 'language_code_value', 'display_name': 'display_name_value'}, 'text_stream': {'codec': 'codec_value', 'language_code': 'language_code_value', 'mapping_': [{'atom_key': 'atom_key_value', 'input_key': 'input_key_value', 'input_track': 1188}], 'display_name': 'display_name_value'}}], 'mux_streams': [{'key': 'key_value', 'file_name': 'file_name_value', 'container': 'container_value', 'elementary_streams': ['elementary_streams_value1', 'elementary_streams_value2'], 'segment_settings': {'segment_duration': {}, 'individual_segments': True}, 'encryption_id': 'encryption_id_value'}], 'manifests': [{'file_name': 'file_name_value', 'type_': 1, 'mux_streams': ['mux_streams_value1', 'mux_streams_value2'], 'dash': {'segment_reference_scheme': 1}}], 'output': {'uri': 'uri_value'}, 'ad_breaks': [{'start_time_offset': {}}], 'pubsub_destination': {'topic': 'topic_value'}, 'sprite_sheets': [{'format_': 'format__value', 'file_prefix': 'file_prefix_value', 'sprite_width_pixels': 2058, 'sprite_height_pixels': 2147, 'column_count': 1302, 'row_count': 992, 'start_time_offset': {}, 'end_time_offset': {}, 'total_count': 1196, 'interval': {}, 'quality': 777}], 'overlays': [{'image': {'uri': 'uri_value', 'resolution': {'x': 0.12, 'y': 0.121}, 'alpha': 0.518}, 'animations': [{'animation_static': {'xy': {}, 'start_time_offset': {}}, 'animation_fade': {'fade_type': 1, 'xy': {}, 'start_time_offset': {}, 'end_time_offset': {}}, 'animation_end': {'start_time_offset': {}}}]}], 'encryptions': [{'id': 'id_value', 'aes_128': {}, 'sample_aes': {}, 'mpeg_cenc': {'scheme': 'scheme_value'}, 'secret_manager_key_source': {'secret_version': 'secret_version_value'}, 'drm_systems': {'widevine': {}, 'fairplay': {}, 'playready': {}, 'clearkey': {}}}]}, 'labels': {}}
    test_field = services.CreateJobTemplateRequest.meta.fields['job_template']

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
    for (field, value) in request_init['job_template'].items():
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
                for i in range(0, len(request_init['job_template'][field])):
                    del request_init['job_template'][field][i][subfield]
            else:
                del request_init['job_template'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.JobTemplate(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.JobTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_job_template(request)
    assert isinstance(response, resources.JobTemplate)
    assert response.name == 'name_value'

def test_create_job_template_rest_required_fields(request_type=services.CreateJobTemplateRequest):
    if False:
        return 10
    transport_class = transports.TranscoderServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['job_template_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'jobTemplateId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_job_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'jobTemplateId' in jsonified_request
    assert jsonified_request['jobTemplateId'] == request_init['job_template_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['jobTemplateId'] = 'job_template_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_job_template._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('job_template_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'jobTemplateId' in jsonified_request
    assert jsonified_request['jobTemplateId'] == 'job_template_id_value'
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.JobTemplate()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.JobTemplate.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_job_template(request)
            expected_params = [('jobTemplateId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_job_template_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.TranscoderServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_job_template._get_unset_required_fields({})
    assert set(unset_fields) == set(('jobTemplateId',)) & set(('parent', 'jobTemplate', 'jobTemplateId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_job_template_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.TranscoderServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TranscoderServiceRestInterceptor())
    client = TranscoderServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TranscoderServiceRestInterceptor, 'post_create_job_template') as post, mock.patch.object(transports.TranscoderServiceRestInterceptor, 'pre_create_job_template') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = services.CreateJobTemplateRequest.pb(services.CreateJobTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.JobTemplate.to_json(resources.JobTemplate())
        request = services.CreateJobTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.JobTemplate()
        client.create_job_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_job_template_rest_bad_request(transport: str='rest', request_type=services.CreateJobTemplateRequest):
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_job_template(request)

def test_create_job_template_rest_flattened():
    if False:
        while True:
            i = 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.JobTemplate()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', job_template=resources.JobTemplate(name='name_value'), job_template_id='job_template_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.JobTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_job_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/jobTemplates' % client.transport._host, args[1])

def test_create_job_template_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_job_template(services.CreateJobTemplateRequest(), parent='parent_value', job_template=resources.JobTemplate(name='name_value'), job_template_id='job_template_id_value')

def test_create_job_template_rest_error():
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [services.ListJobTemplatesRequest, dict])
def test_list_job_templates_rest(request_type):
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = services.ListJobTemplatesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = services.ListJobTemplatesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_job_templates(request)
    assert isinstance(response, pagers.ListJobTemplatesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_job_templates_rest_required_fields(request_type=services.ListJobTemplatesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.TranscoderServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_job_templates._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_job_templates._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = services.ListJobTemplatesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = services.ListJobTemplatesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_job_templates(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_job_templates_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.TranscoderServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_job_templates._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_job_templates_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.TranscoderServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TranscoderServiceRestInterceptor())
    client = TranscoderServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TranscoderServiceRestInterceptor, 'post_list_job_templates') as post, mock.patch.object(transports.TranscoderServiceRestInterceptor, 'pre_list_job_templates') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = services.ListJobTemplatesRequest.pb(services.ListJobTemplatesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = services.ListJobTemplatesResponse.to_json(services.ListJobTemplatesResponse())
        request = services.ListJobTemplatesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = services.ListJobTemplatesResponse()
        client.list_job_templates(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_job_templates_rest_bad_request(transport: str='rest', request_type=services.ListJobTemplatesRequest):
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_job_templates(request)

def test_list_job_templates_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = services.ListJobTemplatesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = services.ListJobTemplatesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_job_templates(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/jobTemplates' % client.transport._host, args[1])

def test_list_job_templates_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_job_templates(services.ListJobTemplatesRequest(), parent='parent_value')

def test_list_job_templates_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (services.ListJobTemplatesResponse(job_templates=[resources.JobTemplate(), resources.JobTemplate(), resources.JobTemplate()], next_page_token='abc'), services.ListJobTemplatesResponse(job_templates=[], next_page_token='def'), services.ListJobTemplatesResponse(job_templates=[resources.JobTemplate()], next_page_token='ghi'), services.ListJobTemplatesResponse(job_templates=[resources.JobTemplate(), resources.JobTemplate()]))
        response = response + response
        response = tuple((services.ListJobTemplatesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_job_templates(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.JobTemplate) for i in results))
        pages = list(client.list_job_templates(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [services.GetJobTemplateRequest, dict])
def test_get_job_template_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/jobTemplates/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.JobTemplate(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.JobTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_job_template(request)
    assert isinstance(response, resources.JobTemplate)
    assert response.name == 'name_value'

def test_get_job_template_rest_required_fields(request_type=services.GetJobTemplateRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.TranscoderServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_job_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_job_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.JobTemplate()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.JobTemplate.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_job_template(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_job_template_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TranscoderServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_job_template._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_job_template_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.TranscoderServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TranscoderServiceRestInterceptor())
    client = TranscoderServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TranscoderServiceRestInterceptor, 'post_get_job_template') as post, mock.patch.object(transports.TranscoderServiceRestInterceptor, 'pre_get_job_template') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = services.GetJobTemplateRequest.pb(services.GetJobTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.JobTemplate.to_json(resources.JobTemplate())
        request = services.GetJobTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.JobTemplate()
        client.get_job_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_job_template_rest_bad_request(transport: str='rest', request_type=services.GetJobTemplateRequest):
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/jobTemplates/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_job_template(request)

def test_get_job_template_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.JobTemplate()
        sample_request = {'name': 'projects/sample1/locations/sample2/jobTemplates/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.JobTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_job_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/jobTemplates/*}' % client.transport._host, args[1])

def test_get_job_template_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_job_template(services.GetJobTemplateRequest(), name='name_value')

def test_get_job_template_rest_error():
    if False:
        i = 10
        return i + 15
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [services.DeleteJobTemplateRequest, dict])
def test_delete_job_template_rest(request_type):
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/jobTemplates/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_job_template(request)
    assert response is None

def test_delete_job_template_rest_required_fields(request_type=services.DeleteJobTemplateRequest):
    if False:
        return 10
    transport_class = transports.TranscoderServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_job_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_job_template._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_job_template(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_job_template_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.TranscoderServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_job_template._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_job_template_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.TranscoderServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TranscoderServiceRestInterceptor())
    client = TranscoderServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TranscoderServiceRestInterceptor, 'pre_delete_job_template') as pre:
        pre.assert_not_called()
        pb_message = services.DeleteJobTemplateRequest.pb(services.DeleteJobTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = services.DeleteJobTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_job_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_job_template_rest_bad_request(transport: str='rest', request_type=services.DeleteJobTemplateRequest):
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/jobTemplates/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_job_template(request)

def test_delete_job_template_rest_flattened():
    if False:
        while True:
            i = 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/jobTemplates/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_job_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/jobTemplates/*}' % client.transport._host, args[1])

def test_delete_job_template_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_job_template(services.DeleteJobTemplateRequest(), name='name_value')

def test_delete_job_template_rest_error():
    if False:
        return 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.TranscoderServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.TranscoderServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TranscoderServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.TranscoderServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = TranscoderServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = TranscoderServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.TranscoderServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TranscoderServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.TranscoderServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = TranscoderServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.TranscoderServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.TranscoderServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.TranscoderServiceGrpcTransport, transports.TranscoderServiceGrpcAsyncIOTransport, transports.TranscoderServiceRestTransport])
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
        i = 10
        return i + 15
    transport = TranscoderServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        i = 10
        return i + 15
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.TranscoderServiceGrpcTransport)

def test_transcoder_service_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.TranscoderServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_transcoder_service_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.video.transcoder_v1.services.transcoder_service.transports.TranscoderServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.TranscoderServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_job', 'list_jobs', 'get_job', 'delete_job', 'create_job_template', 'list_job_templates', 'get_job_template', 'delete_job_template')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_transcoder_service_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.video.transcoder_v1.services.transcoder_service.transports.TranscoderServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.TranscoderServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_transcoder_service_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.video.transcoder_v1.services.transcoder_service.transports.TranscoderServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.TranscoderServiceTransport()
        adc.assert_called_once()

def test_transcoder_service_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        TranscoderServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.TranscoderServiceGrpcTransport, transports.TranscoderServiceGrpcAsyncIOTransport])
def test_transcoder_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.TranscoderServiceGrpcTransport, transports.TranscoderServiceGrpcAsyncIOTransport, transports.TranscoderServiceRestTransport])
def test_transcoder_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.TranscoderServiceGrpcTransport, grpc_helpers), (transports.TranscoderServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_transcoder_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('transcoder.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='transcoder.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.TranscoderServiceGrpcTransport, transports.TranscoderServiceGrpcAsyncIOTransport])
def test_transcoder_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_transcoder_service_http_transport_client_cert_source_for_mtls():
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.TranscoderServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_transcoder_service_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='transcoder.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('transcoder.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://transcoder.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_transcoder_service_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='transcoder.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('transcoder.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://transcoder.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_transcoder_service_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = TranscoderServiceClient(credentials=creds1, transport=transport_name)
    client2 = TranscoderServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_job._session
    session2 = client2.transport.create_job._session
    assert session1 != session2
    session1 = client1.transport.list_jobs._session
    session2 = client2.transport.list_jobs._session
    assert session1 != session2
    session1 = client1.transport.get_job._session
    session2 = client2.transport.get_job._session
    assert session1 != session2
    session1 = client1.transport.delete_job._session
    session2 = client2.transport.delete_job._session
    assert session1 != session2
    session1 = client1.transport.create_job_template._session
    session2 = client2.transport.create_job_template._session
    assert session1 != session2
    session1 = client1.transport.list_job_templates._session
    session2 = client2.transport.list_job_templates._session
    assert session1 != session2
    session1 = client1.transport.get_job_template._session
    session2 = client2.transport.get_job_template._session
    assert session1 != session2
    session1 = client1.transport.delete_job_template._session
    session2 = client2.transport.delete_job_template._session
    assert session1 != session2

def test_transcoder_service_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.TranscoderServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_transcoder_service_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.TranscoderServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.TranscoderServiceGrpcTransport, transports.TranscoderServiceGrpcAsyncIOTransport])
def test_transcoder_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.TranscoderServiceGrpcTransport, transports.TranscoderServiceGrpcAsyncIOTransport])
def test_transcoder_service_transport_channel_mtls_with_adc(transport_class):
    if False:
        return 10
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
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    job = 'whelk'
    expected = 'projects/{project}/locations/{location}/jobs/{job}'.format(project=project, location=location, job=job)
    actual = TranscoderServiceClient.job_path(project, location, job)
    assert expected == actual

def test_parse_job_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'octopus', 'location': 'oyster', 'job': 'nudibranch'}
    path = TranscoderServiceClient.job_path(**expected)
    actual = TranscoderServiceClient.parse_job_path(path)
    assert expected == actual

def test_job_template_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    location = 'mussel'
    job_template = 'winkle'
    expected = 'projects/{project}/locations/{location}/jobTemplates/{job_template}'.format(project=project, location=location, job_template=job_template)
    actual = TranscoderServiceClient.job_template_path(project, location, job_template)
    assert expected == actual

def test_parse_job_template_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nautilus', 'location': 'scallop', 'job_template': 'abalone'}
    path = TranscoderServiceClient.job_template_path(**expected)
    actual = TranscoderServiceClient.parse_job_template_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = TranscoderServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'clam'}
    path = TranscoderServiceClient.common_billing_account_path(**expected)
    actual = TranscoderServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = TranscoderServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'octopus'}
    path = TranscoderServiceClient.common_folder_path(**expected)
    actual = TranscoderServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = TranscoderServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'nudibranch'}
    path = TranscoderServiceClient.common_organization_path(**expected)
    actual = TranscoderServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = TranscoderServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'mussel'}
    path = TranscoderServiceClient.common_project_path(**expected)
    actual = TranscoderServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = TranscoderServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = TranscoderServiceClient.common_location_path(**expected)
    actual = TranscoderServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.TranscoderServiceTransport, '_prep_wrapped_messages') as prep:
        client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.TranscoderServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = TranscoderServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = TranscoderServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
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
        client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = TranscoderServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(TranscoderServiceClient, transports.TranscoderServiceGrpcTransport), (TranscoderServiceAsyncClient, transports.TranscoderServiceGrpcAsyncIOTransport)])
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