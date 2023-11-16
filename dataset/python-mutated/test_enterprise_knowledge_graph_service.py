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
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import any_pb2
from google.protobuf import json_format
from google.protobuf import struct_pb2
from google.protobuf import timestamp_pb2
from google.protobuf import wrappers_pb2
from google.rpc import status_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.enterpriseknowledgegraph_v1.services.enterprise_knowledge_graph_service import EnterpriseKnowledgeGraphServiceAsyncClient, EnterpriseKnowledgeGraphServiceClient, pagers, transports
from google.cloud.enterpriseknowledgegraph_v1.types import job_state, service

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
        for i in range(10):
            print('nop')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert EnterpriseKnowledgeGraphServiceClient._get_default_mtls_endpoint(None) is None
    assert EnterpriseKnowledgeGraphServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert EnterpriseKnowledgeGraphServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert EnterpriseKnowledgeGraphServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert EnterpriseKnowledgeGraphServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert EnterpriseKnowledgeGraphServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(EnterpriseKnowledgeGraphServiceClient, 'grpc'), (EnterpriseKnowledgeGraphServiceAsyncClient, 'grpc_asyncio'), (EnterpriseKnowledgeGraphServiceClient, 'rest')])
def test_enterprise_knowledge_graph_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('enterpriseknowledgegraph.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://enterpriseknowledgegraph.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.EnterpriseKnowledgeGraphServiceGrpcTransport, 'grpc'), (transports.EnterpriseKnowledgeGraphServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.EnterpriseKnowledgeGraphServiceRestTransport, 'rest')])
def test_enterprise_knowledge_graph_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(EnterpriseKnowledgeGraphServiceClient, 'grpc'), (EnterpriseKnowledgeGraphServiceAsyncClient, 'grpc_asyncio'), (EnterpriseKnowledgeGraphServiceClient, 'rest')])
def test_enterprise_knowledge_graph_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('enterpriseknowledgegraph.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://enterpriseknowledgegraph.googleapis.com')

def test_enterprise_knowledge_graph_service_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = EnterpriseKnowledgeGraphServiceClient.get_transport_class()
    available_transports = [transports.EnterpriseKnowledgeGraphServiceGrpcTransport, transports.EnterpriseKnowledgeGraphServiceRestTransport]
    assert transport in available_transports
    transport = EnterpriseKnowledgeGraphServiceClient.get_transport_class('grpc')
    assert transport == transports.EnterpriseKnowledgeGraphServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(EnterpriseKnowledgeGraphServiceClient, transports.EnterpriseKnowledgeGraphServiceGrpcTransport, 'grpc'), (EnterpriseKnowledgeGraphServiceAsyncClient, transports.EnterpriseKnowledgeGraphServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (EnterpriseKnowledgeGraphServiceClient, transports.EnterpriseKnowledgeGraphServiceRestTransport, 'rest')])
@mock.patch.object(EnterpriseKnowledgeGraphServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EnterpriseKnowledgeGraphServiceClient))
@mock.patch.object(EnterpriseKnowledgeGraphServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EnterpriseKnowledgeGraphServiceAsyncClient))
def test_enterprise_knowledge_graph_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(EnterpriseKnowledgeGraphServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(EnterpriseKnowledgeGraphServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(EnterpriseKnowledgeGraphServiceClient, transports.EnterpriseKnowledgeGraphServiceGrpcTransport, 'grpc', 'true'), (EnterpriseKnowledgeGraphServiceAsyncClient, transports.EnterpriseKnowledgeGraphServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (EnterpriseKnowledgeGraphServiceClient, transports.EnterpriseKnowledgeGraphServiceGrpcTransport, 'grpc', 'false'), (EnterpriseKnowledgeGraphServiceAsyncClient, transports.EnterpriseKnowledgeGraphServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (EnterpriseKnowledgeGraphServiceClient, transports.EnterpriseKnowledgeGraphServiceRestTransport, 'rest', 'true'), (EnterpriseKnowledgeGraphServiceClient, transports.EnterpriseKnowledgeGraphServiceRestTransport, 'rest', 'false')])
@mock.patch.object(EnterpriseKnowledgeGraphServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EnterpriseKnowledgeGraphServiceClient))
@mock.patch.object(EnterpriseKnowledgeGraphServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EnterpriseKnowledgeGraphServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_enterprise_knowledge_graph_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [EnterpriseKnowledgeGraphServiceClient, EnterpriseKnowledgeGraphServiceAsyncClient])
@mock.patch.object(EnterpriseKnowledgeGraphServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EnterpriseKnowledgeGraphServiceClient))
@mock.patch.object(EnterpriseKnowledgeGraphServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EnterpriseKnowledgeGraphServiceAsyncClient))
def test_enterprise_knowledge_graph_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(EnterpriseKnowledgeGraphServiceClient, transports.EnterpriseKnowledgeGraphServiceGrpcTransport, 'grpc'), (EnterpriseKnowledgeGraphServiceAsyncClient, transports.EnterpriseKnowledgeGraphServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (EnterpriseKnowledgeGraphServiceClient, transports.EnterpriseKnowledgeGraphServiceRestTransport, 'rest')])
def test_enterprise_knowledge_graph_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(EnterpriseKnowledgeGraphServiceClient, transports.EnterpriseKnowledgeGraphServiceGrpcTransport, 'grpc', grpc_helpers), (EnterpriseKnowledgeGraphServiceAsyncClient, transports.EnterpriseKnowledgeGraphServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (EnterpriseKnowledgeGraphServiceClient, transports.EnterpriseKnowledgeGraphServiceRestTransport, 'rest', None)])
def test_enterprise_knowledge_graph_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_enterprise_knowledge_graph_service_client_client_options_from_dict():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.enterpriseknowledgegraph_v1.services.enterprise_knowledge_graph_service.transports.EnterpriseKnowledgeGraphServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = EnterpriseKnowledgeGraphServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(EnterpriseKnowledgeGraphServiceClient, transports.EnterpriseKnowledgeGraphServiceGrpcTransport, 'grpc', grpc_helpers), (EnterpriseKnowledgeGraphServiceAsyncClient, transports.EnterpriseKnowledgeGraphServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_enterprise_knowledge_graph_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('enterpriseknowledgegraph.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='enterpriseknowledgegraph.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [service.CreateEntityReconciliationJobRequest, dict])
def test_create_entity_reconciliation_job(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_entity_reconciliation_job), '__call__') as call:
        call.return_value = service.EntityReconciliationJob(name='name_value', state=job_state.JobState.JOB_STATE_PENDING)
        response = client.create_entity_reconciliation_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateEntityReconciliationJobRequest()
    assert isinstance(response, service.EntityReconciliationJob)
    assert response.name == 'name_value'
    assert response.state == job_state.JobState.JOB_STATE_PENDING

def test_create_entity_reconciliation_job_empty_call():
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_entity_reconciliation_job), '__call__') as call:
        client.create_entity_reconciliation_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateEntityReconciliationJobRequest()

@pytest.mark.asyncio
async def test_create_entity_reconciliation_job_async(transport: str='grpc_asyncio', request_type=service.CreateEntityReconciliationJobRequest):
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_entity_reconciliation_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.EntityReconciliationJob(name='name_value', state=job_state.JobState.JOB_STATE_PENDING))
        response = await client.create_entity_reconciliation_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateEntityReconciliationJobRequest()
    assert isinstance(response, service.EntityReconciliationJob)
    assert response.name == 'name_value'
    assert response.state == job_state.JobState.JOB_STATE_PENDING

@pytest.mark.asyncio
async def test_create_entity_reconciliation_job_async_from_dict():
    await test_create_entity_reconciliation_job_async(request_type=dict)

def test_create_entity_reconciliation_job_field_headers():
    if False:
        while True:
            i = 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateEntityReconciliationJobRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_entity_reconciliation_job), '__call__') as call:
        call.return_value = service.EntityReconciliationJob()
        client.create_entity_reconciliation_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_entity_reconciliation_job_field_headers_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateEntityReconciliationJobRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_entity_reconciliation_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.EntityReconciliationJob())
        await client.create_entity_reconciliation_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_entity_reconciliation_job_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_entity_reconciliation_job), '__call__') as call:
        call.return_value = service.EntityReconciliationJob()
        client.create_entity_reconciliation_job(parent='parent_value', entity_reconciliation_job=service.EntityReconciliationJob(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].entity_reconciliation_job
        mock_val = service.EntityReconciliationJob(name='name_value')
        assert arg == mock_val

def test_create_entity_reconciliation_job_flattened_error():
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_entity_reconciliation_job(service.CreateEntityReconciliationJobRequest(), parent='parent_value', entity_reconciliation_job=service.EntityReconciliationJob(name='name_value'))

@pytest.mark.asyncio
async def test_create_entity_reconciliation_job_flattened_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_entity_reconciliation_job), '__call__') as call:
        call.return_value = service.EntityReconciliationJob()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.EntityReconciliationJob())
        response = await client.create_entity_reconciliation_job(parent='parent_value', entity_reconciliation_job=service.EntityReconciliationJob(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].entity_reconciliation_job
        mock_val = service.EntityReconciliationJob(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_entity_reconciliation_job_flattened_error_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_entity_reconciliation_job(service.CreateEntityReconciliationJobRequest(), parent='parent_value', entity_reconciliation_job=service.EntityReconciliationJob(name='name_value'))

@pytest.mark.parametrize('request_type', [service.GetEntityReconciliationJobRequest, dict])
def test_get_entity_reconciliation_job(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_entity_reconciliation_job), '__call__') as call:
        call.return_value = service.EntityReconciliationJob(name='name_value', state=job_state.JobState.JOB_STATE_PENDING)
        response = client.get_entity_reconciliation_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetEntityReconciliationJobRequest()
    assert isinstance(response, service.EntityReconciliationJob)
    assert response.name == 'name_value'
    assert response.state == job_state.JobState.JOB_STATE_PENDING

def test_get_entity_reconciliation_job_empty_call():
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_entity_reconciliation_job), '__call__') as call:
        client.get_entity_reconciliation_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetEntityReconciliationJobRequest()

@pytest.mark.asyncio
async def test_get_entity_reconciliation_job_async(transport: str='grpc_asyncio', request_type=service.GetEntityReconciliationJobRequest):
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_entity_reconciliation_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.EntityReconciliationJob(name='name_value', state=job_state.JobState.JOB_STATE_PENDING))
        response = await client.get_entity_reconciliation_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetEntityReconciliationJobRequest()
    assert isinstance(response, service.EntityReconciliationJob)
    assert response.name == 'name_value'
    assert response.state == job_state.JobState.JOB_STATE_PENDING

@pytest.mark.asyncio
async def test_get_entity_reconciliation_job_async_from_dict():
    await test_get_entity_reconciliation_job_async(request_type=dict)

def test_get_entity_reconciliation_job_field_headers():
    if False:
        while True:
            i = 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetEntityReconciliationJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_entity_reconciliation_job), '__call__') as call:
        call.return_value = service.EntityReconciliationJob()
        client.get_entity_reconciliation_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_entity_reconciliation_job_field_headers_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetEntityReconciliationJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_entity_reconciliation_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.EntityReconciliationJob())
        await client.get_entity_reconciliation_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_entity_reconciliation_job_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_entity_reconciliation_job), '__call__') as call:
        call.return_value = service.EntityReconciliationJob()
        client.get_entity_reconciliation_job(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_entity_reconciliation_job_flattened_error():
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_entity_reconciliation_job(service.GetEntityReconciliationJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_entity_reconciliation_job_flattened_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_entity_reconciliation_job), '__call__') as call:
        call.return_value = service.EntityReconciliationJob()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.EntityReconciliationJob())
        response = await client.get_entity_reconciliation_job(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_entity_reconciliation_job_flattened_error_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_entity_reconciliation_job(service.GetEntityReconciliationJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListEntityReconciliationJobsRequest, dict])
def test_list_entity_reconciliation_jobs(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_entity_reconciliation_jobs), '__call__') as call:
        call.return_value = service.ListEntityReconciliationJobsResponse(next_page_token='next_page_token_value')
        response = client.list_entity_reconciliation_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListEntityReconciliationJobsRequest()
    assert isinstance(response, pagers.ListEntityReconciliationJobsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_entity_reconciliation_jobs_empty_call():
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_entity_reconciliation_jobs), '__call__') as call:
        client.list_entity_reconciliation_jobs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListEntityReconciliationJobsRequest()

@pytest.mark.asyncio
async def test_list_entity_reconciliation_jobs_async(transport: str='grpc_asyncio', request_type=service.ListEntityReconciliationJobsRequest):
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_entity_reconciliation_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListEntityReconciliationJobsResponse(next_page_token='next_page_token_value'))
        response = await client.list_entity_reconciliation_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListEntityReconciliationJobsRequest()
    assert isinstance(response, pagers.ListEntityReconciliationJobsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_entity_reconciliation_jobs_async_from_dict():
    await test_list_entity_reconciliation_jobs_async(request_type=dict)

def test_list_entity_reconciliation_jobs_field_headers():
    if False:
        while True:
            i = 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListEntityReconciliationJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_entity_reconciliation_jobs), '__call__') as call:
        call.return_value = service.ListEntityReconciliationJobsResponse()
        client.list_entity_reconciliation_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_entity_reconciliation_jobs_field_headers_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListEntityReconciliationJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_entity_reconciliation_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListEntityReconciliationJobsResponse())
        await client.list_entity_reconciliation_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_entity_reconciliation_jobs_flattened():
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_entity_reconciliation_jobs), '__call__') as call:
        call.return_value = service.ListEntityReconciliationJobsResponse()
        client.list_entity_reconciliation_jobs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_entity_reconciliation_jobs_flattened_error():
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_entity_reconciliation_jobs(service.ListEntityReconciliationJobsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_entity_reconciliation_jobs_flattened_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_entity_reconciliation_jobs), '__call__') as call:
        call.return_value = service.ListEntityReconciliationJobsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListEntityReconciliationJobsResponse())
        response = await client.list_entity_reconciliation_jobs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_entity_reconciliation_jobs_flattened_error_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_entity_reconciliation_jobs(service.ListEntityReconciliationJobsRequest(), parent='parent_value')

def test_list_entity_reconciliation_jobs_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_entity_reconciliation_jobs), '__call__') as call:
        call.side_effect = (service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[service.EntityReconciliationJob(), service.EntityReconciliationJob(), service.EntityReconciliationJob()], next_page_token='abc'), service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[], next_page_token='def'), service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[service.EntityReconciliationJob()], next_page_token='ghi'), service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[service.EntityReconciliationJob(), service.EntityReconciliationJob()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_entity_reconciliation_jobs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service.EntityReconciliationJob) for i in results))

def test_list_entity_reconciliation_jobs_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_entity_reconciliation_jobs), '__call__') as call:
        call.side_effect = (service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[service.EntityReconciliationJob(), service.EntityReconciliationJob(), service.EntityReconciliationJob()], next_page_token='abc'), service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[], next_page_token='def'), service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[service.EntityReconciliationJob()], next_page_token='ghi'), service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[service.EntityReconciliationJob(), service.EntityReconciliationJob()]), RuntimeError)
        pages = list(client.list_entity_reconciliation_jobs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_entity_reconciliation_jobs_async_pager():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_entity_reconciliation_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[service.EntityReconciliationJob(), service.EntityReconciliationJob(), service.EntityReconciliationJob()], next_page_token='abc'), service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[], next_page_token='def'), service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[service.EntityReconciliationJob()], next_page_token='ghi'), service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[service.EntityReconciliationJob(), service.EntityReconciliationJob()]), RuntimeError)
        async_pager = await client.list_entity_reconciliation_jobs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, service.EntityReconciliationJob) for i in responses))

@pytest.mark.asyncio
async def test_list_entity_reconciliation_jobs_async_pages():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_entity_reconciliation_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[service.EntityReconciliationJob(), service.EntityReconciliationJob(), service.EntityReconciliationJob()], next_page_token='abc'), service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[], next_page_token='def'), service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[service.EntityReconciliationJob()], next_page_token='ghi'), service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[service.EntityReconciliationJob(), service.EntityReconciliationJob()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_entity_reconciliation_jobs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.CancelEntityReconciliationJobRequest, dict])
def test_cancel_entity_reconciliation_job(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.cancel_entity_reconciliation_job), '__call__') as call:
        call.return_value = None
        response = client.cancel_entity_reconciliation_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CancelEntityReconciliationJobRequest()
    assert response is None

def test_cancel_entity_reconciliation_job_empty_call():
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.cancel_entity_reconciliation_job), '__call__') as call:
        client.cancel_entity_reconciliation_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CancelEntityReconciliationJobRequest()

@pytest.mark.asyncio
async def test_cancel_entity_reconciliation_job_async(transport: str='grpc_asyncio', request_type=service.CancelEntityReconciliationJobRequest):
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.cancel_entity_reconciliation_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_entity_reconciliation_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CancelEntityReconciliationJobRequest()
    assert response is None

@pytest.mark.asyncio
async def test_cancel_entity_reconciliation_job_async_from_dict():
    await test_cancel_entity_reconciliation_job_async(request_type=dict)

def test_cancel_entity_reconciliation_job_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CancelEntityReconciliationJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.cancel_entity_reconciliation_job), '__call__') as call:
        call.return_value = None
        client.cancel_entity_reconciliation_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_cancel_entity_reconciliation_job_field_headers_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CancelEntityReconciliationJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.cancel_entity_reconciliation_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.cancel_entity_reconciliation_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_cancel_entity_reconciliation_job_flattened():
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_entity_reconciliation_job), '__call__') as call:
        call.return_value = None
        client.cancel_entity_reconciliation_job(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_cancel_entity_reconciliation_job_flattened_error():
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.cancel_entity_reconciliation_job(service.CancelEntityReconciliationJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_cancel_entity_reconciliation_job_flattened_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_entity_reconciliation_job), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_entity_reconciliation_job(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_cancel_entity_reconciliation_job_flattened_error_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.cancel_entity_reconciliation_job(service.CancelEntityReconciliationJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.DeleteEntityReconciliationJobRequest, dict])
def test_delete_entity_reconciliation_job(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_entity_reconciliation_job), '__call__') as call:
        call.return_value = None
        response = client.delete_entity_reconciliation_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteEntityReconciliationJobRequest()
    assert response is None

def test_delete_entity_reconciliation_job_empty_call():
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_entity_reconciliation_job), '__call__') as call:
        client.delete_entity_reconciliation_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteEntityReconciliationJobRequest()

@pytest.mark.asyncio
async def test_delete_entity_reconciliation_job_async(transport: str='grpc_asyncio', request_type=service.DeleteEntityReconciliationJobRequest):
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_entity_reconciliation_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_entity_reconciliation_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteEntityReconciliationJobRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_entity_reconciliation_job_async_from_dict():
    await test_delete_entity_reconciliation_job_async(request_type=dict)

def test_delete_entity_reconciliation_job_field_headers():
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteEntityReconciliationJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_entity_reconciliation_job), '__call__') as call:
        call.return_value = None
        client.delete_entity_reconciliation_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_entity_reconciliation_job_field_headers_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteEntityReconciliationJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_entity_reconciliation_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_entity_reconciliation_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_entity_reconciliation_job_flattened():
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_entity_reconciliation_job), '__call__') as call:
        call.return_value = None
        client.delete_entity_reconciliation_job(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_entity_reconciliation_job_flattened_error():
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_entity_reconciliation_job(service.DeleteEntityReconciliationJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_entity_reconciliation_job_flattened_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_entity_reconciliation_job), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_entity_reconciliation_job(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_entity_reconciliation_job_flattened_error_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_entity_reconciliation_job(service.DeleteEntityReconciliationJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.LookupRequest, dict])
def test_lookup(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.lookup), '__call__') as call:
        call.return_value = service.LookupResponse()
        response = client.lookup(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.LookupRequest()
    assert isinstance(response, service.LookupResponse)

def test_lookup_empty_call():
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.lookup), '__call__') as call:
        client.lookup()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.LookupRequest()

@pytest.mark.asyncio
async def test_lookup_async(transport: str='grpc_asyncio', request_type=service.LookupRequest):
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.lookup), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.LookupResponse())
        response = await client.lookup(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.LookupRequest()
    assert isinstance(response, service.LookupResponse)

@pytest.mark.asyncio
async def test_lookup_async_from_dict():
    await test_lookup_async(request_type=dict)

def test_lookup_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.LookupRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.lookup), '__call__') as call:
        call.return_value = service.LookupResponse()
        client.lookup(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_lookup_field_headers_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.LookupRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.lookup), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.LookupResponse())
        await client.lookup(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_lookup_flattened():
    if False:
        while True:
            i = 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.lookup), '__call__') as call:
        call.return_value = service.LookupResponse()
        client.lookup(parent='parent_value', ids=['ids_value'])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].ids
        mock_val = ['ids_value']
        assert arg == mock_val

def test_lookup_flattened_error():
    if False:
        while True:
            i = 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.lookup(service.LookupRequest(), parent='parent_value', ids=['ids_value'])

@pytest.mark.asyncio
async def test_lookup_flattened_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.lookup), '__call__') as call:
        call.return_value = service.LookupResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.LookupResponse())
        response = await client.lookup(parent='parent_value', ids=['ids_value'])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].ids
        mock_val = ['ids_value']
        assert arg == mock_val

@pytest.mark.asyncio
async def test_lookup_flattened_error_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.lookup(service.LookupRequest(), parent='parent_value', ids=['ids_value'])

@pytest.mark.parametrize('request_type', [service.SearchRequest, dict])
def test_search(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search), '__call__') as call:
        call.return_value = service.SearchResponse()
        response = client.search(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.SearchRequest()
    assert isinstance(response, service.SearchResponse)

def test_search_empty_call():
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search), '__call__') as call:
        client.search()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.SearchRequest()

@pytest.mark.asyncio
async def test_search_async(transport: str='grpc_asyncio', request_type=service.SearchRequest):
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.SearchResponse())
        response = await client.search(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.SearchRequest()
    assert isinstance(response, service.SearchResponse)

@pytest.mark.asyncio
async def test_search_async_from_dict():
    await test_search_async(request_type=dict)

def test_search_field_headers():
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.SearchRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.search), '__call__') as call:
        call.return_value = service.SearchResponse()
        client.search(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_search_field_headers_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.SearchRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.search), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.SearchResponse())
        await client.search(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_search_flattened():
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search), '__call__') as call:
        call.return_value = service.SearchResponse()
        client.search(parent='parent_value', query='query_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].query
        mock_val = 'query_value'
        assert arg == mock_val

def test_search_flattened_error():
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.search(service.SearchRequest(), parent='parent_value', query='query_value')

@pytest.mark.asyncio
async def test_search_flattened_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search), '__call__') as call:
        call.return_value = service.SearchResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.SearchResponse())
        response = await client.search(parent='parent_value', query='query_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].query
        mock_val = 'query_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_search_flattened_error_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.search(service.SearchRequest(), parent='parent_value', query='query_value')

@pytest.mark.parametrize('request_type', [service.LookupPublicKgRequest, dict])
def test_lookup_public_kg(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.lookup_public_kg), '__call__') as call:
        call.return_value = service.LookupPublicKgResponse()
        response = client.lookup_public_kg(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.LookupPublicKgRequest()
    assert isinstance(response, service.LookupPublicKgResponse)

def test_lookup_public_kg_empty_call():
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.lookup_public_kg), '__call__') as call:
        client.lookup_public_kg()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.LookupPublicKgRequest()

@pytest.mark.asyncio
async def test_lookup_public_kg_async(transport: str='grpc_asyncio', request_type=service.LookupPublicKgRequest):
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.lookup_public_kg), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.LookupPublicKgResponse())
        response = await client.lookup_public_kg(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.LookupPublicKgRequest()
    assert isinstance(response, service.LookupPublicKgResponse)

@pytest.mark.asyncio
async def test_lookup_public_kg_async_from_dict():
    await test_lookup_public_kg_async(request_type=dict)

def test_lookup_public_kg_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.LookupPublicKgRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.lookup_public_kg), '__call__') as call:
        call.return_value = service.LookupPublicKgResponse()
        client.lookup_public_kg(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_lookup_public_kg_field_headers_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.LookupPublicKgRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.lookup_public_kg), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.LookupPublicKgResponse())
        await client.lookup_public_kg(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_lookup_public_kg_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.lookup_public_kg), '__call__') as call:
        call.return_value = service.LookupPublicKgResponse()
        client.lookup_public_kg(parent='parent_value', ids=['ids_value'])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].ids
        mock_val = ['ids_value']
        assert arg == mock_val

def test_lookup_public_kg_flattened_error():
    if False:
        while True:
            i = 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.lookup_public_kg(service.LookupPublicKgRequest(), parent='parent_value', ids=['ids_value'])

@pytest.mark.asyncio
async def test_lookup_public_kg_flattened_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.lookup_public_kg), '__call__') as call:
        call.return_value = service.LookupPublicKgResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.LookupPublicKgResponse())
        response = await client.lookup_public_kg(parent='parent_value', ids=['ids_value'])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].ids
        mock_val = ['ids_value']
        assert arg == mock_val

@pytest.mark.asyncio
async def test_lookup_public_kg_flattened_error_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.lookup_public_kg(service.LookupPublicKgRequest(), parent='parent_value', ids=['ids_value'])

@pytest.mark.parametrize('request_type', [service.SearchPublicKgRequest, dict])
def test_search_public_kg(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_public_kg), '__call__') as call:
        call.return_value = service.SearchPublicKgResponse()
        response = client.search_public_kg(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.SearchPublicKgRequest()
    assert isinstance(response, service.SearchPublicKgResponse)

def test_search_public_kg_empty_call():
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search_public_kg), '__call__') as call:
        client.search_public_kg()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.SearchPublicKgRequest()

@pytest.mark.asyncio
async def test_search_public_kg_async(transport: str='grpc_asyncio', request_type=service.SearchPublicKgRequest):
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_public_kg), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.SearchPublicKgResponse())
        response = await client.search_public_kg(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.SearchPublicKgRequest()
    assert isinstance(response, service.SearchPublicKgResponse)

@pytest.mark.asyncio
async def test_search_public_kg_async_from_dict():
    await test_search_public_kg_async(request_type=dict)

def test_search_public_kg_field_headers():
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.SearchPublicKgRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.search_public_kg), '__call__') as call:
        call.return_value = service.SearchPublicKgResponse()
        client.search_public_kg(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_search_public_kg_field_headers_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.SearchPublicKgRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.search_public_kg), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.SearchPublicKgResponse())
        await client.search_public_kg(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_search_public_kg_flattened():
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_public_kg), '__call__') as call:
        call.return_value = service.SearchPublicKgResponse()
        client.search_public_kg(parent='parent_value', query='query_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].query
        mock_val = 'query_value'
        assert arg == mock_val

def test_search_public_kg_flattened_error():
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.search_public_kg(service.SearchPublicKgRequest(), parent='parent_value', query='query_value')

@pytest.mark.asyncio
async def test_search_public_kg_flattened_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.search_public_kg), '__call__') as call:
        call.return_value = service.SearchPublicKgResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.SearchPublicKgResponse())
        response = await client.search_public_kg(parent='parent_value', query='query_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].query
        mock_val = 'query_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_search_public_kg_flattened_error_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.search_public_kg(service.SearchPublicKgRequest(), parent='parent_value', query='query_value')

@pytest.mark.parametrize('request_type', [service.CreateEntityReconciliationJobRequest, dict])
def test_create_entity_reconciliation_job_rest(request_type):
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['entity_reconciliation_job'] = {'name': 'name_value', 'input_config': {'bigquery_input_configs': [{'bigquery_table': 'bigquery_table_value', 'gcs_uri': 'gcs_uri_value'}], 'entity_type': 1, 'previous_result_bigquery_table': 'previous_result_bigquery_table_value'}, 'output_config': {'bigquery_dataset': 'bigquery_dataset_value'}, 'state': 9, 'error': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}, 'create_time': {'seconds': 751, 'nanos': 543}, 'end_time': {}, 'update_time': {}, 'recon_config': {'connected_components_config': {'weight_threshold': 0.1716}, 'affinity_clustering_config': {'compression_round_count': 2497}, 'options': {'enable_geocoding_separation': True}, 'model_config': {'model_name': 'model_name_value', 'version_tag': 'version_tag_value'}}}
    test_field = service.CreateEntityReconciliationJobRequest.meta.fields['entity_reconciliation_job']

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
    for (field, value) in request_init['entity_reconciliation_job'].items():
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
                for i in range(0, len(request_init['entity_reconciliation_job'][field])):
                    del request_init['entity_reconciliation_job'][field][i][subfield]
            else:
                del request_init['entity_reconciliation_job'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.EntityReconciliationJob(name='name_value', state=job_state.JobState.JOB_STATE_PENDING)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.EntityReconciliationJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_entity_reconciliation_job(request)
    assert isinstance(response, service.EntityReconciliationJob)
    assert response.name == 'name_value'
    assert response.state == job_state.JobState.JOB_STATE_PENDING

def test_create_entity_reconciliation_job_rest_required_fields(request_type=service.CreateEntityReconciliationJobRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.EnterpriseKnowledgeGraphServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_entity_reconciliation_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_entity_reconciliation_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.EntityReconciliationJob()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.EntityReconciliationJob.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_entity_reconciliation_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_entity_reconciliation_job_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_entity_reconciliation_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'entityReconciliationJob'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_entity_reconciliation_job_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EnterpriseKnowledgeGraphServiceRestInterceptor())
    client = EnterpriseKnowledgeGraphServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EnterpriseKnowledgeGraphServiceRestInterceptor, 'post_create_entity_reconciliation_job') as post, mock.patch.object(transports.EnterpriseKnowledgeGraphServiceRestInterceptor, 'pre_create_entity_reconciliation_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateEntityReconciliationJobRequest.pb(service.CreateEntityReconciliationJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.EntityReconciliationJob.to_json(service.EntityReconciliationJob())
        request = service.CreateEntityReconciliationJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.EntityReconciliationJob()
        client.create_entity_reconciliation_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_entity_reconciliation_job_rest_bad_request(transport: str='rest', request_type=service.CreateEntityReconciliationJobRequest):
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_entity_reconciliation_job(request)

def test_create_entity_reconciliation_job_rest_flattened():
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.EntityReconciliationJob()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', entity_reconciliation_job=service.EntityReconciliationJob(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.EntityReconciliationJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_entity_reconciliation_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/entityReconciliationJobs' % client.transport._host, args[1])

def test_create_entity_reconciliation_job_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_entity_reconciliation_job(service.CreateEntityReconciliationJobRequest(), parent='parent_value', entity_reconciliation_job=service.EntityReconciliationJob(name='name_value'))

def test_create_entity_reconciliation_job_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.GetEntityReconciliationJobRequest, dict])
def test_get_entity_reconciliation_job_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/entityReconciliationJobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.EntityReconciliationJob(name='name_value', state=job_state.JobState.JOB_STATE_PENDING)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.EntityReconciliationJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_entity_reconciliation_job(request)
    assert isinstance(response, service.EntityReconciliationJob)
    assert response.name == 'name_value'
    assert response.state == job_state.JobState.JOB_STATE_PENDING

def test_get_entity_reconciliation_job_rest_required_fields(request_type=service.GetEntityReconciliationJobRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EnterpriseKnowledgeGraphServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_entity_reconciliation_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_entity_reconciliation_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.EntityReconciliationJob()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.EntityReconciliationJob.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_entity_reconciliation_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_entity_reconciliation_job_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_entity_reconciliation_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_entity_reconciliation_job_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EnterpriseKnowledgeGraphServiceRestInterceptor())
    client = EnterpriseKnowledgeGraphServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EnterpriseKnowledgeGraphServiceRestInterceptor, 'post_get_entity_reconciliation_job') as post, mock.patch.object(transports.EnterpriseKnowledgeGraphServiceRestInterceptor, 'pre_get_entity_reconciliation_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetEntityReconciliationJobRequest.pb(service.GetEntityReconciliationJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.EntityReconciliationJob.to_json(service.EntityReconciliationJob())
        request = service.GetEntityReconciliationJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.EntityReconciliationJob()
        client.get_entity_reconciliation_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_entity_reconciliation_job_rest_bad_request(transport: str='rest', request_type=service.GetEntityReconciliationJobRequest):
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/entityReconciliationJobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_entity_reconciliation_job(request)

def test_get_entity_reconciliation_job_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.EntityReconciliationJob()
        sample_request = {'name': 'projects/sample1/locations/sample2/entityReconciliationJobs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.EntityReconciliationJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_entity_reconciliation_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/entityReconciliationJobs/*}' % client.transport._host, args[1])

def test_get_entity_reconciliation_job_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_entity_reconciliation_job(service.GetEntityReconciliationJobRequest(), name='name_value')

def test_get_entity_reconciliation_job_rest_error():
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListEntityReconciliationJobsRequest, dict])
def test_list_entity_reconciliation_jobs_rest(request_type):
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListEntityReconciliationJobsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListEntityReconciliationJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_entity_reconciliation_jobs(request)
    assert isinstance(response, pagers.ListEntityReconciliationJobsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_entity_reconciliation_jobs_rest_required_fields(request_type=service.ListEntityReconciliationJobsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.EnterpriseKnowledgeGraphServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_entity_reconciliation_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_entity_reconciliation_jobs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListEntityReconciliationJobsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListEntityReconciliationJobsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_entity_reconciliation_jobs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_entity_reconciliation_jobs_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_entity_reconciliation_jobs._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_entity_reconciliation_jobs_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EnterpriseKnowledgeGraphServiceRestInterceptor())
    client = EnterpriseKnowledgeGraphServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EnterpriseKnowledgeGraphServiceRestInterceptor, 'post_list_entity_reconciliation_jobs') as post, mock.patch.object(transports.EnterpriseKnowledgeGraphServiceRestInterceptor, 'pre_list_entity_reconciliation_jobs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListEntityReconciliationJobsRequest.pb(service.ListEntityReconciliationJobsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListEntityReconciliationJobsResponse.to_json(service.ListEntityReconciliationJobsResponse())
        request = service.ListEntityReconciliationJobsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListEntityReconciliationJobsResponse()
        client.list_entity_reconciliation_jobs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_entity_reconciliation_jobs_rest_bad_request(transport: str='rest', request_type=service.ListEntityReconciliationJobsRequest):
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_entity_reconciliation_jobs(request)

def test_list_entity_reconciliation_jobs_rest_flattened():
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListEntityReconciliationJobsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListEntityReconciliationJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_entity_reconciliation_jobs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/entityReconciliationJobs' % client.transport._host, args[1])

def test_list_entity_reconciliation_jobs_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_entity_reconciliation_jobs(service.ListEntityReconciliationJobsRequest(), parent='parent_value')

def test_list_entity_reconciliation_jobs_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[service.EntityReconciliationJob(), service.EntityReconciliationJob(), service.EntityReconciliationJob()], next_page_token='abc'), service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[], next_page_token='def'), service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[service.EntityReconciliationJob()], next_page_token='ghi'), service.ListEntityReconciliationJobsResponse(entity_reconciliation_jobs=[service.EntityReconciliationJob(), service.EntityReconciliationJob()]))
        response = response + response
        response = tuple((service.ListEntityReconciliationJobsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_entity_reconciliation_jobs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service.EntityReconciliationJob) for i in results))
        pages = list(client.list_entity_reconciliation_jobs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.CancelEntityReconciliationJobRequest, dict])
def test_cancel_entity_reconciliation_job_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/entityReconciliationJobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.cancel_entity_reconciliation_job(request)
    assert response is None

def test_cancel_entity_reconciliation_job_rest_required_fields(request_type=service.CancelEntityReconciliationJobRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.EnterpriseKnowledgeGraphServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).cancel_entity_reconciliation_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).cancel_entity_reconciliation_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.cancel_entity_reconciliation_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_cancel_entity_reconciliation_job_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.cancel_entity_reconciliation_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_cancel_entity_reconciliation_job_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EnterpriseKnowledgeGraphServiceRestInterceptor())
    client = EnterpriseKnowledgeGraphServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EnterpriseKnowledgeGraphServiceRestInterceptor, 'pre_cancel_entity_reconciliation_job') as pre:
        pre.assert_not_called()
        pb_message = service.CancelEntityReconciliationJobRequest.pb(service.CancelEntityReconciliationJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = service.CancelEntityReconciliationJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.cancel_entity_reconciliation_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_cancel_entity_reconciliation_job_rest_bad_request(transport: str='rest', request_type=service.CancelEntityReconciliationJobRequest):
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/entityReconciliationJobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.cancel_entity_reconciliation_job(request)

def test_cancel_entity_reconciliation_job_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/entityReconciliationJobs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.cancel_entity_reconciliation_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/entityReconciliationJobs/*}:cancel' % client.transport._host, args[1])

def test_cancel_entity_reconciliation_job_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.cancel_entity_reconciliation_job(service.CancelEntityReconciliationJobRequest(), name='name_value')

def test_cancel_entity_reconciliation_job_rest_error():
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DeleteEntityReconciliationJobRequest, dict])
def test_delete_entity_reconciliation_job_rest(request_type):
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/entityReconciliationJobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_entity_reconciliation_job(request)
    assert response is None

def test_delete_entity_reconciliation_job_rest_required_fields(request_type=service.DeleteEntityReconciliationJobRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EnterpriseKnowledgeGraphServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_entity_reconciliation_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_entity_reconciliation_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_entity_reconciliation_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_entity_reconciliation_job_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_entity_reconciliation_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_entity_reconciliation_job_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EnterpriseKnowledgeGraphServiceRestInterceptor())
    client = EnterpriseKnowledgeGraphServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EnterpriseKnowledgeGraphServiceRestInterceptor, 'pre_delete_entity_reconciliation_job') as pre:
        pre.assert_not_called()
        pb_message = service.DeleteEntityReconciliationJobRequest.pb(service.DeleteEntityReconciliationJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = service.DeleteEntityReconciliationJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_entity_reconciliation_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_entity_reconciliation_job_rest_bad_request(transport: str='rest', request_type=service.DeleteEntityReconciliationJobRequest):
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/entityReconciliationJobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_entity_reconciliation_job(request)

def test_delete_entity_reconciliation_job_rest_flattened():
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/entityReconciliationJobs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_entity_reconciliation_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/entityReconciliationJobs/*}' % client.transport._host, args[1])

def test_delete_entity_reconciliation_job_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_entity_reconciliation_job(service.DeleteEntityReconciliationJobRequest(), name='name_value')

def test_delete_entity_reconciliation_job_rest_error():
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.LookupRequest, dict])
def test_lookup_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.LookupResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = service.LookupResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.lookup(request)
    assert isinstance(response, service.LookupResponse)

def test_lookup_rest_required_fields(request_type=service.LookupRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.EnterpriseKnowledgeGraphServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['ids'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'ids' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).lookup._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'ids' in jsonified_request
    assert jsonified_request['ids'] == request_init['ids']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['ids'] = 'ids_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).lookup._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('ids', 'languages'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'ids' in jsonified_request
    assert jsonified_request['ids'] == 'ids_value'
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.LookupResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.LookupResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.lookup(request)
            expected_params = [('ids', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_lookup_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.lookup._get_unset_required_fields({})
    assert set(unset_fields) == set(('ids', 'languages')) & set(('parent', 'ids'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_lookup_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EnterpriseKnowledgeGraphServiceRestInterceptor())
    client = EnterpriseKnowledgeGraphServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EnterpriseKnowledgeGraphServiceRestInterceptor, 'post_lookup') as post, mock.patch.object(transports.EnterpriseKnowledgeGraphServiceRestInterceptor, 'pre_lookup') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.LookupRequest.pb(service.LookupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.LookupResponse.to_json(service.LookupResponse())
        request = service.LookupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.LookupResponse()
        client.lookup(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_lookup_rest_bad_request(transport: str='rest', request_type=service.LookupRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.lookup(request)

def test_lookup_rest_flattened():
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.LookupResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', ids=['ids_value'])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.LookupResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.lookup(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/cloudKnowledgeGraphEntities:Lookup' % client.transport._host, args[1])

def test_lookup_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.lookup(service.LookupRequest(), parent='parent_value', ids=['ids_value'])

def test_lookup_rest_error():
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.SearchRequest, dict])
def test_search_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.SearchResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = service.SearchResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.search(request)
    assert isinstance(response, service.SearchResponse)

def test_search_rest_required_fields(request_type=service.SearchRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EnterpriseKnowledgeGraphServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['query'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'query' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'query' in jsonified_request
    assert jsonified_request['query'] == request_init['query']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['query'] = 'query_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('languages', 'limit', 'query', 'types'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'query' in jsonified_request
    assert jsonified_request['query'] == 'query_value'
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.SearchResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.SearchResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.search(request)
            expected_params = [('query', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_search_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.search._get_unset_required_fields({})
    assert set(unset_fields) == set(('languages', 'limit', 'query', 'types')) & set(('parent', 'query'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_search_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EnterpriseKnowledgeGraphServiceRestInterceptor())
    client = EnterpriseKnowledgeGraphServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EnterpriseKnowledgeGraphServiceRestInterceptor, 'post_search') as post, mock.patch.object(transports.EnterpriseKnowledgeGraphServiceRestInterceptor, 'pre_search') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.SearchRequest.pb(service.SearchRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.SearchResponse.to_json(service.SearchResponse())
        request = service.SearchRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.SearchResponse()
        client.search(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_search_rest_bad_request(transport: str='rest', request_type=service.SearchRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.search(request)

def test_search_rest_flattened():
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.SearchResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', query='query_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.SearchResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.search(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/cloudKnowledgeGraphEntities:Search' % client.transport._host, args[1])

def test_search_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.search(service.SearchRequest(), parent='parent_value', query='query_value')

def test_search_rest_error():
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.LookupPublicKgRequest, dict])
def test_lookup_public_kg_rest(request_type):
    if False:
        while True:
            i = 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.LookupPublicKgResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = service.LookupPublicKgResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.lookup_public_kg(request)
    assert isinstance(response, service.LookupPublicKgResponse)

def test_lookup_public_kg_rest_required_fields(request_type=service.LookupPublicKgRequest):
    if False:
        return 10
    transport_class = transports.EnterpriseKnowledgeGraphServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['ids'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'ids' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).lookup_public_kg._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'ids' in jsonified_request
    assert jsonified_request['ids'] == request_init['ids']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['ids'] = 'ids_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).lookup_public_kg._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('ids', 'languages'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'ids' in jsonified_request
    assert jsonified_request['ids'] == 'ids_value'
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.LookupPublicKgResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.LookupPublicKgResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.lookup_public_kg(request)
            expected_params = [('ids', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_lookup_public_kg_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.lookup_public_kg._get_unset_required_fields({})
    assert set(unset_fields) == set(('ids', 'languages')) & set(('parent', 'ids'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_lookup_public_kg_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EnterpriseKnowledgeGraphServiceRestInterceptor())
    client = EnterpriseKnowledgeGraphServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EnterpriseKnowledgeGraphServiceRestInterceptor, 'post_lookup_public_kg') as post, mock.patch.object(transports.EnterpriseKnowledgeGraphServiceRestInterceptor, 'pre_lookup_public_kg') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.LookupPublicKgRequest.pb(service.LookupPublicKgRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.LookupPublicKgResponse.to_json(service.LookupPublicKgResponse())
        request = service.LookupPublicKgRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.LookupPublicKgResponse()
        client.lookup_public_kg(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_lookup_public_kg_rest_bad_request(transport: str='rest', request_type=service.LookupPublicKgRequest):
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.lookup_public_kg(request)

def test_lookup_public_kg_rest_flattened():
    if False:
        while True:
            i = 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.LookupPublicKgResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', ids=['ids_value'])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.LookupPublicKgResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.lookup_public_kg(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/publicKnowledgeGraphEntities:Lookup' % client.transport._host, args[1])

def test_lookup_public_kg_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.lookup_public_kg(service.LookupPublicKgRequest(), parent='parent_value', ids=['ids_value'])

def test_lookup_public_kg_rest_error():
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.SearchPublicKgRequest, dict])
def test_search_public_kg_rest(request_type):
    if False:
        while True:
            i = 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.SearchPublicKgResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = service.SearchPublicKgResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.search_public_kg(request)
    assert isinstance(response, service.SearchPublicKgResponse)

def test_search_public_kg_rest_required_fields(request_type=service.SearchPublicKgRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EnterpriseKnowledgeGraphServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['query'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'query' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_public_kg._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'query' in jsonified_request
    assert jsonified_request['query'] == request_init['query']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['query'] = 'query_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).search_public_kg._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('languages', 'limit', 'query', 'types'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'query' in jsonified_request
    assert jsonified_request['query'] == 'query_value'
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.SearchPublicKgResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.SearchPublicKgResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.search_public_kg(request)
            expected_params = [('query', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_search_public_kg_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.search_public_kg._get_unset_required_fields({})
    assert set(unset_fields) == set(('languages', 'limit', 'query', 'types')) & set(('parent', 'query'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_search_public_kg_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EnterpriseKnowledgeGraphServiceRestInterceptor())
    client = EnterpriseKnowledgeGraphServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EnterpriseKnowledgeGraphServiceRestInterceptor, 'post_search_public_kg') as post, mock.patch.object(transports.EnterpriseKnowledgeGraphServiceRestInterceptor, 'pre_search_public_kg') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.SearchPublicKgRequest.pb(service.SearchPublicKgRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.SearchPublicKgResponse.to_json(service.SearchPublicKgResponse())
        request = service.SearchPublicKgRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.SearchPublicKgResponse()
        client.search_public_kg(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_search_public_kg_rest_bad_request(transport: str='rest', request_type=service.SearchPublicKgRequest):
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.search_public_kg(request)

def test_search_public_kg_rest_flattened():
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.SearchPublicKgResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', query='query_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.SearchPublicKgResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.search_public_kg(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/publicKnowledgeGraphEntities:Search' % client.transport._host, args[1])

def test_search_public_kg_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.search_public_kg(service.SearchPublicKgRequest(), parent='parent_value', query='query_value')

def test_search_public_kg_rest_error():
    if False:
        print('Hello World!')
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.EnterpriseKnowledgeGraphServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.EnterpriseKnowledgeGraphServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EnterpriseKnowledgeGraphServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.EnterpriseKnowledgeGraphServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = EnterpriseKnowledgeGraphServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = EnterpriseKnowledgeGraphServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.EnterpriseKnowledgeGraphServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EnterpriseKnowledgeGraphServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.EnterpriseKnowledgeGraphServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = EnterpriseKnowledgeGraphServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EnterpriseKnowledgeGraphServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.EnterpriseKnowledgeGraphServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.EnterpriseKnowledgeGraphServiceGrpcTransport, transports.EnterpriseKnowledgeGraphServiceGrpcAsyncIOTransport, transports.EnterpriseKnowledgeGraphServiceRestTransport])
def test_transport_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        return 10
    transport = EnterpriseKnowledgeGraphServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.EnterpriseKnowledgeGraphServiceGrpcTransport)

def test_enterprise_knowledge_graph_service_base_transport_error():
    if False:
        return 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.EnterpriseKnowledgeGraphServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_enterprise_knowledge_graph_service_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.enterpriseknowledgegraph_v1.services.enterprise_knowledge_graph_service.transports.EnterpriseKnowledgeGraphServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.EnterpriseKnowledgeGraphServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_entity_reconciliation_job', 'get_entity_reconciliation_job', 'list_entity_reconciliation_jobs', 'cancel_entity_reconciliation_job', 'delete_entity_reconciliation_job', 'lookup', 'search', 'lookup_public_kg', 'search_public_kg')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_enterprise_knowledge_graph_service_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.enterpriseknowledgegraph_v1.services.enterprise_knowledge_graph_service.transports.EnterpriseKnowledgeGraphServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.EnterpriseKnowledgeGraphServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_enterprise_knowledge_graph_service_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.enterpriseknowledgegraph_v1.services.enterprise_knowledge_graph_service.transports.EnterpriseKnowledgeGraphServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.EnterpriseKnowledgeGraphServiceTransport()
        adc.assert_called_once()

def test_enterprise_knowledge_graph_service_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        EnterpriseKnowledgeGraphServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.EnterpriseKnowledgeGraphServiceGrpcTransport, transports.EnterpriseKnowledgeGraphServiceGrpcAsyncIOTransport])
def test_enterprise_knowledge_graph_service_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.EnterpriseKnowledgeGraphServiceGrpcTransport, transports.EnterpriseKnowledgeGraphServiceGrpcAsyncIOTransport, transports.EnterpriseKnowledgeGraphServiceRestTransport])
def test_enterprise_knowledge_graph_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.EnterpriseKnowledgeGraphServiceGrpcTransport, grpc_helpers), (transports.EnterpriseKnowledgeGraphServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_enterprise_knowledge_graph_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('enterpriseknowledgegraph.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='enterpriseknowledgegraph.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.EnterpriseKnowledgeGraphServiceGrpcTransport, transports.EnterpriseKnowledgeGraphServiceGrpcAsyncIOTransport])
def test_enterprise_knowledge_graph_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_enterprise_knowledge_graph_service_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.EnterpriseKnowledgeGraphServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_enterprise_knowledge_graph_service_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='enterpriseknowledgegraph.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('enterpriseknowledgegraph.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://enterpriseknowledgegraph.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_enterprise_knowledge_graph_service_host_with_port(transport_name):
    if False:
        return 10
    client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='enterpriseknowledgegraph.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('enterpriseknowledgegraph.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://enterpriseknowledgegraph.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_enterprise_knowledge_graph_service_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = EnterpriseKnowledgeGraphServiceClient(credentials=creds1, transport=transport_name)
    client2 = EnterpriseKnowledgeGraphServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_entity_reconciliation_job._session
    session2 = client2.transport.create_entity_reconciliation_job._session
    assert session1 != session2
    session1 = client1.transport.get_entity_reconciliation_job._session
    session2 = client2.transport.get_entity_reconciliation_job._session
    assert session1 != session2
    session1 = client1.transport.list_entity_reconciliation_jobs._session
    session2 = client2.transport.list_entity_reconciliation_jobs._session
    assert session1 != session2
    session1 = client1.transport.cancel_entity_reconciliation_job._session
    session2 = client2.transport.cancel_entity_reconciliation_job._session
    assert session1 != session2
    session1 = client1.transport.delete_entity_reconciliation_job._session
    session2 = client2.transport.delete_entity_reconciliation_job._session
    assert session1 != session2
    session1 = client1.transport.lookup._session
    session2 = client2.transport.lookup._session
    assert session1 != session2
    session1 = client1.transport.search._session
    session2 = client2.transport.search._session
    assert session1 != session2
    session1 = client1.transport.lookup_public_kg._session
    session2 = client2.transport.lookup_public_kg._session
    assert session1 != session2
    session1 = client1.transport.search_public_kg._session
    session2 = client2.transport.search_public_kg._session
    assert session1 != session2

def test_enterprise_knowledge_graph_service_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.EnterpriseKnowledgeGraphServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_enterprise_knowledge_graph_service_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.EnterpriseKnowledgeGraphServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.EnterpriseKnowledgeGraphServiceGrpcTransport, transports.EnterpriseKnowledgeGraphServiceGrpcAsyncIOTransport])
def test_enterprise_knowledge_graph_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.EnterpriseKnowledgeGraphServiceGrpcTransport, transports.EnterpriseKnowledgeGraphServiceGrpcAsyncIOTransport])
def test_enterprise_knowledge_graph_service_transport_channel_mtls_with_adc(transport_class):
    if False:
        i = 10
        return i + 15
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

def test_cloud_knowledge_graph_entity_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    cloud_knowledge_graph_entity = 'whelk'
    expected = 'projects/{project}/locations/{location}/cloudKnowledgeGraphEntities/{cloud_knowledge_graph_entity}'.format(project=project, location=location, cloud_knowledge_graph_entity=cloud_knowledge_graph_entity)
    actual = EnterpriseKnowledgeGraphServiceClient.cloud_knowledge_graph_entity_path(project, location, cloud_knowledge_graph_entity)
    assert expected == actual

def test_parse_cloud_knowledge_graph_entity_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'octopus', 'location': 'oyster', 'cloud_knowledge_graph_entity': 'nudibranch'}
    path = EnterpriseKnowledgeGraphServiceClient.cloud_knowledge_graph_entity_path(**expected)
    actual = EnterpriseKnowledgeGraphServiceClient.parse_cloud_knowledge_graph_entity_path(path)
    assert expected == actual

def test_dataset_path():
    if False:
        return 10
    project = 'cuttlefish'
    dataset = 'mussel'
    expected = 'projects/{project}/datasets/{dataset}'.format(project=project, dataset=dataset)
    actual = EnterpriseKnowledgeGraphServiceClient.dataset_path(project, dataset)
    assert expected == actual

def test_parse_dataset_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'winkle', 'dataset': 'nautilus'}
    path = EnterpriseKnowledgeGraphServiceClient.dataset_path(**expected)
    actual = EnterpriseKnowledgeGraphServiceClient.parse_dataset_path(path)
    assert expected == actual

def test_entity_reconciliation_job_path():
    if False:
        return 10
    project = 'scallop'
    location = 'abalone'
    entity_reconciliation_job = 'squid'
    expected = 'projects/{project}/locations/{location}/entityReconciliationJobs/{entity_reconciliation_job}'.format(project=project, location=location, entity_reconciliation_job=entity_reconciliation_job)
    actual = EnterpriseKnowledgeGraphServiceClient.entity_reconciliation_job_path(project, location, entity_reconciliation_job)
    assert expected == actual

def test_parse_entity_reconciliation_job_path():
    if False:
        return 10
    expected = {'project': 'clam', 'location': 'whelk', 'entity_reconciliation_job': 'octopus'}
    path = EnterpriseKnowledgeGraphServiceClient.entity_reconciliation_job_path(**expected)
    actual = EnterpriseKnowledgeGraphServiceClient.parse_entity_reconciliation_job_path(path)
    assert expected == actual

def test_public_knowledge_graph_entity_path():
    if False:
        return 10
    project = 'oyster'
    location = 'nudibranch'
    public_knowledge_graph_entity = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/publicKnowledgeGraphEntities/{public_knowledge_graph_entity}'.format(project=project, location=location, public_knowledge_graph_entity=public_knowledge_graph_entity)
    actual = EnterpriseKnowledgeGraphServiceClient.public_knowledge_graph_entity_path(project, location, public_knowledge_graph_entity)
    assert expected == actual

def test_parse_public_knowledge_graph_entity_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'mussel', 'location': 'winkle', 'public_knowledge_graph_entity': 'nautilus'}
    path = EnterpriseKnowledgeGraphServiceClient.public_knowledge_graph_entity_path(**expected)
    actual = EnterpriseKnowledgeGraphServiceClient.parse_public_knowledge_graph_entity_path(path)
    assert expected == actual

def test_table_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'scallop'
    dataset = 'abalone'
    table = 'squid'
    expected = 'projects/{project}/datasets/{dataset}/tables/{table}'.format(project=project, dataset=dataset, table=table)
    actual = EnterpriseKnowledgeGraphServiceClient.table_path(project, dataset, table)
    assert expected == actual

def test_parse_table_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'clam', 'dataset': 'whelk', 'table': 'octopus'}
    path = EnterpriseKnowledgeGraphServiceClient.table_path(**expected)
    actual = EnterpriseKnowledgeGraphServiceClient.parse_table_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = EnterpriseKnowledgeGraphServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'nudibranch'}
    path = EnterpriseKnowledgeGraphServiceClient.common_billing_account_path(**expected)
    actual = EnterpriseKnowledgeGraphServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = EnterpriseKnowledgeGraphServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'mussel'}
    path = EnterpriseKnowledgeGraphServiceClient.common_folder_path(**expected)
    actual = EnterpriseKnowledgeGraphServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = EnterpriseKnowledgeGraphServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'nautilus'}
    path = EnterpriseKnowledgeGraphServiceClient.common_organization_path(**expected)
    actual = EnterpriseKnowledgeGraphServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = EnterpriseKnowledgeGraphServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'abalone'}
    path = EnterpriseKnowledgeGraphServiceClient.common_project_path(**expected)
    actual = EnterpriseKnowledgeGraphServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = EnterpriseKnowledgeGraphServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = EnterpriseKnowledgeGraphServiceClient.common_location_path(**expected)
    actual = EnterpriseKnowledgeGraphServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.EnterpriseKnowledgeGraphServiceTransport, '_prep_wrapped_messages') as prep:
        client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.EnterpriseKnowledgeGraphServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = EnterpriseKnowledgeGraphServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = EnterpriseKnowledgeGraphServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
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
        client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = EnterpriseKnowledgeGraphServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(EnterpriseKnowledgeGraphServiceClient, transports.EnterpriseKnowledgeGraphServiceGrpcTransport), (EnterpriseKnowledgeGraphServiceAsyncClient, transports.EnterpriseKnowledgeGraphServiceGrpcAsyncIOTransport)])
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