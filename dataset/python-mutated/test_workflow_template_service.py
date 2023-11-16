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
from google.protobuf import duration_pb2
from google.protobuf import empty_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.dataproc_v1.services.workflow_template_service import WorkflowTemplateServiceAsyncClient, WorkflowTemplateServiceClient, pagers, transports
from google.cloud.dataproc_v1.types import clusters, jobs, shared, workflow_templates

def client_cert_source_callback():
    if False:
        while True:
            i = 10
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
    assert WorkflowTemplateServiceClient._get_default_mtls_endpoint(None) is None
    assert WorkflowTemplateServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert WorkflowTemplateServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert WorkflowTemplateServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert WorkflowTemplateServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert WorkflowTemplateServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(WorkflowTemplateServiceClient, 'grpc'), (WorkflowTemplateServiceAsyncClient, 'grpc_asyncio'), (WorkflowTemplateServiceClient, 'rest')])
def test_workflow_template_service_client_from_service_account_info(client_class, transport_name):
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

@pytest.mark.parametrize('transport_class,transport_name', [(transports.WorkflowTemplateServiceGrpcTransport, 'grpc'), (transports.WorkflowTemplateServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.WorkflowTemplateServiceRestTransport, 'rest')])
def test_workflow_template_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(WorkflowTemplateServiceClient, 'grpc'), (WorkflowTemplateServiceAsyncClient, 'grpc_asyncio'), (WorkflowTemplateServiceClient, 'rest')])
def test_workflow_template_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('dataproc.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dataproc.googleapis.com')

def test_workflow_template_service_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = WorkflowTemplateServiceClient.get_transport_class()
    available_transports = [transports.WorkflowTemplateServiceGrpcTransport, transports.WorkflowTemplateServiceRestTransport]
    assert transport in available_transports
    transport = WorkflowTemplateServiceClient.get_transport_class('grpc')
    assert transport == transports.WorkflowTemplateServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(WorkflowTemplateServiceClient, transports.WorkflowTemplateServiceGrpcTransport, 'grpc'), (WorkflowTemplateServiceAsyncClient, transports.WorkflowTemplateServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (WorkflowTemplateServiceClient, transports.WorkflowTemplateServiceRestTransport, 'rest')])
@mock.patch.object(WorkflowTemplateServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WorkflowTemplateServiceClient))
@mock.patch.object(WorkflowTemplateServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WorkflowTemplateServiceAsyncClient))
def test_workflow_template_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(WorkflowTemplateServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(WorkflowTemplateServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(WorkflowTemplateServiceClient, transports.WorkflowTemplateServiceGrpcTransport, 'grpc', 'true'), (WorkflowTemplateServiceAsyncClient, transports.WorkflowTemplateServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (WorkflowTemplateServiceClient, transports.WorkflowTemplateServiceGrpcTransport, 'grpc', 'false'), (WorkflowTemplateServiceAsyncClient, transports.WorkflowTemplateServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (WorkflowTemplateServiceClient, transports.WorkflowTemplateServiceRestTransport, 'rest', 'true'), (WorkflowTemplateServiceClient, transports.WorkflowTemplateServiceRestTransport, 'rest', 'false')])
@mock.patch.object(WorkflowTemplateServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WorkflowTemplateServiceClient))
@mock.patch.object(WorkflowTemplateServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WorkflowTemplateServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_workflow_template_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [WorkflowTemplateServiceClient, WorkflowTemplateServiceAsyncClient])
@mock.patch.object(WorkflowTemplateServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WorkflowTemplateServiceClient))
@mock.patch.object(WorkflowTemplateServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(WorkflowTemplateServiceAsyncClient))
def test_workflow_template_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(WorkflowTemplateServiceClient, transports.WorkflowTemplateServiceGrpcTransport, 'grpc'), (WorkflowTemplateServiceAsyncClient, transports.WorkflowTemplateServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (WorkflowTemplateServiceClient, transports.WorkflowTemplateServiceRestTransport, 'rest')])
def test_workflow_template_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(WorkflowTemplateServiceClient, transports.WorkflowTemplateServiceGrpcTransport, 'grpc', grpc_helpers), (WorkflowTemplateServiceAsyncClient, transports.WorkflowTemplateServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (WorkflowTemplateServiceClient, transports.WorkflowTemplateServiceRestTransport, 'rest', None)])
def test_workflow_template_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_workflow_template_service_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.dataproc_v1.services.workflow_template_service.transports.WorkflowTemplateServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = WorkflowTemplateServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(WorkflowTemplateServiceClient, transports.WorkflowTemplateServiceGrpcTransport, 'grpc', grpc_helpers), (WorkflowTemplateServiceAsyncClient, transports.WorkflowTemplateServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_workflow_template_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('dataproc.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='dataproc.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [workflow_templates.CreateWorkflowTemplateRequest, dict])
def test_create_workflow_template(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_workflow_template), '__call__') as call:
        call.return_value = workflow_templates.WorkflowTemplate(id='id_value', name='name_value', version=774)
        response = client.create_workflow_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.CreateWorkflowTemplateRequest()
    assert isinstance(response, workflow_templates.WorkflowTemplate)
    assert response.id == 'id_value'
    assert response.name == 'name_value'
    assert response.version == 774

def test_create_workflow_template_empty_call():
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_workflow_template), '__call__') as call:
        client.create_workflow_template()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.CreateWorkflowTemplateRequest()

@pytest.mark.asyncio
async def test_create_workflow_template_async(transport: str='grpc_asyncio', request_type=workflow_templates.CreateWorkflowTemplateRequest):
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_workflow_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workflow_templates.WorkflowTemplate(id='id_value', name='name_value', version=774))
        response = await client.create_workflow_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.CreateWorkflowTemplateRequest()
    assert isinstance(response, workflow_templates.WorkflowTemplate)
    assert response.id == 'id_value'
    assert response.name == 'name_value'
    assert response.version == 774

@pytest.mark.asyncio
async def test_create_workflow_template_async_from_dict():
    await test_create_workflow_template_async(request_type=dict)

def test_create_workflow_template_field_headers():
    if False:
        print('Hello World!')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = workflow_templates.CreateWorkflowTemplateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_workflow_template), '__call__') as call:
        call.return_value = workflow_templates.WorkflowTemplate()
        client.create_workflow_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_workflow_template_field_headers_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workflow_templates.CreateWorkflowTemplateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_workflow_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workflow_templates.WorkflowTemplate())
        await client.create_workflow_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_workflow_template_flattened():
    if False:
        print('Hello World!')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_workflow_template), '__call__') as call:
        call.return_value = workflow_templates.WorkflowTemplate()
        client.create_workflow_template(parent='parent_value', template=workflow_templates.WorkflowTemplate(id='id_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].template
        mock_val = workflow_templates.WorkflowTemplate(id='id_value')
        assert arg == mock_val

def test_create_workflow_template_flattened_error():
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_workflow_template(workflow_templates.CreateWorkflowTemplateRequest(), parent='parent_value', template=workflow_templates.WorkflowTemplate(id='id_value'))

@pytest.mark.asyncio
async def test_create_workflow_template_flattened_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_workflow_template), '__call__') as call:
        call.return_value = workflow_templates.WorkflowTemplate()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workflow_templates.WorkflowTemplate())
        response = await client.create_workflow_template(parent='parent_value', template=workflow_templates.WorkflowTemplate(id='id_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].template
        mock_val = workflow_templates.WorkflowTemplate(id='id_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_workflow_template_flattened_error_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_workflow_template(workflow_templates.CreateWorkflowTemplateRequest(), parent='parent_value', template=workflow_templates.WorkflowTemplate(id='id_value'))

@pytest.mark.parametrize('request_type', [workflow_templates.GetWorkflowTemplateRequest, dict])
def test_get_workflow_template(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_workflow_template), '__call__') as call:
        call.return_value = workflow_templates.WorkflowTemplate(id='id_value', name='name_value', version=774)
        response = client.get_workflow_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.GetWorkflowTemplateRequest()
    assert isinstance(response, workflow_templates.WorkflowTemplate)
    assert response.id == 'id_value'
    assert response.name == 'name_value'
    assert response.version == 774

def test_get_workflow_template_empty_call():
    if False:
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_workflow_template), '__call__') as call:
        client.get_workflow_template()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.GetWorkflowTemplateRequest()

@pytest.mark.asyncio
async def test_get_workflow_template_async(transport: str='grpc_asyncio', request_type=workflow_templates.GetWorkflowTemplateRequest):
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_workflow_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workflow_templates.WorkflowTemplate(id='id_value', name='name_value', version=774))
        response = await client.get_workflow_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.GetWorkflowTemplateRequest()
    assert isinstance(response, workflow_templates.WorkflowTemplate)
    assert response.id == 'id_value'
    assert response.name == 'name_value'
    assert response.version == 774

@pytest.mark.asyncio
async def test_get_workflow_template_async_from_dict():
    await test_get_workflow_template_async(request_type=dict)

def test_get_workflow_template_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = workflow_templates.GetWorkflowTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_workflow_template), '__call__') as call:
        call.return_value = workflow_templates.WorkflowTemplate()
        client.get_workflow_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_workflow_template_field_headers_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workflow_templates.GetWorkflowTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_workflow_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workflow_templates.WorkflowTemplate())
        await client.get_workflow_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_workflow_template_flattened():
    if False:
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_workflow_template), '__call__') as call:
        call.return_value = workflow_templates.WorkflowTemplate()
        client.get_workflow_template(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_workflow_template_flattened_error():
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_workflow_template(workflow_templates.GetWorkflowTemplateRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_workflow_template_flattened_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_workflow_template), '__call__') as call:
        call.return_value = workflow_templates.WorkflowTemplate()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workflow_templates.WorkflowTemplate())
        response = await client.get_workflow_template(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_workflow_template_flattened_error_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_workflow_template(workflow_templates.GetWorkflowTemplateRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [workflow_templates.InstantiateWorkflowTemplateRequest, dict])
def test_instantiate_workflow_template(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.instantiate_workflow_template), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.instantiate_workflow_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.InstantiateWorkflowTemplateRequest()
    assert isinstance(response, future.Future)

def test_instantiate_workflow_template_empty_call():
    if False:
        while True:
            i = 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.instantiate_workflow_template), '__call__') as call:
        client.instantiate_workflow_template()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.InstantiateWorkflowTemplateRequest()

@pytest.mark.asyncio
async def test_instantiate_workflow_template_async(transport: str='grpc_asyncio', request_type=workflow_templates.InstantiateWorkflowTemplateRequest):
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.instantiate_workflow_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.instantiate_workflow_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.InstantiateWorkflowTemplateRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_instantiate_workflow_template_async_from_dict():
    await test_instantiate_workflow_template_async(request_type=dict)

def test_instantiate_workflow_template_field_headers():
    if False:
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = workflow_templates.InstantiateWorkflowTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.instantiate_workflow_template), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.instantiate_workflow_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_instantiate_workflow_template_field_headers_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workflow_templates.InstantiateWorkflowTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.instantiate_workflow_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.instantiate_workflow_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_instantiate_workflow_template_flattened():
    if False:
        while True:
            i = 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.instantiate_workflow_template), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.instantiate_workflow_template(name='name_value', parameters={'key_value': 'value_value'})
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].parameters
        mock_val = {'key_value': 'value_value'}
        assert arg == mock_val

def test_instantiate_workflow_template_flattened_error():
    if False:
        while True:
            i = 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.instantiate_workflow_template(workflow_templates.InstantiateWorkflowTemplateRequest(), name='name_value', parameters={'key_value': 'value_value'})

@pytest.mark.asyncio
async def test_instantiate_workflow_template_flattened_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.instantiate_workflow_template), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.instantiate_workflow_template(name='name_value', parameters={'key_value': 'value_value'})
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].parameters
        mock_val = {'key_value': 'value_value'}
        assert arg == mock_val

@pytest.mark.asyncio
async def test_instantiate_workflow_template_flattened_error_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.instantiate_workflow_template(workflow_templates.InstantiateWorkflowTemplateRequest(), name='name_value', parameters={'key_value': 'value_value'})

@pytest.mark.parametrize('request_type', [workflow_templates.InstantiateInlineWorkflowTemplateRequest, dict])
def test_instantiate_inline_workflow_template(request_type, transport: str='grpc'):
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.instantiate_inline_workflow_template), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.instantiate_inline_workflow_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.InstantiateInlineWorkflowTemplateRequest()
    assert isinstance(response, future.Future)

def test_instantiate_inline_workflow_template_empty_call():
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.instantiate_inline_workflow_template), '__call__') as call:
        client.instantiate_inline_workflow_template()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.InstantiateInlineWorkflowTemplateRequest()

@pytest.mark.asyncio
async def test_instantiate_inline_workflow_template_async(transport: str='grpc_asyncio', request_type=workflow_templates.InstantiateInlineWorkflowTemplateRequest):
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.instantiate_inline_workflow_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.instantiate_inline_workflow_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.InstantiateInlineWorkflowTemplateRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_instantiate_inline_workflow_template_async_from_dict():
    await test_instantiate_inline_workflow_template_async(request_type=dict)

def test_instantiate_inline_workflow_template_field_headers():
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = workflow_templates.InstantiateInlineWorkflowTemplateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.instantiate_inline_workflow_template), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.instantiate_inline_workflow_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_instantiate_inline_workflow_template_field_headers_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workflow_templates.InstantiateInlineWorkflowTemplateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.instantiate_inline_workflow_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.instantiate_inline_workflow_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_instantiate_inline_workflow_template_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.instantiate_inline_workflow_template), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.instantiate_inline_workflow_template(parent='parent_value', template=workflow_templates.WorkflowTemplate(id='id_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].template
        mock_val = workflow_templates.WorkflowTemplate(id='id_value')
        assert arg == mock_val

def test_instantiate_inline_workflow_template_flattened_error():
    if False:
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.instantiate_inline_workflow_template(workflow_templates.InstantiateInlineWorkflowTemplateRequest(), parent='parent_value', template=workflow_templates.WorkflowTemplate(id='id_value'))

@pytest.mark.asyncio
async def test_instantiate_inline_workflow_template_flattened_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.instantiate_inline_workflow_template), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.instantiate_inline_workflow_template(parent='parent_value', template=workflow_templates.WorkflowTemplate(id='id_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].template
        mock_val = workflow_templates.WorkflowTemplate(id='id_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_instantiate_inline_workflow_template_flattened_error_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.instantiate_inline_workflow_template(workflow_templates.InstantiateInlineWorkflowTemplateRequest(), parent='parent_value', template=workflow_templates.WorkflowTemplate(id='id_value'))

@pytest.mark.parametrize('request_type', [workflow_templates.UpdateWorkflowTemplateRequest, dict])
def test_update_workflow_template(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_workflow_template), '__call__') as call:
        call.return_value = workflow_templates.WorkflowTemplate(id='id_value', name='name_value', version=774)
        response = client.update_workflow_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.UpdateWorkflowTemplateRequest()
    assert isinstance(response, workflow_templates.WorkflowTemplate)
    assert response.id == 'id_value'
    assert response.name == 'name_value'
    assert response.version == 774

def test_update_workflow_template_empty_call():
    if False:
        print('Hello World!')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_workflow_template), '__call__') as call:
        client.update_workflow_template()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.UpdateWorkflowTemplateRequest()

@pytest.mark.asyncio
async def test_update_workflow_template_async(transport: str='grpc_asyncio', request_type=workflow_templates.UpdateWorkflowTemplateRequest):
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_workflow_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workflow_templates.WorkflowTemplate(id='id_value', name='name_value', version=774))
        response = await client.update_workflow_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.UpdateWorkflowTemplateRequest()
    assert isinstance(response, workflow_templates.WorkflowTemplate)
    assert response.id == 'id_value'
    assert response.name == 'name_value'
    assert response.version == 774

@pytest.mark.asyncio
async def test_update_workflow_template_async_from_dict():
    await test_update_workflow_template_async(request_type=dict)

def test_update_workflow_template_field_headers():
    if False:
        print('Hello World!')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = workflow_templates.UpdateWorkflowTemplateRequest()
    request.template.name = 'name_value'
    with mock.patch.object(type(client.transport.update_workflow_template), '__call__') as call:
        call.return_value = workflow_templates.WorkflowTemplate()
        client.update_workflow_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'template.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_workflow_template_field_headers_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workflow_templates.UpdateWorkflowTemplateRequest()
    request.template.name = 'name_value'
    with mock.patch.object(type(client.transport.update_workflow_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workflow_templates.WorkflowTemplate())
        await client.update_workflow_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'template.name=name_value') in kw['metadata']

def test_update_workflow_template_flattened():
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_workflow_template), '__call__') as call:
        call.return_value = workflow_templates.WorkflowTemplate()
        client.update_workflow_template(template=workflow_templates.WorkflowTemplate(id='id_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].template
        mock_val = workflow_templates.WorkflowTemplate(id='id_value')
        assert arg == mock_val

def test_update_workflow_template_flattened_error():
    if False:
        print('Hello World!')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_workflow_template(workflow_templates.UpdateWorkflowTemplateRequest(), template=workflow_templates.WorkflowTemplate(id='id_value'))

@pytest.mark.asyncio
async def test_update_workflow_template_flattened_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_workflow_template), '__call__') as call:
        call.return_value = workflow_templates.WorkflowTemplate()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workflow_templates.WorkflowTemplate())
        response = await client.update_workflow_template(template=workflow_templates.WorkflowTemplate(id='id_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].template
        mock_val = workflow_templates.WorkflowTemplate(id='id_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_workflow_template_flattened_error_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_workflow_template(workflow_templates.UpdateWorkflowTemplateRequest(), template=workflow_templates.WorkflowTemplate(id='id_value'))

@pytest.mark.parametrize('request_type', [workflow_templates.ListWorkflowTemplatesRequest, dict])
def test_list_workflow_templates(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_workflow_templates), '__call__') as call:
        call.return_value = workflow_templates.ListWorkflowTemplatesResponse(next_page_token='next_page_token_value')
        response = client.list_workflow_templates(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.ListWorkflowTemplatesRequest()
    assert isinstance(response, pagers.ListWorkflowTemplatesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_workflow_templates_empty_call():
    if False:
        print('Hello World!')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_workflow_templates), '__call__') as call:
        client.list_workflow_templates()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.ListWorkflowTemplatesRequest()

@pytest.mark.asyncio
async def test_list_workflow_templates_async(transport: str='grpc_asyncio', request_type=workflow_templates.ListWorkflowTemplatesRequest):
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_workflow_templates), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workflow_templates.ListWorkflowTemplatesResponse(next_page_token='next_page_token_value'))
        response = await client.list_workflow_templates(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.ListWorkflowTemplatesRequest()
    assert isinstance(response, pagers.ListWorkflowTemplatesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_workflow_templates_async_from_dict():
    await test_list_workflow_templates_async(request_type=dict)

def test_list_workflow_templates_field_headers():
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = workflow_templates.ListWorkflowTemplatesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_workflow_templates), '__call__') as call:
        call.return_value = workflow_templates.ListWorkflowTemplatesResponse()
        client.list_workflow_templates(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_workflow_templates_field_headers_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workflow_templates.ListWorkflowTemplatesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_workflow_templates), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workflow_templates.ListWorkflowTemplatesResponse())
        await client.list_workflow_templates(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_workflow_templates_flattened():
    if False:
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_workflow_templates), '__call__') as call:
        call.return_value = workflow_templates.ListWorkflowTemplatesResponse()
        client.list_workflow_templates(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_workflow_templates_flattened_error():
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_workflow_templates(workflow_templates.ListWorkflowTemplatesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_workflow_templates_flattened_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_workflow_templates), '__call__') as call:
        call.return_value = workflow_templates.ListWorkflowTemplatesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(workflow_templates.ListWorkflowTemplatesResponse())
        response = await client.list_workflow_templates(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_workflow_templates_flattened_error_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_workflow_templates(workflow_templates.ListWorkflowTemplatesRequest(), parent='parent_value')

def test_list_workflow_templates_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_workflow_templates), '__call__') as call:
        call.side_effect = (workflow_templates.ListWorkflowTemplatesResponse(templates=[workflow_templates.WorkflowTemplate(), workflow_templates.WorkflowTemplate(), workflow_templates.WorkflowTemplate()], next_page_token='abc'), workflow_templates.ListWorkflowTemplatesResponse(templates=[], next_page_token='def'), workflow_templates.ListWorkflowTemplatesResponse(templates=[workflow_templates.WorkflowTemplate()], next_page_token='ghi'), workflow_templates.ListWorkflowTemplatesResponse(templates=[workflow_templates.WorkflowTemplate(), workflow_templates.WorkflowTemplate()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_workflow_templates(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, workflow_templates.WorkflowTemplate) for i in results))

def test_list_workflow_templates_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_workflow_templates), '__call__') as call:
        call.side_effect = (workflow_templates.ListWorkflowTemplatesResponse(templates=[workflow_templates.WorkflowTemplate(), workflow_templates.WorkflowTemplate(), workflow_templates.WorkflowTemplate()], next_page_token='abc'), workflow_templates.ListWorkflowTemplatesResponse(templates=[], next_page_token='def'), workflow_templates.ListWorkflowTemplatesResponse(templates=[workflow_templates.WorkflowTemplate()], next_page_token='ghi'), workflow_templates.ListWorkflowTemplatesResponse(templates=[workflow_templates.WorkflowTemplate(), workflow_templates.WorkflowTemplate()]), RuntimeError)
        pages = list(client.list_workflow_templates(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_workflow_templates_async_pager():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_workflow_templates), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (workflow_templates.ListWorkflowTemplatesResponse(templates=[workflow_templates.WorkflowTemplate(), workflow_templates.WorkflowTemplate(), workflow_templates.WorkflowTemplate()], next_page_token='abc'), workflow_templates.ListWorkflowTemplatesResponse(templates=[], next_page_token='def'), workflow_templates.ListWorkflowTemplatesResponse(templates=[workflow_templates.WorkflowTemplate()], next_page_token='ghi'), workflow_templates.ListWorkflowTemplatesResponse(templates=[workflow_templates.WorkflowTemplate(), workflow_templates.WorkflowTemplate()]), RuntimeError)
        async_pager = await client.list_workflow_templates(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, workflow_templates.WorkflowTemplate) for i in responses))

@pytest.mark.asyncio
async def test_list_workflow_templates_async_pages():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_workflow_templates), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (workflow_templates.ListWorkflowTemplatesResponse(templates=[workflow_templates.WorkflowTemplate(), workflow_templates.WorkflowTemplate(), workflow_templates.WorkflowTemplate()], next_page_token='abc'), workflow_templates.ListWorkflowTemplatesResponse(templates=[], next_page_token='def'), workflow_templates.ListWorkflowTemplatesResponse(templates=[workflow_templates.WorkflowTemplate()], next_page_token='ghi'), workflow_templates.ListWorkflowTemplatesResponse(templates=[workflow_templates.WorkflowTemplate(), workflow_templates.WorkflowTemplate()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_workflow_templates(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [workflow_templates.DeleteWorkflowTemplateRequest, dict])
def test_delete_workflow_template(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_workflow_template), '__call__') as call:
        call.return_value = None
        response = client.delete_workflow_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.DeleteWorkflowTemplateRequest()
    assert response is None

def test_delete_workflow_template_empty_call():
    if False:
        print('Hello World!')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_workflow_template), '__call__') as call:
        client.delete_workflow_template()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.DeleteWorkflowTemplateRequest()

@pytest.mark.asyncio
async def test_delete_workflow_template_async(transport: str='grpc_asyncio', request_type=workflow_templates.DeleteWorkflowTemplateRequest):
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_workflow_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_workflow_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == workflow_templates.DeleteWorkflowTemplateRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_workflow_template_async_from_dict():
    await test_delete_workflow_template_async(request_type=dict)

def test_delete_workflow_template_field_headers():
    if False:
        while True:
            i = 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = workflow_templates.DeleteWorkflowTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_workflow_template), '__call__') as call:
        call.return_value = None
        client.delete_workflow_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_workflow_template_field_headers_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = workflow_templates.DeleteWorkflowTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_workflow_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_workflow_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_workflow_template_flattened():
    if False:
        print('Hello World!')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_workflow_template), '__call__') as call:
        call.return_value = None
        client.delete_workflow_template(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_workflow_template_flattened_error():
    if False:
        print('Hello World!')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_workflow_template(workflow_templates.DeleteWorkflowTemplateRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_workflow_template_flattened_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_workflow_template), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_workflow_template(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_workflow_template_flattened_error_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_workflow_template(workflow_templates.DeleteWorkflowTemplateRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [workflow_templates.CreateWorkflowTemplateRequest, dict])
def test_create_workflow_template_rest(request_type):
    if False:
        while True:
            i = 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['template'] = {'id': 'id_value', 'name': 'name_value', 'version': 774, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'placement': {'managed_cluster': {'cluster_name': 'cluster_name_value', 'config': {'config_bucket': 'config_bucket_value', 'temp_bucket': 'temp_bucket_value', 'gce_cluster_config': {'zone_uri': 'zone_uri_value', 'network_uri': 'network_uri_value', 'subnetwork_uri': 'subnetwork_uri_value', 'internal_ip_only': True, 'private_ipv6_google_access': 1, 'service_account': 'service_account_value', 'service_account_scopes': ['service_account_scopes_value1', 'service_account_scopes_value2'], 'tags': ['tags_value1', 'tags_value2'], 'metadata': {}, 'reservation_affinity': {'consume_reservation_type': 1, 'key': 'key_value', 'values': ['values_value1', 'values_value2']}, 'node_group_affinity': {'node_group_uri': 'node_group_uri_value'}, 'shielded_instance_config': {'enable_secure_boot': True, 'enable_vtpm': True, 'enable_integrity_monitoring': True}, 'confidential_instance_config': {'enable_confidential_compute': True}}, 'master_config': {'num_instances': 1399, 'instance_names': ['instance_names_value1', 'instance_names_value2'], 'instance_references': [{'instance_name': 'instance_name_value', 'instance_id': 'instance_id_value', 'public_key': 'public_key_value', 'public_ecies_key': 'public_ecies_key_value'}], 'image_uri': 'image_uri_value', 'machine_type_uri': 'machine_type_uri_value', 'disk_config': {'boot_disk_type': 'boot_disk_type_value', 'boot_disk_size_gb': 1792, 'num_local_ssds': 1494, 'local_ssd_interface': 'local_ssd_interface_value'}, 'is_preemptible': True, 'preemptibility': 1, 'managed_group_config': {'instance_template_name': 'instance_template_name_value', 'instance_group_manager_name': 'instance_group_manager_name_value', 'instance_group_manager_uri': 'instance_group_manager_uri_value'}, 'accelerators': [{'accelerator_type_uri': 'accelerator_type_uri_value', 'accelerator_count': 1805}], 'min_cpu_platform': 'min_cpu_platform_value', 'min_num_instances': 1818, 'instance_flexibility_policy': {'instance_selection_list': [{'machine_types': ['machine_types_value1', 'machine_types_value2'], 'rank': 428}], 'instance_selection_results': [{'machine_type': 'machine_type_value', 'vm_count': 875}]}, 'startup_config': {'required_registration_fraction': 0.3216}}, 'worker_config': {}, 'secondary_worker_config': {}, 'software_config': {'image_version': 'image_version_value', 'properties': {}, 'optional_components': [5]}, 'initialization_actions': [{'executable_file': 'executable_file_value', 'execution_timeout': {'seconds': 751, 'nanos': 543}}], 'encryption_config': {'gce_pd_kms_key_name': 'gce_pd_kms_key_name_value'}, 'autoscaling_config': {'policy_uri': 'policy_uri_value'}, 'security_config': {'kerberos_config': {'enable_kerberos': True, 'root_principal_password_uri': 'root_principal_password_uri_value', 'kms_key_uri': 'kms_key_uri_value', 'keystore_uri': 'keystore_uri_value', 'truststore_uri': 'truststore_uri_value', 'keystore_password_uri': 'keystore_password_uri_value', 'key_password_uri': 'key_password_uri_value', 'truststore_password_uri': 'truststore_password_uri_value', 'cross_realm_trust_realm': 'cross_realm_trust_realm_value', 'cross_realm_trust_kdc': 'cross_realm_trust_kdc_value', 'cross_realm_trust_admin_server': 'cross_realm_trust_admin_server_value', 'cross_realm_trust_shared_password_uri': 'cross_realm_trust_shared_password_uri_value', 'kdc_db_key_uri': 'kdc_db_key_uri_value', 'tgt_lifetime_hours': 1933, 'realm': 'realm_value'}, 'identity_config': {'user_service_account_mapping': {}}}, 'lifecycle_config': {'idle_delete_ttl': {}, 'auto_delete_time': {}, 'auto_delete_ttl': {}, 'idle_start_time': {}}, 'endpoint_config': {'http_ports': {}, 'enable_http_port_access': True}, 'metastore_config': {'dataproc_metastore_service': 'dataproc_metastore_service_value'}, 'dataproc_metric_config': {'metrics': [{'metric_source': 1, 'metric_overrides': ['metric_overrides_value1', 'metric_overrides_value2']}]}, 'auxiliary_node_groups': [{'node_group': {'name': 'name_value', 'roles': [1], 'node_group_config': {}, 'labels': {}}, 'node_group_id': 'node_group_id_value'}]}, 'labels': {}}, 'cluster_selector': {'zone': 'zone_value', 'cluster_labels': {}}}, 'jobs': [{'step_id': 'step_id_value', 'hadoop_job': {'main_jar_file_uri': 'main_jar_file_uri_value', 'main_class': 'main_class_value', 'args': ['args_value1', 'args_value2'], 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'file_uris': ['file_uris_value1', 'file_uris_value2'], 'archive_uris': ['archive_uris_value1', 'archive_uris_value2'], 'properties': {}, 'logging_config': {'driver_log_levels': {}}}, 'spark_job': {'main_jar_file_uri': 'main_jar_file_uri_value', 'main_class': 'main_class_value', 'args': ['args_value1', 'args_value2'], 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'file_uris': ['file_uris_value1', 'file_uris_value2'], 'archive_uris': ['archive_uris_value1', 'archive_uris_value2'], 'properties': {}, 'logging_config': {}}, 'pyspark_job': {'main_python_file_uri': 'main_python_file_uri_value', 'args': ['args_value1', 'args_value2'], 'python_file_uris': ['python_file_uris_value1', 'python_file_uris_value2'], 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'file_uris': ['file_uris_value1', 'file_uris_value2'], 'archive_uris': ['archive_uris_value1', 'archive_uris_value2'], 'properties': {}, 'logging_config': {}}, 'hive_job': {'query_file_uri': 'query_file_uri_value', 'query_list': {'queries': ['queries_value1', 'queries_value2']}, 'continue_on_failure': True, 'script_variables': {}, 'properties': {}, 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2']}, 'pig_job': {'query_file_uri': 'query_file_uri_value', 'query_list': {}, 'continue_on_failure': True, 'script_variables': {}, 'properties': {}, 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'logging_config': {}}, 'spark_r_job': {'main_r_file_uri': 'main_r_file_uri_value', 'args': ['args_value1', 'args_value2'], 'file_uris': ['file_uris_value1', 'file_uris_value2'], 'archive_uris': ['archive_uris_value1', 'archive_uris_value2'], 'properties': {}, 'logging_config': {}}, 'spark_sql_job': {'query_file_uri': 'query_file_uri_value', 'query_list': {}, 'script_variables': {}, 'properties': {}, 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'logging_config': {}}, 'presto_job': {'query_file_uri': 'query_file_uri_value', 'query_list': {}, 'continue_on_failure': True, 'output_format': 'output_format_value', 'client_tags': ['client_tags_value1', 'client_tags_value2'], 'properties': {}, 'logging_config': {}}, 'labels': {}, 'scheduling': {'max_failures_per_hour': 2243, 'max_failures_total': 1923}, 'prerequisite_step_ids': ['prerequisite_step_ids_value1', 'prerequisite_step_ids_value2']}], 'parameters': [{'name': 'name_value', 'fields': ['fields_value1', 'fields_value2'], 'description': 'description_value', 'validation': {'regex': {'regexes': ['regexes_value1', 'regexes_value2']}, 'values': {'values': ['values_value1', 'values_value2']}}}], 'dag_timeout': {}}
    test_field = workflow_templates.CreateWorkflowTemplateRequest.meta.fields['template']

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
    for (field, value) in request_init['template'].items():
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
                for i in range(0, len(request_init['template'][field])):
                    del request_init['template'][field][i][subfield]
            else:
                del request_init['template'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workflow_templates.WorkflowTemplate(id='id_value', name='name_value', version=774)
        response_value = Response()
        response_value.status_code = 200
        return_value = workflow_templates.WorkflowTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_workflow_template(request)
    assert isinstance(response, workflow_templates.WorkflowTemplate)
    assert response.id == 'id_value'
    assert response.name == 'name_value'
    assert response.version == 774

def test_create_workflow_template_rest_required_fields(request_type=workflow_templates.CreateWorkflowTemplateRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.WorkflowTemplateServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_workflow_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_workflow_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = workflow_templates.WorkflowTemplate()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = workflow_templates.WorkflowTemplate.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_workflow_template(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_workflow_template_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.WorkflowTemplateServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_workflow_template._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'template'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_workflow_template_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.WorkflowTemplateServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkflowTemplateServiceRestInterceptor())
    client = WorkflowTemplateServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WorkflowTemplateServiceRestInterceptor, 'post_create_workflow_template') as post, mock.patch.object(transports.WorkflowTemplateServiceRestInterceptor, 'pre_create_workflow_template') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workflow_templates.CreateWorkflowTemplateRequest.pb(workflow_templates.CreateWorkflowTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = workflow_templates.WorkflowTemplate.to_json(workflow_templates.WorkflowTemplate())
        request = workflow_templates.CreateWorkflowTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = workflow_templates.WorkflowTemplate()
        client.create_workflow_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_workflow_template_rest_bad_request(transport: str='rest', request_type=workflow_templates.CreateWorkflowTemplateRequest):
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_workflow_template(request)

def test_create_workflow_template_rest_flattened():
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workflow_templates.WorkflowTemplate()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', template=workflow_templates.WorkflowTemplate(id='id_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = workflow_templates.WorkflowTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_workflow_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/workflowTemplates' % client.transport._host, args[1])

def test_create_workflow_template_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_workflow_template(workflow_templates.CreateWorkflowTemplateRequest(), parent='parent_value', template=workflow_templates.WorkflowTemplate(id='id_value'))

def test_create_workflow_template_rest_error():
    if False:
        print('Hello World!')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workflow_templates.GetWorkflowTemplateRequest, dict])
def test_get_workflow_template_rest(request_type):
    if False:
        while True:
            i = 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/workflowTemplates/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workflow_templates.WorkflowTemplate(id='id_value', name='name_value', version=774)
        response_value = Response()
        response_value.status_code = 200
        return_value = workflow_templates.WorkflowTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_workflow_template(request)
    assert isinstance(response, workflow_templates.WorkflowTemplate)
    assert response.id == 'id_value'
    assert response.name == 'name_value'
    assert response.version == 774

def test_get_workflow_template_rest_required_fields(request_type=workflow_templates.GetWorkflowTemplateRequest):
    if False:
        print('Hello World!')
    transport_class = transports.WorkflowTemplateServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_workflow_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_workflow_template._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('version',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = workflow_templates.WorkflowTemplate()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = workflow_templates.WorkflowTemplate.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_workflow_template(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_workflow_template_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.WorkflowTemplateServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_workflow_template._get_unset_required_fields({})
    assert set(unset_fields) == set(('version',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_workflow_template_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.WorkflowTemplateServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkflowTemplateServiceRestInterceptor())
    client = WorkflowTemplateServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WorkflowTemplateServiceRestInterceptor, 'post_get_workflow_template') as post, mock.patch.object(transports.WorkflowTemplateServiceRestInterceptor, 'pre_get_workflow_template') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workflow_templates.GetWorkflowTemplateRequest.pb(workflow_templates.GetWorkflowTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = workflow_templates.WorkflowTemplate.to_json(workflow_templates.WorkflowTemplate())
        request = workflow_templates.GetWorkflowTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = workflow_templates.WorkflowTemplate()
        client.get_workflow_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_workflow_template_rest_bad_request(transport: str='rest', request_type=workflow_templates.GetWorkflowTemplateRequest):
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/workflowTemplates/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_workflow_template(request)

def test_get_workflow_template_rest_flattened():
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workflow_templates.WorkflowTemplate()
        sample_request = {'name': 'projects/sample1/locations/sample2/workflowTemplates/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = workflow_templates.WorkflowTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_workflow_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/workflowTemplates/*}' % client.transport._host, args[1])

def test_get_workflow_template_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_workflow_template(workflow_templates.GetWorkflowTemplateRequest(), name='name_value')

def test_get_workflow_template_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workflow_templates.InstantiateWorkflowTemplateRequest, dict])
def test_instantiate_workflow_template_rest(request_type):
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/workflowTemplates/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.instantiate_workflow_template(request)
    assert response.operation.name == 'operations/spam'

def test_instantiate_workflow_template_rest_required_fields(request_type=workflow_templates.InstantiateWorkflowTemplateRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.WorkflowTemplateServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).instantiate_workflow_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).instantiate_workflow_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.instantiate_workflow_template(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_instantiate_workflow_template_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.WorkflowTemplateServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.instantiate_workflow_template._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_instantiate_workflow_template_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.WorkflowTemplateServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkflowTemplateServiceRestInterceptor())
    client = WorkflowTemplateServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.WorkflowTemplateServiceRestInterceptor, 'post_instantiate_workflow_template') as post, mock.patch.object(transports.WorkflowTemplateServiceRestInterceptor, 'pre_instantiate_workflow_template') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workflow_templates.InstantiateWorkflowTemplateRequest.pb(workflow_templates.InstantiateWorkflowTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = workflow_templates.InstantiateWorkflowTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.instantiate_workflow_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_instantiate_workflow_template_rest_bad_request(transport: str='rest', request_type=workflow_templates.InstantiateWorkflowTemplateRequest):
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/workflowTemplates/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.instantiate_workflow_template(request)

def test_instantiate_workflow_template_rest_flattened():
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/workflowTemplates/sample3'}
        mock_args = dict(name='name_value', parameters={'key_value': 'value_value'})
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.instantiate_workflow_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/workflowTemplates/*}:instantiate' % client.transport._host, args[1])

def test_instantiate_workflow_template_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.instantiate_workflow_template(workflow_templates.InstantiateWorkflowTemplateRequest(), name='name_value', parameters={'key_value': 'value_value'})

def test_instantiate_workflow_template_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workflow_templates.InstantiateInlineWorkflowTemplateRequest, dict])
def test_instantiate_inline_workflow_template_rest(request_type):
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['template'] = {'id': 'id_value', 'name': 'name_value', 'version': 774, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'placement': {'managed_cluster': {'cluster_name': 'cluster_name_value', 'config': {'config_bucket': 'config_bucket_value', 'temp_bucket': 'temp_bucket_value', 'gce_cluster_config': {'zone_uri': 'zone_uri_value', 'network_uri': 'network_uri_value', 'subnetwork_uri': 'subnetwork_uri_value', 'internal_ip_only': True, 'private_ipv6_google_access': 1, 'service_account': 'service_account_value', 'service_account_scopes': ['service_account_scopes_value1', 'service_account_scopes_value2'], 'tags': ['tags_value1', 'tags_value2'], 'metadata': {}, 'reservation_affinity': {'consume_reservation_type': 1, 'key': 'key_value', 'values': ['values_value1', 'values_value2']}, 'node_group_affinity': {'node_group_uri': 'node_group_uri_value'}, 'shielded_instance_config': {'enable_secure_boot': True, 'enable_vtpm': True, 'enable_integrity_monitoring': True}, 'confidential_instance_config': {'enable_confidential_compute': True}}, 'master_config': {'num_instances': 1399, 'instance_names': ['instance_names_value1', 'instance_names_value2'], 'instance_references': [{'instance_name': 'instance_name_value', 'instance_id': 'instance_id_value', 'public_key': 'public_key_value', 'public_ecies_key': 'public_ecies_key_value'}], 'image_uri': 'image_uri_value', 'machine_type_uri': 'machine_type_uri_value', 'disk_config': {'boot_disk_type': 'boot_disk_type_value', 'boot_disk_size_gb': 1792, 'num_local_ssds': 1494, 'local_ssd_interface': 'local_ssd_interface_value'}, 'is_preemptible': True, 'preemptibility': 1, 'managed_group_config': {'instance_template_name': 'instance_template_name_value', 'instance_group_manager_name': 'instance_group_manager_name_value', 'instance_group_manager_uri': 'instance_group_manager_uri_value'}, 'accelerators': [{'accelerator_type_uri': 'accelerator_type_uri_value', 'accelerator_count': 1805}], 'min_cpu_platform': 'min_cpu_platform_value', 'min_num_instances': 1818, 'instance_flexibility_policy': {'instance_selection_list': [{'machine_types': ['machine_types_value1', 'machine_types_value2'], 'rank': 428}], 'instance_selection_results': [{'machine_type': 'machine_type_value', 'vm_count': 875}]}, 'startup_config': {'required_registration_fraction': 0.3216}}, 'worker_config': {}, 'secondary_worker_config': {}, 'software_config': {'image_version': 'image_version_value', 'properties': {}, 'optional_components': [5]}, 'initialization_actions': [{'executable_file': 'executable_file_value', 'execution_timeout': {'seconds': 751, 'nanos': 543}}], 'encryption_config': {'gce_pd_kms_key_name': 'gce_pd_kms_key_name_value'}, 'autoscaling_config': {'policy_uri': 'policy_uri_value'}, 'security_config': {'kerberos_config': {'enable_kerberos': True, 'root_principal_password_uri': 'root_principal_password_uri_value', 'kms_key_uri': 'kms_key_uri_value', 'keystore_uri': 'keystore_uri_value', 'truststore_uri': 'truststore_uri_value', 'keystore_password_uri': 'keystore_password_uri_value', 'key_password_uri': 'key_password_uri_value', 'truststore_password_uri': 'truststore_password_uri_value', 'cross_realm_trust_realm': 'cross_realm_trust_realm_value', 'cross_realm_trust_kdc': 'cross_realm_trust_kdc_value', 'cross_realm_trust_admin_server': 'cross_realm_trust_admin_server_value', 'cross_realm_trust_shared_password_uri': 'cross_realm_trust_shared_password_uri_value', 'kdc_db_key_uri': 'kdc_db_key_uri_value', 'tgt_lifetime_hours': 1933, 'realm': 'realm_value'}, 'identity_config': {'user_service_account_mapping': {}}}, 'lifecycle_config': {'idle_delete_ttl': {}, 'auto_delete_time': {}, 'auto_delete_ttl': {}, 'idle_start_time': {}}, 'endpoint_config': {'http_ports': {}, 'enable_http_port_access': True}, 'metastore_config': {'dataproc_metastore_service': 'dataproc_metastore_service_value'}, 'dataproc_metric_config': {'metrics': [{'metric_source': 1, 'metric_overrides': ['metric_overrides_value1', 'metric_overrides_value2']}]}, 'auxiliary_node_groups': [{'node_group': {'name': 'name_value', 'roles': [1], 'node_group_config': {}, 'labels': {}}, 'node_group_id': 'node_group_id_value'}]}, 'labels': {}}, 'cluster_selector': {'zone': 'zone_value', 'cluster_labels': {}}}, 'jobs': [{'step_id': 'step_id_value', 'hadoop_job': {'main_jar_file_uri': 'main_jar_file_uri_value', 'main_class': 'main_class_value', 'args': ['args_value1', 'args_value2'], 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'file_uris': ['file_uris_value1', 'file_uris_value2'], 'archive_uris': ['archive_uris_value1', 'archive_uris_value2'], 'properties': {}, 'logging_config': {'driver_log_levels': {}}}, 'spark_job': {'main_jar_file_uri': 'main_jar_file_uri_value', 'main_class': 'main_class_value', 'args': ['args_value1', 'args_value2'], 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'file_uris': ['file_uris_value1', 'file_uris_value2'], 'archive_uris': ['archive_uris_value1', 'archive_uris_value2'], 'properties': {}, 'logging_config': {}}, 'pyspark_job': {'main_python_file_uri': 'main_python_file_uri_value', 'args': ['args_value1', 'args_value2'], 'python_file_uris': ['python_file_uris_value1', 'python_file_uris_value2'], 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'file_uris': ['file_uris_value1', 'file_uris_value2'], 'archive_uris': ['archive_uris_value1', 'archive_uris_value2'], 'properties': {}, 'logging_config': {}}, 'hive_job': {'query_file_uri': 'query_file_uri_value', 'query_list': {'queries': ['queries_value1', 'queries_value2']}, 'continue_on_failure': True, 'script_variables': {}, 'properties': {}, 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2']}, 'pig_job': {'query_file_uri': 'query_file_uri_value', 'query_list': {}, 'continue_on_failure': True, 'script_variables': {}, 'properties': {}, 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'logging_config': {}}, 'spark_r_job': {'main_r_file_uri': 'main_r_file_uri_value', 'args': ['args_value1', 'args_value2'], 'file_uris': ['file_uris_value1', 'file_uris_value2'], 'archive_uris': ['archive_uris_value1', 'archive_uris_value2'], 'properties': {}, 'logging_config': {}}, 'spark_sql_job': {'query_file_uri': 'query_file_uri_value', 'query_list': {}, 'script_variables': {}, 'properties': {}, 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'logging_config': {}}, 'presto_job': {'query_file_uri': 'query_file_uri_value', 'query_list': {}, 'continue_on_failure': True, 'output_format': 'output_format_value', 'client_tags': ['client_tags_value1', 'client_tags_value2'], 'properties': {}, 'logging_config': {}}, 'labels': {}, 'scheduling': {'max_failures_per_hour': 2243, 'max_failures_total': 1923}, 'prerequisite_step_ids': ['prerequisite_step_ids_value1', 'prerequisite_step_ids_value2']}], 'parameters': [{'name': 'name_value', 'fields': ['fields_value1', 'fields_value2'], 'description': 'description_value', 'validation': {'regex': {'regexes': ['regexes_value1', 'regexes_value2']}, 'values': {'values': ['values_value1', 'values_value2']}}}], 'dag_timeout': {}}
    test_field = workflow_templates.InstantiateInlineWorkflowTemplateRequest.meta.fields['template']

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
    for (field, value) in request_init['template'].items():
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
                for i in range(0, len(request_init['template'][field])):
                    del request_init['template'][field][i][subfield]
            else:
                del request_init['template'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.instantiate_inline_workflow_template(request)
    assert response.operation.name == 'operations/spam'

def test_instantiate_inline_workflow_template_rest_required_fields(request_type=workflow_templates.InstantiateInlineWorkflowTemplateRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.WorkflowTemplateServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).instantiate_inline_workflow_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).instantiate_inline_workflow_template._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.instantiate_inline_workflow_template(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_instantiate_inline_workflow_template_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.WorkflowTemplateServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.instantiate_inline_workflow_template._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('parent', 'template'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_instantiate_inline_workflow_template_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.WorkflowTemplateServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkflowTemplateServiceRestInterceptor())
    client = WorkflowTemplateServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.WorkflowTemplateServiceRestInterceptor, 'post_instantiate_inline_workflow_template') as post, mock.patch.object(transports.WorkflowTemplateServiceRestInterceptor, 'pre_instantiate_inline_workflow_template') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workflow_templates.InstantiateInlineWorkflowTemplateRequest.pb(workflow_templates.InstantiateInlineWorkflowTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = workflow_templates.InstantiateInlineWorkflowTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.instantiate_inline_workflow_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_instantiate_inline_workflow_template_rest_bad_request(transport: str='rest', request_type=workflow_templates.InstantiateInlineWorkflowTemplateRequest):
    if False:
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.instantiate_inline_workflow_template(request)

def test_instantiate_inline_workflow_template_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', template=workflow_templates.WorkflowTemplate(id='id_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.instantiate_inline_workflow_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/workflowTemplates:instantiateInline' % client.transport._host, args[1])

def test_instantiate_inline_workflow_template_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.instantiate_inline_workflow_template(workflow_templates.InstantiateInlineWorkflowTemplateRequest(), parent='parent_value', template=workflow_templates.WorkflowTemplate(id='id_value'))

def test_instantiate_inline_workflow_template_rest_error():
    if False:
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workflow_templates.UpdateWorkflowTemplateRequest, dict])
def test_update_workflow_template_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'template': {'name': 'projects/sample1/locations/sample2/workflowTemplates/sample3'}}
    request_init['template'] = {'id': 'id_value', 'name': 'projects/sample1/locations/sample2/workflowTemplates/sample3', 'version': 774, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'placement': {'managed_cluster': {'cluster_name': 'cluster_name_value', 'config': {'config_bucket': 'config_bucket_value', 'temp_bucket': 'temp_bucket_value', 'gce_cluster_config': {'zone_uri': 'zone_uri_value', 'network_uri': 'network_uri_value', 'subnetwork_uri': 'subnetwork_uri_value', 'internal_ip_only': True, 'private_ipv6_google_access': 1, 'service_account': 'service_account_value', 'service_account_scopes': ['service_account_scopes_value1', 'service_account_scopes_value2'], 'tags': ['tags_value1', 'tags_value2'], 'metadata': {}, 'reservation_affinity': {'consume_reservation_type': 1, 'key': 'key_value', 'values': ['values_value1', 'values_value2']}, 'node_group_affinity': {'node_group_uri': 'node_group_uri_value'}, 'shielded_instance_config': {'enable_secure_boot': True, 'enable_vtpm': True, 'enable_integrity_monitoring': True}, 'confidential_instance_config': {'enable_confidential_compute': True}}, 'master_config': {'num_instances': 1399, 'instance_names': ['instance_names_value1', 'instance_names_value2'], 'instance_references': [{'instance_name': 'instance_name_value', 'instance_id': 'instance_id_value', 'public_key': 'public_key_value', 'public_ecies_key': 'public_ecies_key_value'}], 'image_uri': 'image_uri_value', 'machine_type_uri': 'machine_type_uri_value', 'disk_config': {'boot_disk_type': 'boot_disk_type_value', 'boot_disk_size_gb': 1792, 'num_local_ssds': 1494, 'local_ssd_interface': 'local_ssd_interface_value'}, 'is_preemptible': True, 'preemptibility': 1, 'managed_group_config': {'instance_template_name': 'instance_template_name_value', 'instance_group_manager_name': 'instance_group_manager_name_value', 'instance_group_manager_uri': 'instance_group_manager_uri_value'}, 'accelerators': [{'accelerator_type_uri': 'accelerator_type_uri_value', 'accelerator_count': 1805}], 'min_cpu_platform': 'min_cpu_platform_value', 'min_num_instances': 1818, 'instance_flexibility_policy': {'instance_selection_list': [{'machine_types': ['machine_types_value1', 'machine_types_value2'], 'rank': 428}], 'instance_selection_results': [{'machine_type': 'machine_type_value', 'vm_count': 875}]}, 'startup_config': {'required_registration_fraction': 0.3216}}, 'worker_config': {}, 'secondary_worker_config': {}, 'software_config': {'image_version': 'image_version_value', 'properties': {}, 'optional_components': [5]}, 'initialization_actions': [{'executable_file': 'executable_file_value', 'execution_timeout': {'seconds': 751, 'nanos': 543}}], 'encryption_config': {'gce_pd_kms_key_name': 'gce_pd_kms_key_name_value'}, 'autoscaling_config': {'policy_uri': 'policy_uri_value'}, 'security_config': {'kerberos_config': {'enable_kerberos': True, 'root_principal_password_uri': 'root_principal_password_uri_value', 'kms_key_uri': 'kms_key_uri_value', 'keystore_uri': 'keystore_uri_value', 'truststore_uri': 'truststore_uri_value', 'keystore_password_uri': 'keystore_password_uri_value', 'key_password_uri': 'key_password_uri_value', 'truststore_password_uri': 'truststore_password_uri_value', 'cross_realm_trust_realm': 'cross_realm_trust_realm_value', 'cross_realm_trust_kdc': 'cross_realm_trust_kdc_value', 'cross_realm_trust_admin_server': 'cross_realm_trust_admin_server_value', 'cross_realm_trust_shared_password_uri': 'cross_realm_trust_shared_password_uri_value', 'kdc_db_key_uri': 'kdc_db_key_uri_value', 'tgt_lifetime_hours': 1933, 'realm': 'realm_value'}, 'identity_config': {'user_service_account_mapping': {}}}, 'lifecycle_config': {'idle_delete_ttl': {}, 'auto_delete_time': {}, 'auto_delete_ttl': {}, 'idle_start_time': {}}, 'endpoint_config': {'http_ports': {}, 'enable_http_port_access': True}, 'metastore_config': {'dataproc_metastore_service': 'dataproc_metastore_service_value'}, 'dataproc_metric_config': {'metrics': [{'metric_source': 1, 'metric_overrides': ['metric_overrides_value1', 'metric_overrides_value2']}]}, 'auxiliary_node_groups': [{'node_group': {'name': 'name_value', 'roles': [1], 'node_group_config': {}, 'labels': {}}, 'node_group_id': 'node_group_id_value'}]}, 'labels': {}}, 'cluster_selector': {'zone': 'zone_value', 'cluster_labels': {}}}, 'jobs': [{'step_id': 'step_id_value', 'hadoop_job': {'main_jar_file_uri': 'main_jar_file_uri_value', 'main_class': 'main_class_value', 'args': ['args_value1', 'args_value2'], 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'file_uris': ['file_uris_value1', 'file_uris_value2'], 'archive_uris': ['archive_uris_value1', 'archive_uris_value2'], 'properties': {}, 'logging_config': {'driver_log_levels': {}}}, 'spark_job': {'main_jar_file_uri': 'main_jar_file_uri_value', 'main_class': 'main_class_value', 'args': ['args_value1', 'args_value2'], 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'file_uris': ['file_uris_value1', 'file_uris_value2'], 'archive_uris': ['archive_uris_value1', 'archive_uris_value2'], 'properties': {}, 'logging_config': {}}, 'pyspark_job': {'main_python_file_uri': 'main_python_file_uri_value', 'args': ['args_value1', 'args_value2'], 'python_file_uris': ['python_file_uris_value1', 'python_file_uris_value2'], 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'file_uris': ['file_uris_value1', 'file_uris_value2'], 'archive_uris': ['archive_uris_value1', 'archive_uris_value2'], 'properties': {}, 'logging_config': {}}, 'hive_job': {'query_file_uri': 'query_file_uri_value', 'query_list': {'queries': ['queries_value1', 'queries_value2']}, 'continue_on_failure': True, 'script_variables': {}, 'properties': {}, 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2']}, 'pig_job': {'query_file_uri': 'query_file_uri_value', 'query_list': {}, 'continue_on_failure': True, 'script_variables': {}, 'properties': {}, 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'logging_config': {}}, 'spark_r_job': {'main_r_file_uri': 'main_r_file_uri_value', 'args': ['args_value1', 'args_value2'], 'file_uris': ['file_uris_value1', 'file_uris_value2'], 'archive_uris': ['archive_uris_value1', 'archive_uris_value2'], 'properties': {}, 'logging_config': {}}, 'spark_sql_job': {'query_file_uri': 'query_file_uri_value', 'query_list': {}, 'script_variables': {}, 'properties': {}, 'jar_file_uris': ['jar_file_uris_value1', 'jar_file_uris_value2'], 'logging_config': {}}, 'presto_job': {'query_file_uri': 'query_file_uri_value', 'query_list': {}, 'continue_on_failure': True, 'output_format': 'output_format_value', 'client_tags': ['client_tags_value1', 'client_tags_value2'], 'properties': {}, 'logging_config': {}}, 'labels': {}, 'scheduling': {'max_failures_per_hour': 2243, 'max_failures_total': 1923}, 'prerequisite_step_ids': ['prerequisite_step_ids_value1', 'prerequisite_step_ids_value2']}], 'parameters': [{'name': 'name_value', 'fields': ['fields_value1', 'fields_value2'], 'description': 'description_value', 'validation': {'regex': {'regexes': ['regexes_value1', 'regexes_value2']}, 'values': {'values': ['values_value1', 'values_value2']}}}], 'dag_timeout': {}}
    test_field = workflow_templates.UpdateWorkflowTemplateRequest.meta.fields['template']

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
    for (field, value) in request_init['template'].items():
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
                for i in range(0, len(request_init['template'][field])):
                    del request_init['template'][field][i][subfield]
            else:
                del request_init['template'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workflow_templates.WorkflowTemplate(id='id_value', name='name_value', version=774)
        response_value = Response()
        response_value.status_code = 200
        return_value = workflow_templates.WorkflowTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_workflow_template(request)
    assert isinstance(response, workflow_templates.WorkflowTemplate)
    assert response.id == 'id_value'
    assert response.name == 'name_value'
    assert response.version == 774

def test_update_workflow_template_rest_required_fields(request_type=workflow_templates.UpdateWorkflowTemplateRequest):
    if False:
        print('Hello World!')
    transport_class = transports.WorkflowTemplateServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_workflow_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_workflow_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = workflow_templates.WorkflowTemplate()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'put', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = workflow_templates.WorkflowTemplate.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_workflow_template(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_workflow_template_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.WorkflowTemplateServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_workflow_template._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('template',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_workflow_template_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.WorkflowTemplateServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkflowTemplateServiceRestInterceptor())
    client = WorkflowTemplateServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WorkflowTemplateServiceRestInterceptor, 'post_update_workflow_template') as post, mock.patch.object(transports.WorkflowTemplateServiceRestInterceptor, 'pre_update_workflow_template') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workflow_templates.UpdateWorkflowTemplateRequest.pb(workflow_templates.UpdateWorkflowTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = workflow_templates.WorkflowTemplate.to_json(workflow_templates.WorkflowTemplate())
        request = workflow_templates.UpdateWorkflowTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = workflow_templates.WorkflowTemplate()
        client.update_workflow_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_workflow_template_rest_bad_request(transport: str='rest', request_type=workflow_templates.UpdateWorkflowTemplateRequest):
    if False:
        print('Hello World!')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'template': {'name': 'projects/sample1/locations/sample2/workflowTemplates/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_workflow_template(request)

def test_update_workflow_template_rest_flattened():
    if False:
        print('Hello World!')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workflow_templates.WorkflowTemplate()
        sample_request = {'template': {'name': 'projects/sample1/locations/sample2/workflowTemplates/sample3'}}
        mock_args = dict(template=workflow_templates.WorkflowTemplate(id='id_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = workflow_templates.WorkflowTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_workflow_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{template.name=projects/*/locations/*/workflowTemplates/*}' % client.transport._host, args[1])

def test_update_workflow_template_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_workflow_template(workflow_templates.UpdateWorkflowTemplateRequest(), template=workflow_templates.WorkflowTemplate(id='id_value'))

def test_update_workflow_template_rest_error():
    if False:
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [workflow_templates.ListWorkflowTemplatesRequest, dict])
def test_list_workflow_templates_rest(request_type):
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workflow_templates.ListWorkflowTemplatesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = workflow_templates.ListWorkflowTemplatesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_workflow_templates(request)
    assert isinstance(response, pagers.ListWorkflowTemplatesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_workflow_templates_rest_required_fields(request_type=workflow_templates.ListWorkflowTemplatesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.WorkflowTemplateServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_workflow_templates._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_workflow_templates._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = workflow_templates.ListWorkflowTemplatesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = workflow_templates.ListWorkflowTemplatesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_workflow_templates(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_workflow_templates_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.WorkflowTemplateServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_workflow_templates._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_workflow_templates_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.WorkflowTemplateServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkflowTemplateServiceRestInterceptor())
    client = WorkflowTemplateServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WorkflowTemplateServiceRestInterceptor, 'post_list_workflow_templates') as post, mock.patch.object(transports.WorkflowTemplateServiceRestInterceptor, 'pre_list_workflow_templates') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = workflow_templates.ListWorkflowTemplatesRequest.pb(workflow_templates.ListWorkflowTemplatesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = workflow_templates.ListWorkflowTemplatesResponse.to_json(workflow_templates.ListWorkflowTemplatesResponse())
        request = workflow_templates.ListWorkflowTemplatesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = workflow_templates.ListWorkflowTemplatesResponse()
        client.list_workflow_templates(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_workflow_templates_rest_bad_request(transport: str='rest', request_type=workflow_templates.ListWorkflowTemplatesRequest):
    if False:
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_workflow_templates(request)

def test_list_workflow_templates_rest_flattened():
    if False:
        print('Hello World!')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = workflow_templates.ListWorkflowTemplatesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = workflow_templates.ListWorkflowTemplatesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_workflow_templates(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/workflowTemplates' % client.transport._host, args[1])

def test_list_workflow_templates_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_workflow_templates(workflow_templates.ListWorkflowTemplatesRequest(), parent='parent_value')

def test_list_workflow_templates_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (workflow_templates.ListWorkflowTemplatesResponse(templates=[workflow_templates.WorkflowTemplate(), workflow_templates.WorkflowTemplate(), workflow_templates.WorkflowTemplate()], next_page_token='abc'), workflow_templates.ListWorkflowTemplatesResponse(templates=[], next_page_token='def'), workflow_templates.ListWorkflowTemplatesResponse(templates=[workflow_templates.WorkflowTemplate()], next_page_token='ghi'), workflow_templates.ListWorkflowTemplatesResponse(templates=[workflow_templates.WorkflowTemplate(), workflow_templates.WorkflowTemplate()]))
        response = response + response
        response = tuple((workflow_templates.ListWorkflowTemplatesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_workflow_templates(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, workflow_templates.WorkflowTemplate) for i in results))
        pages = list(client.list_workflow_templates(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [workflow_templates.DeleteWorkflowTemplateRequest, dict])
def test_delete_workflow_template_rest(request_type):
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/workflowTemplates/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_workflow_template(request)
    assert response is None

def test_delete_workflow_template_rest_required_fields(request_type=workflow_templates.DeleteWorkflowTemplateRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.WorkflowTemplateServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_workflow_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_workflow_template._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('version',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_workflow_template(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_workflow_template_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.WorkflowTemplateServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_workflow_template._get_unset_required_fields({})
    assert set(unset_fields) == set(('version',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_workflow_template_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.WorkflowTemplateServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.WorkflowTemplateServiceRestInterceptor())
    client = WorkflowTemplateServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.WorkflowTemplateServiceRestInterceptor, 'pre_delete_workflow_template') as pre:
        pre.assert_not_called()
        pb_message = workflow_templates.DeleteWorkflowTemplateRequest.pb(workflow_templates.DeleteWorkflowTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = workflow_templates.DeleteWorkflowTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_workflow_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_workflow_template_rest_bad_request(transport: str='rest', request_type=workflow_templates.DeleteWorkflowTemplateRequest):
    if False:
        for i in range(10):
            print('nop')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/workflowTemplates/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_workflow_template(request)

def test_delete_workflow_template_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/workflowTemplates/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_workflow_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/workflowTemplates/*}' % client.transport._host, args[1])

def test_delete_workflow_template_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_workflow_template(workflow_templates.DeleteWorkflowTemplateRequest(), name='name_value')

def test_delete_workflow_template_rest_error():
    if False:
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.WorkflowTemplateServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.WorkflowTemplateServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = WorkflowTemplateServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.WorkflowTemplateServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = WorkflowTemplateServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = WorkflowTemplateServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.WorkflowTemplateServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = WorkflowTemplateServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.WorkflowTemplateServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = WorkflowTemplateServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.WorkflowTemplateServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.WorkflowTemplateServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.WorkflowTemplateServiceGrpcTransport, transports.WorkflowTemplateServiceGrpcAsyncIOTransport, transports.WorkflowTemplateServiceRestTransport])
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
    transport = WorkflowTemplateServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.WorkflowTemplateServiceGrpcTransport)

def test_workflow_template_service_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.WorkflowTemplateServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_workflow_template_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.dataproc_v1.services.workflow_template_service.transports.WorkflowTemplateServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.WorkflowTemplateServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_workflow_template', 'get_workflow_template', 'instantiate_workflow_template', 'instantiate_inline_workflow_template', 'update_workflow_template', 'list_workflow_templates', 'delete_workflow_template', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_workflow_template_service_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.dataproc_v1.services.workflow_template_service.transports.WorkflowTemplateServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.WorkflowTemplateServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_workflow_template_service_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.dataproc_v1.services.workflow_template_service.transports.WorkflowTemplateServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.WorkflowTemplateServiceTransport()
        adc.assert_called_once()

def test_workflow_template_service_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        WorkflowTemplateServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.WorkflowTemplateServiceGrpcTransport, transports.WorkflowTemplateServiceGrpcAsyncIOTransport])
def test_workflow_template_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.WorkflowTemplateServiceGrpcTransport, transports.WorkflowTemplateServiceGrpcAsyncIOTransport, transports.WorkflowTemplateServiceRestTransport])
def test_workflow_template_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.WorkflowTemplateServiceGrpcTransport, grpc_helpers), (transports.WorkflowTemplateServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_workflow_template_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('dataproc.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='dataproc.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.WorkflowTemplateServiceGrpcTransport, transports.WorkflowTemplateServiceGrpcAsyncIOTransport])
def test_workflow_template_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_workflow_template_service_http_transport_client_cert_source_for_mtls():
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.WorkflowTemplateServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_workflow_template_service_rest_lro_client():
    if False:
        while True:
            i = 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_workflow_template_service_host_no_port(transport_name):
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dataproc.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('dataproc.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dataproc.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_workflow_template_service_host_with_port(transport_name):
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dataproc.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('dataproc.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dataproc.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_workflow_template_service_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = WorkflowTemplateServiceClient(credentials=creds1, transport=transport_name)
    client2 = WorkflowTemplateServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_workflow_template._session
    session2 = client2.transport.create_workflow_template._session
    assert session1 != session2
    session1 = client1.transport.get_workflow_template._session
    session2 = client2.transport.get_workflow_template._session
    assert session1 != session2
    session1 = client1.transport.instantiate_workflow_template._session
    session2 = client2.transport.instantiate_workflow_template._session
    assert session1 != session2
    session1 = client1.transport.instantiate_inline_workflow_template._session
    session2 = client2.transport.instantiate_inline_workflow_template._session
    assert session1 != session2
    session1 = client1.transport.update_workflow_template._session
    session2 = client2.transport.update_workflow_template._session
    assert session1 != session2
    session1 = client1.transport.list_workflow_templates._session
    session2 = client2.transport.list_workflow_templates._session
    assert session1 != session2
    session1 = client1.transport.delete_workflow_template._session
    session2 = client2.transport.delete_workflow_template._session
    assert session1 != session2

def test_workflow_template_service_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.WorkflowTemplateServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_workflow_template_service_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.WorkflowTemplateServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.WorkflowTemplateServiceGrpcTransport, transports.WorkflowTemplateServiceGrpcAsyncIOTransport])
def test_workflow_template_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.WorkflowTemplateServiceGrpcTransport, transports.WorkflowTemplateServiceGrpcAsyncIOTransport])
def test_workflow_template_service_transport_channel_mtls_with_adc(transport_class):
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

def test_workflow_template_service_grpc_lro_client():
    if False:
        while True:
            i = 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_workflow_template_service_grpc_lro_async_client():
    if False:
        return 10
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_node_group_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    region = 'clam'
    cluster = 'whelk'
    node_group = 'octopus'
    expected = 'projects/{project}/regions/{region}/clusters/{cluster}/nodeGroups/{node_group}'.format(project=project, region=region, cluster=cluster, node_group=node_group)
    actual = WorkflowTemplateServiceClient.node_group_path(project, region, cluster, node_group)
    assert expected == actual

def test_parse_node_group_path():
    if False:
        return 10
    expected = {'project': 'oyster', 'region': 'nudibranch', 'cluster': 'cuttlefish', 'node_group': 'mussel'}
    path = WorkflowTemplateServiceClient.node_group_path(**expected)
    actual = WorkflowTemplateServiceClient.parse_node_group_path(path)
    assert expected == actual

def test_service_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'winkle'
    location = 'nautilus'
    service = 'scallop'
    expected = 'projects/{project}/locations/{location}/services/{service}'.format(project=project, location=location, service=service)
    actual = WorkflowTemplateServiceClient.service_path(project, location, service)
    assert expected == actual

def test_parse_service_path():
    if False:
        return 10
    expected = {'project': 'abalone', 'location': 'squid', 'service': 'clam'}
    path = WorkflowTemplateServiceClient.service_path(**expected)
    actual = WorkflowTemplateServiceClient.parse_service_path(path)
    assert expected == actual

def test_workflow_template_path():
    if False:
        while True:
            i = 10
    project = 'whelk'
    region = 'octopus'
    workflow_template = 'oyster'
    expected = 'projects/{project}/regions/{region}/workflowTemplates/{workflow_template}'.format(project=project, region=region, workflow_template=workflow_template)
    actual = WorkflowTemplateServiceClient.workflow_template_path(project, region, workflow_template)
    assert expected == actual

def test_parse_workflow_template_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'nudibranch', 'region': 'cuttlefish', 'workflow_template': 'mussel'}
    path = WorkflowTemplateServiceClient.workflow_template_path(**expected)
    actual = WorkflowTemplateServiceClient.parse_workflow_template_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'winkle'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = WorkflowTemplateServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'nautilus'}
    path = WorkflowTemplateServiceClient.common_billing_account_path(**expected)
    actual = WorkflowTemplateServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'scallop'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = WorkflowTemplateServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'abalone'}
    path = WorkflowTemplateServiceClient.common_folder_path(**expected)
    actual = WorkflowTemplateServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'squid'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = WorkflowTemplateServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'clam'}
    path = WorkflowTemplateServiceClient.common_organization_path(**expected)
    actual = WorkflowTemplateServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'whelk'
    expected = 'projects/{project}'.format(project=project)
    actual = WorkflowTemplateServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'octopus'}
    path = WorkflowTemplateServiceClient.common_project_path(**expected)
    actual = WorkflowTemplateServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'oyster'
    location = 'nudibranch'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = WorkflowTemplateServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'cuttlefish', 'location': 'mussel'}
    path = WorkflowTemplateServiceClient.common_location_path(**expected)
    actual = WorkflowTemplateServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.WorkflowTemplateServiceTransport, '_prep_wrapped_messages') as prep:
        client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.WorkflowTemplateServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = WorkflowTemplateServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = WorkflowTemplateServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = WorkflowTemplateServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(WorkflowTemplateServiceClient, transports.WorkflowTemplateServiceGrpcTransport), (WorkflowTemplateServiceAsyncClient, transports.WorkflowTemplateServiceGrpcAsyncIOTransport)])
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