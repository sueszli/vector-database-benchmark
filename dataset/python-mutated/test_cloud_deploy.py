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
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.deploy_v1.services.cloud_deploy import CloudDeployAsyncClient, CloudDeployClient, pagers, transports
from google.cloud.deploy_v1.types import cloud_deploy

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
    assert CloudDeployClient._get_default_mtls_endpoint(None) is None
    assert CloudDeployClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert CloudDeployClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert CloudDeployClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert CloudDeployClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert CloudDeployClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(CloudDeployClient, 'grpc'), (CloudDeployAsyncClient, 'grpc_asyncio'), (CloudDeployClient, 'rest')])
def test_cloud_deploy_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('clouddeploy.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://clouddeploy.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.CloudDeployGrpcTransport, 'grpc'), (transports.CloudDeployGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.CloudDeployRestTransport, 'rest')])
def test_cloud_deploy_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(CloudDeployClient, 'grpc'), (CloudDeployAsyncClient, 'grpc_asyncio'), (CloudDeployClient, 'rest')])
def test_cloud_deploy_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('clouddeploy.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://clouddeploy.googleapis.com')

def test_cloud_deploy_client_get_transport_class():
    if False:
        return 10
    transport = CloudDeployClient.get_transport_class()
    available_transports = [transports.CloudDeployGrpcTransport, transports.CloudDeployRestTransport]
    assert transport in available_transports
    transport = CloudDeployClient.get_transport_class('grpc')
    assert transport == transports.CloudDeployGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CloudDeployClient, transports.CloudDeployGrpcTransport, 'grpc'), (CloudDeployAsyncClient, transports.CloudDeployGrpcAsyncIOTransport, 'grpc_asyncio'), (CloudDeployClient, transports.CloudDeployRestTransport, 'rest')])
@mock.patch.object(CloudDeployClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudDeployClient))
@mock.patch.object(CloudDeployAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudDeployAsyncClient))
def test_cloud_deploy_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(CloudDeployClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(CloudDeployClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(CloudDeployClient, transports.CloudDeployGrpcTransport, 'grpc', 'true'), (CloudDeployAsyncClient, transports.CloudDeployGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (CloudDeployClient, transports.CloudDeployGrpcTransport, 'grpc', 'false'), (CloudDeployAsyncClient, transports.CloudDeployGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (CloudDeployClient, transports.CloudDeployRestTransport, 'rest', 'true'), (CloudDeployClient, transports.CloudDeployRestTransport, 'rest', 'false')])
@mock.patch.object(CloudDeployClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudDeployClient))
@mock.patch.object(CloudDeployAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudDeployAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_cloud_deploy_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [CloudDeployClient, CloudDeployAsyncClient])
@mock.patch.object(CloudDeployClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudDeployClient))
@mock.patch.object(CloudDeployAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudDeployAsyncClient))
def test_cloud_deploy_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CloudDeployClient, transports.CloudDeployGrpcTransport, 'grpc'), (CloudDeployAsyncClient, transports.CloudDeployGrpcAsyncIOTransport, 'grpc_asyncio'), (CloudDeployClient, transports.CloudDeployRestTransport, 'rest')])
def test_cloud_deploy_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CloudDeployClient, transports.CloudDeployGrpcTransport, 'grpc', grpc_helpers), (CloudDeployAsyncClient, transports.CloudDeployGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (CloudDeployClient, transports.CloudDeployRestTransport, 'rest', None)])
def test_cloud_deploy_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_cloud_deploy_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.deploy_v1.services.cloud_deploy.transports.CloudDeployGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = CloudDeployClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CloudDeployClient, transports.CloudDeployGrpcTransport, 'grpc', grpc_helpers), (CloudDeployAsyncClient, transports.CloudDeployGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_cloud_deploy_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('clouddeploy.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='clouddeploy.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [cloud_deploy.ListDeliveryPipelinesRequest, dict])
def test_list_delivery_pipelines(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_delivery_pipelines), '__call__') as call:
        call.return_value = cloud_deploy.ListDeliveryPipelinesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_delivery_pipelines(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListDeliveryPipelinesRequest()
    assert isinstance(response, pagers.ListDeliveryPipelinesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_delivery_pipelines_empty_call():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_delivery_pipelines), '__call__') as call:
        client.list_delivery_pipelines()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListDeliveryPipelinesRequest()

@pytest.mark.asyncio
async def test_list_delivery_pipelines_async(transport: str='grpc_asyncio', request_type=cloud_deploy.ListDeliveryPipelinesRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_delivery_pipelines), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListDeliveryPipelinesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_delivery_pipelines(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListDeliveryPipelinesRequest()
    assert isinstance(response, pagers.ListDeliveryPipelinesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_delivery_pipelines_async_from_dict():
    await test_list_delivery_pipelines_async(request_type=dict)

def test_list_delivery_pipelines_field_headers():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.ListDeliveryPipelinesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_delivery_pipelines), '__call__') as call:
        call.return_value = cloud_deploy.ListDeliveryPipelinesResponse()
        client.list_delivery_pipelines(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_delivery_pipelines_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.ListDeliveryPipelinesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_delivery_pipelines), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListDeliveryPipelinesResponse())
        await client.list_delivery_pipelines(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_delivery_pipelines_flattened():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_delivery_pipelines), '__call__') as call:
        call.return_value = cloud_deploy.ListDeliveryPipelinesResponse()
        client.list_delivery_pipelines(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_delivery_pipelines_flattened_error():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_delivery_pipelines(cloud_deploy.ListDeliveryPipelinesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_delivery_pipelines_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_delivery_pipelines), '__call__') as call:
        call.return_value = cloud_deploy.ListDeliveryPipelinesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListDeliveryPipelinesResponse())
        response = await client.list_delivery_pipelines(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_delivery_pipelines_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_delivery_pipelines(cloud_deploy.ListDeliveryPipelinesRequest(), parent='parent_value')

def test_list_delivery_pipelines_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_delivery_pipelines), '__call__') as call:
        call.side_effect = (cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[cloud_deploy.DeliveryPipeline(), cloud_deploy.DeliveryPipeline(), cloud_deploy.DeliveryPipeline()], next_page_token='abc'), cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[], next_page_token='def'), cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[cloud_deploy.DeliveryPipeline()], next_page_token='ghi'), cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[cloud_deploy.DeliveryPipeline(), cloud_deploy.DeliveryPipeline()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_delivery_pipelines(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_deploy.DeliveryPipeline) for i in results))

def test_list_delivery_pipelines_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_delivery_pipelines), '__call__') as call:
        call.side_effect = (cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[cloud_deploy.DeliveryPipeline(), cloud_deploy.DeliveryPipeline(), cloud_deploy.DeliveryPipeline()], next_page_token='abc'), cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[], next_page_token='def'), cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[cloud_deploy.DeliveryPipeline()], next_page_token='ghi'), cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[cloud_deploy.DeliveryPipeline(), cloud_deploy.DeliveryPipeline()]), RuntimeError)
        pages = list(client.list_delivery_pipelines(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_delivery_pipelines_async_pager():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_delivery_pipelines), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[cloud_deploy.DeliveryPipeline(), cloud_deploy.DeliveryPipeline(), cloud_deploy.DeliveryPipeline()], next_page_token='abc'), cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[], next_page_token='def'), cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[cloud_deploy.DeliveryPipeline()], next_page_token='ghi'), cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[cloud_deploy.DeliveryPipeline(), cloud_deploy.DeliveryPipeline()]), RuntimeError)
        async_pager = await client.list_delivery_pipelines(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, cloud_deploy.DeliveryPipeline) for i in responses))

@pytest.mark.asyncio
async def test_list_delivery_pipelines_async_pages():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_delivery_pipelines), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[cloud_deploy.DeliveryPipeline(), cloud_deploy.DeliveryPipeline(), cloud_deploy.DeliveryPipeline()], next_page_token='abc'), cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[], next_page_token='def'), cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[cloud_deploy.DeliveryPipeline()], next_page_token='ghi'), cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[cloud_deploy.DeliveryPipeline(), cloud_deploy.DeliveryPipeline()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_delivery_pipelines(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_deploy.GetDeliveryPipelineRequest, dict])
def test_get_delivery_pipeline(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_delivery_pipeline), '__call__') as call:
        call.return_value = cloud_deploy.DeliveryPipeline(name='name_value', uid='uid_value', description='description_value', etag='etag_value', suspended=True)
        response = client.get_delivery_pipeline(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetDeliveryPipelineRequest()
    assert isinstance(response, cloud_deploy.DeliveryPipeline)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.etag == 'etag_value'
    assert response.suspended is True

def test_get_delivery_pipeline_empty_call():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_delivery_pipeline), '__call__') as call:
        client.get_delivery_pipeline()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetDeliveryPipelineRequest()

@pytest.mark.asyncio
async def test_get_delivery_pipeline_async(transport: str='grpc_asyncio', request_type=cloud_deploy.GetDeliveryPipelineRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_delivery_pipeline), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.DeliveryPipeline(name='name_value', uid='uid_value', description='description_value', etag='etag_value', suspended=True))
        response = await client.get_delivery_pipeline(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetDeliveryPipelineRequest()
    assert isinstance(response, cloud_deploy.DeliveryPipeline)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.etag == 'etag_value'
    assert response.suspended is True

@pytest.mark.asyncio
async def test_get_delivery_pipeline_async_from_dict():
    await test_get_delivery_pipeline_async(request_type=dict)

def test_get_delivery_pipeline_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.GetDeliveryPipelineRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_delivery_pipeline), '__call__') as call:
        call.return_value = cloud_deploy.DeliveryPipeline()
        client.get_delivery_pipeline(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_delivery_pipeline_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.GetDeliveryPipelineRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_delivery_pipeline), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.DeliveryPipeline())
        await client.get_delivery_pipeline(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_delivery_pipeline_flattened():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_delivery_pipeline), '__call__') as call:
        call.return_value = cloud_deploy.DeliveryPipeline()
        client.get_delivery_pipeline(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_delivery_pipeline_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_delivery_pipeline(cloud_deploy.GetDeliveryPipelineRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_delivery_pipeline_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_delivery_pipeline), '__call__') as call:
        call.return_value = cloud_deploy.DeliveryPipeline()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.DeliveryPipeline())
        response = await client.get_delivery_pipeline(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_delivery_pipeline_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_delivery_pipeline(cloud_deploy.GetDeliveryPipelineRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.CreateDeliveryPipelineRequest, dict])
def test_create_delivery_pipeline(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_delivery_pipeline), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_delivery_pipeline(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CreateDeliveryPipelineRequest()
    assert isinstance(response, future.Future)

def test_create_delivery_pipeline_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_delivery_pipeline), '__call__') as call:
        client.create_delivery_pipeline()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CreateDeliveryPipelineRequest()

@pytest.mark.asyncio
async def test_create_delivery_pipeline_async(transport: str='grpc_asyncio', request_type=cloud_deploy.CreateDeliveryPipelineRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_delivery_pipeline), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_delivery_pipeline(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CreateDeliveryPipelineRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_delivery_pipeline_async_from_dict():
    await test_create_delivery_pipeline_async(request_type=dict)

def test_create_delivery_pipeline_field_headers():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.CreateDeliveryPipelineRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_delivery_pipeline), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_delivery_pipeline(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_delivery_pipeline_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.CreateDeliveryPipelineRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_delivery_pipeline), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_delivery_pipeline(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_delivery_pipeline_flattened():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_delivery_pipeline), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_delivery_pipeline(parent='parent_value', delivery_pipeline=cloud_deploy.DeliveryPipeline(name='name_value'), delivery_pipeline_id='delivery_pipeline_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].delivery_pipeline
        mock_val = cloud_deploy.DeliveryPipeline(name='name_value')
        assert arg == mock_val
        arg = args[0].delivery_pipeline_id
        mock_val = 'delivery_pipeline_id_value'
        assert arg == mock_val

def test_create_delivery_pipeline_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_delivery_pipeline(cloud_deploy.CreateDeliveryPipelineRequest(), parent='parent_value', delivery_pipeline=cloud_deploy.DeliveryPipeline(name='name_value'), delivery_pipeline_id='delivery_pipeline_id_value')

@pytest.mark.asyncio
async def test_create_delivery_pipeline_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_delivery_pipeline), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_delivery_pipeline(parent='parent_value', delivery_pipeline=cloud_deploy.DeliveryPipeline(name='name_value'), delivery_pipeline_id='delivery_pipeline_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].delivery_pipeline
        mock_val = cloud_deploy.DeliveryPipeline(name='name_value')
        assert arg == mock_val
        arg = args[0].delivery_pipeline_id
        mock_val = 'delivery_pipeline_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_delivery_pipeline_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_delivery_pipeline(cloud_deploy.CreateDeliveryPipelineRequest(), parent='parent_value', delivery_pipeline=cloud_deploy.DeliveryPipeline(name='name_value'), delivery_pipeline_id='delivery_pipeline_id_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.UpdateDeliveryPipelineRequest, dict])
def test_update_delivery_pipeline(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_delivery_pipeline), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_delivery_pipeline(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.UpdateDeliveryPipelineRequest()
    assert isinstance(response, future.Future)

def test_update_delivery_pipeline_empty_call():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_delivery_pipeline), '__call__') as call:
        client.update_delivery_pipeline()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.UpdateDeliveryPipelineRequest()

@pytest.mark.asyncio
async def test_update_delivery_pipeline_async(transport: str='grpc_asyncio', request_type=cloud_deploy.UpdateDeliveryPipelineRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_delivery_pipeline), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_delivery_pipeline(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.UpdateDeliveryPipelineRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_delivery_pipeline_async_from_dict():
    await test_update_delivery_pipeline_async(request_type=dict)

def test_update_delivery_pipeline_field_headers():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.UpdateDeliveryPipelineRequest()
    request.delivery_pipeline.name = 'name_value'
    with mock.patch.object(type(client.transport.update_delivery_pipeline), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_delivery_pipeline(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'delivery_pipeline.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_delivery_pipeline_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.UpdateDeliveryPipelineRequest()
    request.delivery_pipeline.name = 'name_value'
    with mock.patch.object(type(client.transport.update_delivery_pipeline), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_delivery_pipeline(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'delivery_pipeline.name=name_value') in kw['metadata']

def test_update_delivery_pipeline_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_delivery_pipeline), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_delivery_pipeline(delivery_pipeline=cloud_deploy.DeliveryPipeline(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].delivery_pipeline
        mock_val = cloud_deploy.DeliveryPipeline(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_delivery_pipeline_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_delivery_pipeline(cloud_deploy.UpdateDeliveryPipelineRequest(), delivery_pipeline=cloud_deploy.DeliveryPipeline(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_delivery_pipeline_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_delivery_pipeline), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_delivery_pipeline(delivery_pipeline=cloud_deploy.DeliveryPipeline(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].delivery_pipeline
        mock_val = cloud_deploy.DeliveryPipeline(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_delivery_pipeline_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_delivery_pipeline(cloud_deploy.UpdateDeliveryPipelineRequest(), delivery_pipeline=cloud_deploy.DeliveryPipeline(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [cloud_deploy.DeleteDeliveryPipelineRequest, dict])
def test_delete_delivery_pipeline(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_delivery_pipeline), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_delivery_pipeline(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.DeleteDeliveryPipelineRequest()
    assert isinstance(response, future.Future)

def test_delete_delivery_pipeline_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_delivery_pipeline), '__call__') as call:
        client.delete_delivery_pipeline()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.DeleteDeliveryPipelineRequest()

@pytest.mark.asyncio
async def test_delete_delivery_pipeline_async(transport: str='grpc_asyncio', request_type=cloud_deploy.DeleteDeliveryPipelineRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_delivery_pipeline), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_delivery_pipeline(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.DeleteDeliveryPipelineRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_delivery_pipeline_async_from_dict():
    await test_delete_delivery_pipeline_async(request_type=dict)

def test_delete_delivery_pipeline_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.DeleteDeliveryPipelineRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_delivery_pipeline), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_delivery_pipeline(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_delivery_pipeline_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.DeleteDeliveryPipelineRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_delivery_pipeline), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_delivery_pipeline(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_delivery_pipeline_flattened():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_delivery_pipeline), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_delivery_pipeline(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_delivery_pipeline_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_delivery_pipeline(cloud_deploy.DeleteDeliveryPipelineRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_delivery_pipeline_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_delivery_pipeline), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_delivery_pipeline(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_delivery_pipeline_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_delivery_pipeline(cloud_deploy.DeleteDeliveryPipelineRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.ListTargetsRequest, dict])
def test_list_targets(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_targets), '__call__') as call:
        call.return_value = cloud_deploy.ListTargetsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_targets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListTargetsRequest()
    assert isinstance(response, pagers.ListTargetsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_targets_empty_call():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_targets), '__call__') as call:
        client.list_targets()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListTargetsRequest()

@pytest.mark.asyncio
async def test_list_targets_async(transport: str='grpc_asyncio', request_type=cloud_deploy.ListTargetsRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_targets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListTargetsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_targets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListTargetsRequest()
    assert isinstance(response, pagers.ListTargetsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_targets_async_from_dict():
    await test_list_targets_async(request_type=dict)

def test_list_targets_field_headers():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.ListTargetsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_targets), '__call__') as call:
        call.return_value = cloud_deploy.ListTargetsResponse()
        client.list_targets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_targets_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.ListTargetsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_targets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListTargetsResponse())
        await client.list_targets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_targets_flattened():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_targets), '__call__') as call:
        call.return_value = cloud_deploy.ListTargetsResponse()
        client.list_targets(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_targets_flattened_error():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_targets(cloud_deploy.ListTargetsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_targets_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_targets), '__call__') as call:
        call.return_value = cloud_deploy.ListTargetsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListTargetsResponse())
        response = await client.list_targets(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_targets_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_targets(cloud_deploy.ListTargetsRequest(), parent='parent_value')

def test_list_targets_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_targets), '__call__') as call:
        call.side_effect = (cloud_deploy.ListTargetsResponse(targets=[cloud_deploy.Target(), cloud_deploy.Target(), cloud_deploy.Target()], next_page_token='abc'), cloud_deploy.ListTargetsResponse(targets=[], next_page_token='def'), cloud_deploy.ListTargetsResponse(targets=[cloud_deploy.Target()], next_page_token='ghi'), cloud_deploy.ListTargetsResponse(targets=[cloud_deploy.Target(), cloud_deploy.Target()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_targets(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_deploy.Target) for i in results))

def test_list_targets_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_targets), '__call__') as call:
        call.side_effect = (cloud_deploy.ListTargetsResponse(targets=[cloud_deploy.Target(), cloud_deploy.Target(), cloud_deploy.Target()], next_page_token='abc'), cloud_deploy.ListTargetsResponse(targets=[], next_page_token='def'), cloud_deploy.ListTargetsResponse(targets=[cloud_deploy.Target()], next_page_token='ghi'), cloud_deploy.ListTargetsResponse(targets=[cloud_deploy.Target(), cloud_deploy.Target()]), RuntimeError)
        pages = list(client.list_targets(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_targets_async_pager():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_targets), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_deploy.ListTargetsResponse(targets=[cloud_deploy.Target(), cloud_deploy.Target(), cloud_deploy.Target()], next_page_token='abc'), cloud_deploy.ListTargetsResponse(targets=[], next_page_token='def'), cloud_deploy.ListTargetsResponse(targets=[cloud_deploy.Target()], next_page_token='ghi'), cloud_deploy.ListTargetsResponse(targets=[cloud_deploy.Target(), cloud_deploy.Target()]), RuntimeError)
        async_pager = await client.list_targets(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, cloud_deploy.Target) for i in responses))

@pytest.mark.asyncio
async def test_list_targets_async_pages():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_targets), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_deploy.ListTargetsResponse(targets=[cloud_deploy.Target(), cloud_deploy.Target(), cloud_deploy.Target()], next_page_token='abc'), cloud_deploy.ListTargetsResponse(targets=[], next_page_token='def'), cloud_deploy.ListTargetsResponse(targets=[cloud_deploy.Target()], next_page_token='ghi'), cloud_deploy.ListTargetsResponse(targets=[cloud_deploy.Target(), cloud_deploy.Target()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_targets(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_deploy.RollbackTargetRequest, dict])
def test_rollback_target(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.rollback_target), '__call__') as call:
        call.return_value = cloud_deploy.RollbackTargetResponse()
        response = client.rollback_target(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.RollbackTargetRequest()
    assert isinstance(response, cloud_deploy.RollbackTargetResponse)

def test_rollback_target_empty_call():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.rollback_target), '__call__') as call:
        client.rollback_target()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.RollbackTargetRequest()

@pytest.mark.asyncio
async def test_rollback_target_async(transport: str='grpc_asyncio', request_type=cloud_deploy.RollbackTargetRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.rollback_target), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.RollbackTargetResponse())
        response = await client.rollback_target(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.RollbackTargetRequest()
    assert isinstance(response, cloud_deploy.RollbackTargetResponse)

@pytest.mark.asyncio
async def test_rollback_target_async_from_dict():
    await test_rollback_target_async(request_type=dict)

def test_rollback_target_field_headers():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.RollbackTargetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.rollback_target), '__call__') as call:
        call.return_value = cloud_deploy.RollbackTargetResponse()
        client.rollback_target(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_rollback_target_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.RollbackTargetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.rollback_target), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.RollbackTargetResponse())
        await client.rollback_target(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_rollback_target_flattened():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.rollback_target), '__call__') as call:
        call.return_value = cloud_deploy.RollbackTargetResponse()
        client.rollback_target(name='name_value', target_id='target_id_value', rollout_id='rollout_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].target_id
        mock_val = 'target_id_value'
        assert arg == mock_val
        arg = args[0].rollout_id
        mock_val = 'rollout_id_value'
        assert arg == mock_val

def test_rollback_target_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.rollback_target(cloud_deploy.RollbackTargetRequest(), name='name_value', target_id='target_id_value', rollout_id='rollout_id_value')

@pytest.mark.asyncio
async def test_rollback_target_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.rollback_target), '__call__') as call:
        call.return_value = cloud_deploy.RollbackTargetResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.RollbackTargetResponse())
        response = await client.rollback_target(name='name_value', target_id='target_id_value', rollout_id='rollout_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].target_id
        mock_val = 'target_id_value'
        assert arg == mock_val
        arg = args[0].rollout_id
        mock_val = 'rollout_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_rollback_target_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.rollback_target(cloud_deploy.RollbackTargetRequest(), name='name_value', target_id='target_id_value', rollout_id='rollout_id_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.GetTargetRequest, dict])
def test_get_target(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_target), '__call__') as call:
        call.return_value = cloud_deploy.Target(name='name_value', target_id='target_id_value', uid='uid_value', description='description_value', require_approval=True, etag='etag_value')
        response = client.get_target(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetTargetRequest()
    assert isinstance(response, cloud_deploy.Target)
    assert response.name == 'name_value'
    assert response.target_id == 'target_id_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.require_approval is True
    assert response.etag == 'etag_value'

def test_get_target_empty_call():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_target), '__call__') as call:
        client.get_target()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetTargetRequest()

@pytest.mark.asyncio
async def test_get_target_async(transport: str='grpc_asyncio', request_type=cloud_deploy.GetTargetRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_target), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.Target(name='name_value', target_id='target_id_value', uid='uid_value', description='description_value', require_approval=True, etag='etag_value'))
        response = await client.get_target(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetTargetRequest()
    assert isinstance(response, cloud_deploy.Target)
    assert response.name == 'name_value'
    assert response.target_id == 'target_id_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.require_approval is True
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_get_target_async_from_dict():
    await test_get_target_async(request_type=dict)

def test_get_target_field_headers():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.GetTargetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_target), '__call__') as call:
        call.return_value = cloud_deploy.Target()
        client.get_target(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_target_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.GetTargetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_target), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.Target())
        await client.get_target(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_target_flattened():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_target), '__call__') as call:
        call.return_value = cloud_deploy.Target()
        client.get_target(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_target_flattened_error():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_target(cloud_deploy.GetTargetRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_target_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_target), '__call__') as call:
        call.return_value = cloud_deploy.Target()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.Target())
        response = await client.get_target(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_target_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_target(cloud_deploy.GetTargetRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.CreateTargetRequest, dict])
def test_create_target(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_target), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_target(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CreateTargetRequest()
    assert isinstance(response, future.Future)

def test_create_target_empty_call():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_target), '__call__') as call:
        client.create_target()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CreateTargetRequest()

@pytest.mark.asyncio
async def test_create_target_async(transport: str='grpc_asyncio', request_type=cloud_deploy.CreateTargetRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_target), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_target(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CreateTargetRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_target_async_from_dict():
    await test_create_target_async(request_type=dict)

def test_create_target_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.CreateTargetRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_target), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_target(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_target_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.CreateTargetRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_target), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_target(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_target_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_target), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_target(parent='parent_value', target=cloud_deploy.Target(name='name_value'), target_id='target_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].target
        mock_val = cloud_deploy.Target(name='name_value')
        assert arg == mock_val
        arg = args[0].target_id
        mock_val = 'target_id_value'
        assert arg == mock_val

def test_create_target_flattened_error():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_target(cloud_deploy.CreateTargetRequest(), parent='parent_value', target=cloud_deploy.Target(name='name_value'), target_id='target_id_value')

@pytest.mark.asyncio
async def test_create_target_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_target), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_target(parent='parent_value', target=cloud_deploy.Target(name='name_value'), target_id='target_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].target
        mock_val = cloud_deploy.Target(name='name_value')
        assert arg == mock_val
        arg = args[0].target_id
        mock_val = 'target_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_target_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_target(cloud_deploy.CreateTargetRequest(), parent='parent_value', target=cloud_deploy.Target(name='name_value'), target_id='target_id_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.UpdateTargetRequest, dict])
def test_update_target(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_target), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_target(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.UpdateTargetRequest()
    assert isinstance(response, future.Future)

def test_update_target_empty_call():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_target), '__call__') as call:
        client.update_target()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.UpdateTargetRequest()

@pytest.mark.asyncio
async def test_update_target_async(transport: str='grpc_asyncio', request_type=cloud_deploy.UpdateTargetRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_target), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_target(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.UpdateTargetRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_target_async_from_dict():
    await test_update_target_async(request_type=dict)

def test_update_target_field_headers():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.UpdateTargetRequest()
    request.target.name = 'name_value'
    with mock.patch.object(type(client.transport.update_target), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_target(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'target.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_target_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.UpdateTargetRequest()
    request.target.name = 'name_value'
    with mock.patch.object(type(client.transport.update_target), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_target(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'target.name=name_value') in kw['metadata']

def test_update_target_flattened():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_target), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_target(target=cloud_deploy.Target(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].target
        mock_val = cloud_deploy.Target(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_target_flattened_error():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_target(cloud_deploy.UpdateTargetRequest(), target=cloud_deploy.Target(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_target_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_target), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_target(target=cloud_deploy.Target(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].target
        mock_val = cloud_deploy.Target(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_target_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_target(cloud_deploy.UpdateTargetRequest(), target=cloud_deploy.Target(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [cloud_deploy.DeleteTargetRequest, dict])
def test_delete_target(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_target), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_target(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.DeleteTargetRequest()
    assert isinstance(response, future.Future)

def test_delete_target_empty_call():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_target), '__call__') as call:
        client.delete_target()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.DeleteTargetRequest()

@pytest.mark.asyncio
async def test_delete_target_async(transport: str='grpc_asyncio', request_type=cloud_deploy.DeleteTargetRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_target), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_target(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.DeleteTargetRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_target_async_from_dict():
    await test_delete_target_async(request_type=dict)

def test_delete_target_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.DeleteTargetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_target), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_target(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_target_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.DeleteTargetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_target), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_target(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_target_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_target), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_target(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_target_flattened_error():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_target(cloud_deploy.DeleteTargetRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_target_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_target), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_target(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_target_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_target(cloud_deploy.DeleteTargetRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.ListReleasesRequest, dict])
def test_list_releases(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_releases), '__call__') as call:
        call.return_value = cloud_deploy.ListReleasesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_releases(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListReleasesRequest()
    assert isinstance(response, pagers.ListReleasesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_releases_empty_call():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_releases), '__call__') as call:
        client.list_releases()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListReleasesRequest()

@pytest.mark.asyncio
async def test_list_releases_async(transport: str='grpc_asyncio', request_type=cloud_deploy.ListReleasesRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_releases), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListReleasesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_releases(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListReleasesRequest()
    assert isinstance(response, pagers.ListReleasesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_releases_async_from_dict():
    await test_list_releases_async(request_type=dict)

def test_list_releases_field_headers():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.ListReleasesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_releases), '__call__') as call:
        call.return_value = cloud_deploy.ListReleasesResponse()
        client.list_releases(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_releases_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.ListReleasesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_releases), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListReleasesResponse())
        await client.list_releases(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_releases_flattened():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_releases), '__call__') as call:
        call.return_value = cloud_deploy.ListReleasesResponse()
        client.list_releases(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_releases_flattened_error():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_releases(cloud_deploy.ListReleasesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_releases_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_releases), '__call__') as call:
        call.return_value = cloud_deploy.ListReleasesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListReleasesResponse())
        response = await client.list_releases(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_releases_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_releases(cloud_deploy.ListReleasesRequest(), parent='parent_value')

def test_list_releases_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_releases), '__call__') as call:
        call.side_effect = (cloud_deploy.ListReleasesResponse(releases=[cloud_deploy.Release(), cloud_deploy.Release(), cloud_deploy.Release()], next_page_token='abc'), cloud_deploy.ListReleasesResponse(releases=[], next_page_token='def'), cloud_deploy.ListReleasesResponse(releases=[cloud_deploy.Release()], next_page_token='ghi'), cloud_deploy.ListReleasesResponse(releases=[cloud_deploy.Release(), cloud_deploy.Release()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_releases(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_deploy.Release) for i in results))

def test_list_releases_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_releases), '__call__') as call:
        call.side_effect = (cloud_deploy.ListReleasesResponse(releases=[cloud_deploy.Release(), cloud_deploy.Release(), cloud_deploy.Release()], next_page_token='abc'), cloud_deploy.ListReleasesResponse(releases=[], next_page_token='def'), cloud_deploy.ListReleasesResponse(releases=[cloud_deploy.Release()], next_page_token='ghi'), cloud_deploy.ListReleasesResponse(releases=[cloud_deploy.Release(), cloud_deploy.Release()]), RuntimeError)
        pages = list(client.list_releases(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_releases_async_pager():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_releases), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_deploy.ListReleasesResponse(releases=[cloud_deploy.Release(), cloud_deploy.Release(), cloud_deploy.Release()], next_page_token='abc'), cloud_deploy.ListReleasesResponse(releases=[], next_page_token='def'), cloud_deploy.ListReleasesResponse(releases=[cloud_deploy.Release()], next_page_token='ghi'), cloud_deploy.ListReleasesResponse(releases=[cloud_deploy.Release(), cloud_deploy.Release()]), RuntimeError)
        async_pager = await client.list_releases(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, cloud_deploy.Release) for i in responses))

@pytest.mark.asyncio
async def test_list_releases_async_pages():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_releases), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_deploy.ListReleasesResponse(releases=[cloud_deploy.Release(), cloud_deploy.Release(), cloud_deploy.Release()], next_page_token='abc'), cloud_deploy.ListReleasesResponse(releases=[], next_page_token='def'), cloud_deploy.ListReleasesResponse(releases=[cloud_deploy.Release()], next_page_token='ghi'), cloud_deploy.ListReleasesResponse(releases=[cloud_deploy.Release(), cloud_deploy.Release()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_releases(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_deploy.GetReleaseRequest, dict])
def test_get_release(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_release), '__call__') as call:
        call.return_value = cloud_deploy.Release(name='name_value', uid='uid_value', description='description_value', abandoned=True, skaffold_config_uri='skaffold_config_uri_value', skaffold_config_path='skaffold_config_path_value', render_state=cloud_deploy.Release.RenderState.SUCCEEDED, etag='etag_value', skaffold_version='skaffold_version_value')
        response = client.get_release(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetReleaseRequest()
    assert isinstance(response, cloud_deploy.Release)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.abandoned is True
    assert response.skaffold_config_uri == 'skaffold_config_uri_value'
    assert response.skaffold_config_path == 'skaffold_config_path_value'
    assert response.render_state == cloud_deploy.Release.RenderState.SUCCEEDED
    assert response.etag == 'etag_value'
    assert response.skaffold_version == 'skaffold_version_value'

def test_get_release_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_release), '__call__') as call:
        client.get_release()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetReleaseRequest()

@pytest.mark.asyncio
async def test_get_release_async(transport: str='grpc_asyncio', request_type=cloud_deploy.GetReleaseRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_release), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.Release(name='name_value', uid='uid_value', description='description_value', abandoned=True, skaffold_config_uri='skaffold_config_uri_value', skaffold_config_path='skaffold_config_path_value', render_state=cloud_deploy.Release.RenderState.SUCCEEDED, etag='etag_value', skaffold_version='skaffold_version_value'))
        response = await client.get_release(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetReleaseRequest()
    assert isinstance(response, cloud_deploy.Release)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.abandoned is True
    assert response.skaffold_config_uri == 'skaffold_config_uri_value'
    assert response.skaffold_config_path == 'skaffold_config_path_value'
    assert response.render_state == cloud_deploy.Release.RenderState.SUCCEEDED
    assert response.etag == 'etag_value'
    assert response.skaffold_version == 'skaffold_version_value'

@pytest.mark.asyncio
async def test_get_release_async_from_dict():
    await test_get_release_async(request_type=dict)

def test_get_release_field_headers():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.GetReleaseRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_release), '__call__') as call:
        call.return_value = cloud_deploy.Release()
        client.get_release(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_release_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.GetReleaseRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_release), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.Release())
        await client.get_release(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_release_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_release), '__call__') as call:
        call.return_value = cloud_deploy.Release()
        client.get_release(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_release_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_release(cloud_deploy.GetReleaseRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_release_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_release), '__call__') as call:
        call.return_value = cloud_deploy.Release()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.Release())
        response = await client.get_release(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_release_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_release(cloud_deploy.GetReleaseRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.CreateReleaseRequest, dict])
def test_create_release(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_release), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_release(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CreateReleaseRequest()
    assert isinstance(response, future.Future)

def test_create_release_empty_call():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_release), '__call__') as call:
        client.create_release()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CreateReleaseRequest()

@pytest.mark.asyncio
async def test_create_release_async(transport: str='grpc_asyncio', request_type=cloud_deploy.CreateReleaseRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_release), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_release(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CreateReleaseRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_release_async_from_dict():
    await test_create_release_async(request_type=dict)

def test_create_release_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.CreateReleaseRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_release), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_release(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_release_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.CreateReleaseRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_release), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_release(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_release_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_release), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_release(parent='parent_value', release=cloud_deploy.Release(name='name_value'), release_id='release_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].release
        mock_val = cloud_deploy.Release(name='name_value')
        assert arg == mock_val
        arg = args[0].release_id
        mock_val = 'release_id_value'
        assert arg == mock_val

def test_create_release_flattened_error():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_release(cloud_deploy.CreateReleaseRequest(), parent='parent_value', release=cloud_deploy.Release(name='name_value'), release_id='release_id_value')

@pytest.mark.asyncio
async def test_create_release_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_release), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_release(parent='parent_value', release=cloud_deploy.Release(name='name_value'), release_id='release_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].release
        mock_val = cloud_deploy.Release(name='name_value')
        assert arg == mock_val
        arg = args[0].release_id
        mock_val = 'release_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_release_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_release(cloud_deploy.CreateReleaseRequest(), parent='parent_value', release=cloud_deploy.Release(name='name_value'), release_id='release_id_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.AbandonReleaseRequest, dict])
def test_abandon_release(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.abandon_release), '__call__') as call:
        call.return_value = cloud_deploy.AbandonReleaseResponse()
        response = client.abandon_release(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.AbandonReleaseRequest()
    assert isinstance(response, cloud_deploy.AbandonReleaseResponse)

def test_abandon_release_empty_call():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.abandon_release), '__call__') as call:
        client.abandon_release()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.AbandonReleaseRequest()

@pytest.mark.asyncio
async def test_abandon_release_async(transport: str='grpc_asyncio', request_type=cloud_deploy.AbandonReleaseRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.abandon_release), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.AbandonReleaseResponse())
        response = await client.abandon_release(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.AbandonReleaseRequest()
    assert isinstance(response, cloud_deploy.AbandonReleaseResponse)

@pytest.mark.asyncio
async def test_abandon_release_async_from_dict():
    await test_abandon_release_async(request_type=dict)

def test_abandon_release_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.AbandonReleaseRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.abandon_release), '__call__') as call:
        call.return_value = cloud_deploy.AbandonReleaseResponse()
        client.abandon_release(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_abandon_release_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.AbandonReleaseRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.abandon_release), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.AbandonReleaseResponse())
        await client.abandon_release(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_abandon_release_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.abandon_release), '__call__') as call:
        call.return_value = cloud_deploy.AbandonReleaseResponse()
        client.abandon_release(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_abandon_release_flattened_error():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.abandon_release(cloud_deploy.AbandonReleaseRequest(), name='name_value')

@pytest.mark.asyncio
async def test_abandon_release_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.abandon_release), '__call__') as call:
        call.return_value = cloud_deploy.AbandonReleaseResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.AbandonReleaseResponse())
        response = await client.abandon_release(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_abandon_release_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.abandon_release(cloud_deploy.AbandonReleaseRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.ApproveRolloutRequest, dict])
def test_approve_rollout(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.approve_rollout), '__call__') as call:
        call.return_value = cloud_deploy.ApproveRolloutResponse()
        response = client.approve_rollout(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ApproveRolloutRequest()
    assert isinstance(response, cloud_deploy.ApproveRolloutResponse)

def test_approve_rollout_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.approve_rollout), '__call__') as call:
        client.approve_rollout()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ApproveRolloutRequest()

@pytest.mark.asyncio
async def test_approve_rollout_async(transport: str='grpc_asyncio', request_type=cloud_deploy.ApproveRolloutRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.approve_rollout), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ApproveRolloutResponse())
        response = await client.approve_rollout(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ApproveRolloutRequest()
    assert isinstance(response, cloud_deploy.ApproveRolloutResponse)

@pytest.mark.asyncio
async def test_approve_rollout_async_from_dict():
    await test_approve_rollout_async(request_type=dict)

def test_approve_rollout_field_headers():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.ApproveRolloutRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.approve_rollout), '__call__') as call:
        call.return_value = cloud_deploy.ApproveRolloutResponse()
        client.approve_rollout(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_approve_rollout_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.ApproveRolloutRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.approve_rollout), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ApproveRolloutResponse())
        await client.approve_rollout(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_approve_rollout_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.approve_rollout), '__call__') as call:
        call.return_value = cloud_deploy.ApproveRolloutResponse()
        client.approve_rollout(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_approve_rollout_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.approve_rollout(cloud_deploy.ApproveRolloutRequest(), name='name_value')

@pytest.mark.asyncio
async def test_approve_rollout_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.approve_rollout), '__call__') as call:
        call.return_value = cloud_deploy.ApproveRolloutResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ApproveRolloutResponse())
        response = await client.approve_rollout(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_approve_rollout_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.approve_rollout(cloud_deploy.ApproveRolloutRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.AdvanceRolloutRequest, dict])
def test_advance_rollout(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.advance_rollout), '__call__') as call:
        call.return_value = cloud_deploy.AdvanceRolloutResponse()
        response = client.advance_rollout(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.AdvanceRolloutRequest()
    assert isinstance(response, cloud_deploy.AdvanceRolloutResponse)

def test_advance_rollout_empty_call():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.advance_rollout), '__call__') as call:
        client.advance_rollout()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.AdvanceRolloutRequest()

@pytest.mark.asyncio
async def test_advance_rollout_async(transport: str='grpc_asyncio', request_type=cloud_deploy.AdvanceRolloutRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.advance_rollout), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.AdvanceRolloutResponse())
        response = await client.advance_rollout(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.AdvanceRolloutRequest()
    assert isinstance(response, cloud_deploy.AdvanceRolloutResponse)

@pytest.mark.asyncio
async def test_advance_rollout_async_from_dict():
    await test_advance_rollout_async(request_type=dict)

def test_advance_rollout_field_headers():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.AdvanceRolloutRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.advance_rollout), '__call__') as call:
        call.return_value = cloud_deploy.AdvanceRolloutResponse()
        client.advance_rollout(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_advance_rollout_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.AdvanceRolloutRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.advance_rollout), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.AdvanceRolloutResponse())
        await client.advance_rollout(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_advance_rollout_flattened():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.advance_rollout), '__call__') as call:
        call.return_value = cloud_deploy.AdvanceRolloutResponse()
        client.advance_rollout(name='name_value', phase_id='phase_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].phase_id
        mock_val = 'phase_id_value'
        assert arg == mock_val

def test_advance_rollout_flattened_error():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.advance_rollout(cloud_deploy.AdvanceRolloutRequest(), name='name_value', phase_id='phase_id_value')

@pytest.mark.asyncio
async def test_advance_rollout_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.advance_rollout), '__call__') as call:
        call.return_value = cloud_deploy.AdvanceRolloutResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.AdvanceRolloutResponse())
        response = await client.advance_rollout(name='name_value', phase_id='phase_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].phase_id
        mock_val = 'phase_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_advance_rollout_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.advance_rollout(cloud_deploy.AdvanceRolloutRequest(), name='name_value', phase_id='phase_id_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.CancelRolloutRequest, dict])
def test_cancel_rollout(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.cancel_rollout), '__call__') as call:
        call.return_value = cloud_deploy.CancelRolloutResponse()
        response = client.cancel_rollout(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CancelRolloutRequest()
    assert isinstance(response, cloud_deploy.CancelRolloutResponse)

def test_cancel_rollout_empty_call():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.cancel_rollout), '__call__') as call:
        client.cancel_rollout()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CancelRolloutRequest()

@pytest.mark.asyncio
async def test_cancel_rollout_async(transport: str='grpc_asyncio', request_type=cloud_deploy.CancelRolloutRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.cancel_rollout), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.CancelRolloutResponse())
        response = await client.cancel_rollout(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CancelRolloutRequest()
    assert isinstance(response, cloud_deploy.CancelRolloutResponse)

@pytest.mark.asyncio
async def test_cancel_rollout_async_from_dict():
    await test_cancel_rollout_async(request_type=dict)

def test_cancel_rollout_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.CancelRolloutRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.cancel_rollout), '__call__') as call:
        call.return_value = cloud_deploy.CancelRolloutResponse()
        client.cancel_rollout(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_cancel_rollout_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.CancelRolloutRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.cancel_rollout), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.CancelRolloutResponse())
        await client.cancel_rollout(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_cancel_rollout_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_rollout), '__call__') as call:
        call.return_value = cloud_deploy.CancelRolloutResponse()
        client.cancel_rollout(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_cancel_rollout_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.cancel_rollout(cloud_deploy.CancelRolloutRequest(), name='name_value')

@pytest.mark.asyncio
async def test_cancel_rollout_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_rollout), '__call__') as call:
        call.return_value = cloud_deploy.CancelRolloutResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.CancelRolloutResponse())
        response = await client.cancel_rollout(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_cancel_rollout_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.cancel_rollout(cloud_deploy.CancelRolloutRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.ListRolloutsRequest, dict])
def test_list_rollouts(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_rollouts), '__call__') as call:
        call.return_value = cloud_deploy.ListRolloutsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_rollouts(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListRolloutsRequest()
    assert isinstance(response, pagers.ListRolloutsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_rollouts_empty_call():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_rollouts), '__call__') as call:
        client.list_rollouts()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListRolloutsRequest()

@pytest.mark.asyncio
async def test_list_rollouts_async(transport: str='grpc_asyncio', request_type=cloud_deploy.ListRolloutsRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_rollouts), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListRolloutsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_rollouts(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListRolloutsRequest()
    assert isinstance(response, pagers.ListRolloutsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_rollouts_async_from_dict():
    await test_list_rollouts_async(request_type=dict)

def test_list_rollouts_field_headers():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.ListRolloutsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_rollouts), '__call__') as call:
        call.return_value = cloud_deploy.ListRolloutsResponse()
        client.list_rollouts(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_rollouts_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.ListRolloutsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_rollouts), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListRolloutsResponse())
        await client.list_rollouts(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_rollouts_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_rollouts), '__call__') as call:
        call.return_value = cloud_deploy.ListRolloutsResponse()
        client.list_rollouts(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_rollouts_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_rollouts(cloud_deploy.ListRolloutsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_rollouts_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_rollouts), '__call__') as call:
        call.return_value = cloud_deploy.ListRolloutsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListRolloutsResponse())
        response = await client.list_rollouts(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_rollouts_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_rollouts(cloud_deploy.ListRolloutsRequest(), parent='parent_value')

def test_list_rollouts_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_rollouts), '__call__') as call:
        call.side_effect = (cloud_deploy.ListRolloutsResponse(rollouts=[cloud_deploy.Rollout(), cloud_deploy.Rollout(), cloud_deploy.Rollout()], next_page_token='abc'), cloud_deploy.ListRolloutsResponse(rollouts=[], next_page_token='def'), cloud_deploy.ListRolloutsResponse(rollouts=[cloud_deploy.Rollout()], next_page_token='ghi'), cloud_deploy.ListRolloutsResponse(rollouts=[cloud_deploy.Rollout(), cloud_deploy.Rollout()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_rollouts(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_deploy.Rollout) for i in results))

def test_list_rollouts_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_rollouts), '__call__') as call:
        call.side_effect = (cloud_deploy.ListRolloutsResponse(rollouts=[cloud_deploy.Rollout(), cloud_deploy.Rollout(), cloud_deploy.Rollout()], next_page_token='abc'), cloud_deploy.ListRolloutsResponse(rollouts=[], next_page_token='def'), cloud_deploy.ListRolloutsResponse(rollouts=[cloud_deploy.Rollout()], next_page_token='ghi'), cloud_deploy.ListRolloutsResponse(rollouts=[cloud_deploy.Rollout(), cloud_deploy.Rollout()]), RuntimeError)
        pages = list(client.list_rollouts(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_rollouts_async_pager():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_rollouts), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_deploy.ListRolloutsResponse(rollouts=[cloud_deploy.Rollout(), cloud_deploy.Rollout(), cloud_deploy.Rollout()], next_page_token='abc'), cloud_deploy.ListRolloutsResponse(rollouts=[], next_page_token='def'), cloud_deploy.ListRolloutsResponse(rollouts=[cloud_deploy.Rollout()], next_page_token='ghi'), cloud_deploy.ListRolloutsResponse(rollouts=[cloud_deploy.Rollout(), cloud_deploy.Rollout()]), RuntimeError)
        async_pager = await client.list_rollouts(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, cloud_deploy.Rollout) for i in responses))

@pytest.mark.asyncio
async def test_list_rollouts_async_pages():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_rollouts), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_deploy.ListRolloutsResponse(rollouts=[cloud_deploy.Rollout(), cloud_deploy.Rollout(), cloud_deploy.Rollout()], next_page_token='abc'), cloud_deploy.ListRolloutsResponse(rollouts=[], next_page_token='def'), cloud_deploy.ListRolloutsResponse(rollouts=[cloud_deploy.Rollout()], next_page_token='ghi'), cloud_deploy.ListRolloutsResponse(rollouts=[cloud_deploy.Rollout(), cloud_deploy.Rollout()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_rollouts(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_deploy.GetRolloutRequest, dict])
def test_get_rollout(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_rollout), '__call__') as call:
        call.return_value = cloud_deploy.Rollout(name='name_value', uid='uid_value', description='description_value', target_id='target_id_value', approval_state=cloud_deploy.Rollout.ApprovalState.NEEDS_APPROVAL, state=cloud_deploy.Rollout.State.SUCCEEDED, failure_reason='failure_reason_value', deploying_build='deploying_build_value', etag='etag_value', deploy_failure_cause=cloud_deploy.Rollout.FailureCause.CLOUD_BUILD_UNAVAILABLE, controller_rollout='controller_rollout_value', rollback_of_rollout='rollback_of_rollout_value', rolled_back_by_rollouts=['rolled_back_by_rollouts_value'])
        response = client.get_rollout(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetRolloutRequest()
    assert isinstance(response, cloud_deploy.Rollout)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.target_id == 'target_id_value'
    assert response.approval_state == cloud_deploy.Rollout.ApprovalState.NEEDS_APPROVAL
    assert response.state == cloud_deploy.Rollout.State.SUCCEEDED
    assert response.failure_reason == 'failure_reason_value'
    assert response.deploying_build == 'deploying_build_value'
    assert response.etag == 'etag_value'
    assert response.deploy_failure_cause == cloud_deploy.Rollout.FailureCause.CLOUD_BUILD_UNAVAILABLE
    assert response.controller_rollout == 'controller_rollout_value'
    assert response.rollback_of_rollout == 'rollback_of_rollout_value'
    assert response.rolled_back_by_rollouts == ['rolled_back_by_rollouts_value']

def test_get_rollout_empty_call():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_rollout), '__call__') as call:
        client.get_rollout()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetRolloutRequest()

@pytest.mark.asyncio
async def test_get_rollout_async(transport: str='grpc_asyncio', request_type=cloud_deploy.GetRolloutRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_rollout), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.Rollout(name='name_value', uid='uid_value', description='description_value', target_id='target_id_value', approval_state=cloud_deploy.Rollout.ApprovalState.NEEDS_APPROVAL, state=cloud_deploy.Rollout.State.SUCCEEDED, failure_reason='failure_reason_value', deploying_build='deploying_build_value', etag='etag_value', deploy_failure_cause=cloud_deploy.Rollout.FailureCause.CLOUD_BUILD_UNAVAILABLE, controller_rollout='controller_rollout_value', rollback_of_rollout='rollback_of_rollout_value', rolled_back_by_rollouts=['rolled_back_by_rollouts_value']))
        response = await client.get_rollout(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetRolloutRequest()
    assert isinstance(response, cloud_deploy.Rollout)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.target_id == 'target_id_value'
    assert response.approval_state == cloud_deploy.Rollout.ApprovalState.NEEDS_APPROVAL
    assert response.state == cloud_deploy.Rollout.State.SUCCEEDED
    assert response.failure_reason == 'failure_reason_value'
    assert response.deploying_build == 'deploying_build_value'
    assert response.etag == 'etag_value'
    assert response.deploy_failure_cause == cloud_deploy.Rollout.FailureCause.CLOUD_BUILD_UNAVAILABLE
    assert response.controller_rollout == 'controller_rollout_value'
    assert response.rollback_of_rollout == 'rollback_of_rollout_value'
    assert response.rolled_back_by_rollouts == ['rolled_back_by_rollouts_value']

@pytest.mark.asyncio
async def test_get_rollout_async_from_dict():
    await test_get_rollout_async(request_type=dict)

def test_get_rollout_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.GetRolloutRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_rollout), '__call__') as call:
        call.return_value = cloud_deploy.Rollout()
        client.get_rollout(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_rollout_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.GetRolloutRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_rollout), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.Rollout())
        await client.get_rollout(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_rollout_flattened():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_rollout), '__call__') as call:
        call.return_value = cloud_deploy.Rollout()
        client.get_rollout(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_rollout_flattened_error():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_rollout(cloud_deploy.GetRolloutRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_rollout_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_rollout), '__call__') as call:
        call.return_value = cloud_deploy.Rollout()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.Rollout())
        response = await client.get_rollout(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_rollout_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_rollout(cloud_deploy.GetRolloutRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.CreateRolloutRequest, dict])
def test_create_rollout(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_rollout), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_rollout(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CreateRolloutRequest()
    assert isinstance(response, future.Future)

def test_create_rollout_empty_call():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_rollout), '__call__') as call:
        client.create_rollout()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CreateRolloutRequest()

@pytest.mark.asyncio
async def test_create_rollout_async(transport: str='grpc_asyncio', request_type=cloud_deploy.CreateRolloutRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_rollout), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_rollout(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CreateRolloutRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_rollout_async_from_dict():
    await test_create_rollout_async(request_type=dict)

def test_create_rollout_field_headers():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.CreateRolloutRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_rollout), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_rollout(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_rollout_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.CreateRolloutRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_rollout), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_rollout(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_rollout_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_rollout), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_rollout(parent='parent_value', rollout=cloud_deploy.Rollout(name='name_value'), rollout_id='rollout_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].rollout
        mock_val = cloud_deploy.Rollout(name='name_value')
        assert arg == mock_val
        arg = args[0].rollout_id
        mock_val = 'rollout_id_value'
        assert arg == mock_val

def test_create_rollout_flattened_error():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_rollout(cloud_deploy.CreateRolloutRequest(), parent='parent_value', rollout=cloud_deploy.Rollout(name='name_value'), rollout_id='rollout_id_value')

@pytest.mark.asyncio
async def test_create_rollout_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_rollout), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_rollout(parent='parent_value', rollout=cloud_deploy.Rollout(name='name_value'), rollout_id='rollout_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].rollout
        mock_val = cloud_deploy.Rollout(name='name_value')
        assert arg == mock_val
        arg = args[0].rollout_id
        mock_val = 'rollout_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_rollout_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_rollout(cloud_deploy.CreateRolloutRequest(), parent='parent_value', rollout=cloud_deploy.Rollout(name='name_value'), rollout_id='rollout_id_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.IgnoreJobRequest, dict])
def test_ignore_job(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.ignore_job), '__call__') as call:
        call.return_value = cloud_deploy.IgnoreJobResponse()
        response = client.ignore_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.IgnoreJobRequest()
    assert isinstance(response, cloud_deploy.IgnoreJobResponse)

def test_ignore_job_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.ignore_job), '__call__') as call:
        client.ignore_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.IgnoreJobRequest()

@pytest.mark.asyncio
async def test_ignore_job_async(transport: str='grpc_asyncio', request_type=cloud_deploy.IgnoreJobRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.ignore_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.IgnoreJobResponse())
        response = await client.ignore_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.IgnoreJobRequest()
    assert isinstance(response, cloud_deploy.IgnoreJobResponse)

@pytest.mark.asyncio
async def test_ignore_job_async_from_dict():
    await test_ignore_job_async(request_type=dict)

def test_ignore_job_field_headers():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.IgnoreJobRequest()
    request.rollout = 'rollout_value'
    with mock.patch.object(type(client.transport.ignore_job), '__call__') as call:
        call.return_value = cloud_deploy.IgnoreJobResponse()
        client.ignore_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'rollout=rollout_value') in kw['metadata']

@pytest.mark.asyncio
async def test_ignore_job_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.IgnoreJobRequest()
    request.rollout = 'rollout_value'
    with mock.patch.object(type(client.transport.ignore_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.IgnoreJobResponse())
        await client.ignore_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'rollout=rollout_value') in kw['metadata']

def test_ignore_job_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.ignore_job), '__call__') as call:
        call.return_value = cloud_deploy.IgnoreJobResponse()
        client.ignore_job(rollout='rollout_value', phase_id='phase_id_value', job_id='job_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].rollout
        mock_val = 'rollout_value'
        assert arg == mock_val
        arg = args[0].phase_id
        mock_val = 'phase_id_value'
        assert arg == mock_val
        arg = args[0].job_id
        mock_val = 'job_id_value'
        assert arg == mock_val

def test_ignore_job_flattened_error():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.ignore_job(cloud_deploy.IgnoreJobRequest(), rollout='rollout_value', phase_id='phase_id_value', job_id='job_id_value')

@pytest.mark.asyncio
async def test_ignore_job_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.ignore_job), '__call__') as call:
        call.return_value = cloud_deploy.IgnoreJobResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.IgnoreJobResponse())
        response = await client.ignore_job(rollout='rollout_value', phase_id='phase_id_value', job_id='job_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].rollout
        mock_val = 'rollout_value'
        assert arg == mock_val
        arg = args[0].phase_id
        mock_val = 'phase_id_value'
        assert arg == mock_val
        arg = args[0].job_id
        mock_val = 'job_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_ignore_job_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.ignore_job(cloud_deploy.IgnoreJobRequest(), rollout='rollout_value', phase_id='phase_id_value', job_id='job_id_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.RetryJobRequest, dict])
def test_retry_job(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.retry_job), '__call__') as call:
        call.return_value = cloud_deploy.RetryJobResponse()
        response = client.retry_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.RetryJobRequest()
    assert isinstance(response, cloud_deploy.RetryJobResponse)

def test_retry_job_empty_call():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.retry_job), '__call__') as call:
        client.retry_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.RetryJobRequest()

@pytest.mark.asyncio
async def test_retry_job_async(transport: str='grpc_asyncio', request_type=cloud_deploy.RetryJobRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.retry_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.RetryJobResponse())
        response = await client.retry_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.RetryJobRequest()
    assert isinstance(response, cloud_deploy.RetryJobResponse)

@pytest.mark.asyncio
async def test_retry_job_async_from_dict():
    await test_retry_job_async(request_type=dict)

def test_retry_job_field_headers():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.RetryJobRequest()
    request.rollout = 'rollout_value'
    with mock.patch.object(type(client.transport.retry_job), '__call__') as call:
        call.return_value = cloud_deploy.RetryJobResponse()
        client.retry_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'rollout=rollout_value') in kw['metadata']

@pytest.mark.asyncio
async def test_retry_job_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.RetryJobRequest()
    request.rollout = 'rollout_value'
    with mock.patch.object(type(client.transport.retry_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.RetryJobResponse())
        await client.retry_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'rollout=rollout_value') in kw['metadata']

def test_retry_job_flattened():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.retry_job), '__call__') as call:
        call.return_value = cloud_deploy.RetryJobResponse()
        client.retry_job(rollout='rollout_value', phase_id='phase_id_value', job_id='job_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].rollout
        mock_val = 'rollout_value'
        assert arg == mock_val
        arg = args[0].phase_id
        mock_val = 'phase_id_value'
        assert arg == mock_val
        arg = args[0].job_id
        mock_val = 'job_id_value'
        assert arg == mock_val

def test_retry_job_flattened_error():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.retry_job(cloud_deploy.RetryJobRequest(), rollout='rollout_value', phase_id='phase_id_value', job_id='job_id_value')

@pytest.mark.asyncio
async def test_retry_job_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.retry_job), '__call__') as call:
        call.return_value = cloud_deploy.RetryJobResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.RetryJobResponse())
        response = await client.retry_job(rollout='rollout_value', phase_id='phase_id_value', job_id='job_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].rollout
        mock_val = 'rollout_value'
        assert arg == mock_val
        arg = args[0].phase_id
        mock_val = 'phase_id_value'
        assert arg == mock_val
        arg = args[0].job_id
        mock_val = 'job_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_retry_job_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.retry_job(cloud_deploy.RetryJobRequest(), rollout='rollout_value', phase_id='phase_id_value', job_id='job_id_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.ListJobRunsRequest, dict])
def test_list_job_runs(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_job_runs), '__call__') as call:
        call.return_value = cloud_deploy.ListJobRunsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_job_runs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListJobRunsRequest()
    assert isinstance(response, pagers.ListJobRunsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_job_runs_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_job_runs), '__call__') as call:
        client.list_job_runs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListJobRunsRequest()

@pytest.mark.asyncio
async def test_list_job_runs_async(transport: str='grpc_asyncio', request_type=cloud_deploy.ListJobRunsRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_job_runs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListJobRunsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_job_runs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListJobRunsRequest()
    assert isinstance(response, pagers.ListJobRunsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_job_runs_async_from_dict():
    await test_list_job_runs_async(request_type=dict)

def test_list_job_runs_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.ListJobRunsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_job_runs), '__call__') as call:
        call.return_value = cloud_deploy.ListJobRunsResponse()
        client.list_job_runs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_job_runs_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.ListJobRunsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_job_runs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListJobRunsResponse())
        await client.list_job_runs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_job_runs_flattened():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_job_runs), '__call__') as call:
        call.return_value = cloud_deploy.ListJobRunsResponse()
        client.list_job_runs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_job_runs_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_job_runs(cloud_deploy.ListJobRunsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_job_runs_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_job_runs), '__call__') as call:
        call.return_value = cloud_deploy.ListJobRunsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListJobRunsResponse())
        response = await client.list_job_runs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_job_runs_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_job_runs(cloud_deploy.ListJobRunsRequest(), parent='parent_value')

def test_list_job_runs_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_job_runs), '__call__') as call:
        call.side_effect = (cloud_deploy.ListJobRunsResponse(job_runs=[cloud_deploy.JobRun(), cloud_deploy.JobRun(), cloud_deploy.JobRun()], next_page_token='abc'), cloud_deploy.ListJobRunsResponse(job_runs=[], next_page_token='def'), cloud_deploy.ListJobRunsResponse(job_runs=[cloud_deploy.JobRun()], next_page_token='ghi'), cloud_deploy.ListJobRunsResponse(job_runs=[cloud_deploy.JobRun(), cloud_deploy.JobRun()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_job_runs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_deploy.JobRun) for i in results))

def test_list_job_runs_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_job_runs), '__call__') as call:
        call.side_effect = (cloud_deploy.ListJobRunsResponse(job_runs=[cloud_deploy.JobRun(), cloud_deploy.JobRun(), cloud_deploy.JobRun()], next_page_token='abc'), cloud_deploy.ListJobRunsResponse(job_runs=[], next_page_token='def'), cloud_deploy.ListJobRunsResponse(job_runs=[cloud_deploy.JobRun()], next_page_token='ghi'), cloud_deploy.ListJobRunsResponse(job_runs=[cloud_deploy.JobRun(), cloud_deploy.JobRun()]), RuntimeError)
        pages = list(client.list_job_runs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_job_runs_async_pager():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_job_runs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_deploy.ListJobRunsResponse(job_runs=[cloud_deploy.JobRun(), cloud_deploy.JobRun(), cloud_deploy.JobRun()], next_page_token='abc'), cloud_deploy.ListJobRunsResponse(job_runs=[], next_page_token='def'), cloud_deploy.ListJobRunsResponse(job_runs=[cloud_deploy.JobRun()], next_page_token='ghi'), cloud_deploy.ListJobRunsResponse(job_runs=[cloud_deploy.JobRun(), cloud_deploy.JobRun()]), RuntimeError)
        async_pager = await client.list_job_runs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, cloud_deploy.JobRun) for i in responses))

@pytest.mark.asyncio
async def test_list_job_runs_async_pages():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_job_runs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_deploy.ListJobRunsResponse(job_runs=[cloud_deploy.JobRun(), cloud_deploy.JobRun(), cloud_deploy.JobRun()], next_page_token='abc'), cloud_deploy.ListJobRunsResponse(job_runs=[], next_page_token='def'), cloud_deploy.ListJobRunsResponse(job_runs=[cloud_deploy.JobRun()], next_page_token='ghi'), cloud_deploy.ListJobRunsResponse(job_runs=[cloud_deploy.JobRun(), cloud_deploy.JobRun()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_job_runs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_deploy.GetJobRunRequest, dict])
def test_get_job_run(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_job_run), '__call__') as call:
        call.return_value = cloud_deploy.JobRun(name='name_value', uid='uid_value', phase_id='phase_id_value', job_id='job_id_value', state=cloud_deploy.JobRun.State.IN_PROGRESS, etag='etag_value')
        response = client.get_job_run(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetJobRunRequest()
    assert isinstance(response, cloud_deploy.JobRun)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.phase_id == 'phase_id_value'
    assert response.job_id == 'job_id_value'
    assert response.state == cloud_deploy.JobRun.State.IN_PROGRESS
    assert response.etag == 'etag_value'

def test_get_job_run_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_job_run), '__call__') as call:
        client.get_job_run()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetJobRunRequest()

@pytest.mark.asyncio
async def test_get_job_run_async(transport: str='grpc_asyncio', request_type=cloud_deploy.GetJobRunRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_job_run), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.JobRun(name='name_value', uid='uid_value', phase_id='phase_id_value', job_id='job_id_value', state=cloud_deploy.JobRun.State.IN_PROGRESS, etag='etag_value'))
        response = await client.get_job_run(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetJobRunRequest()
    assert isinstance(response, cloud_deploy.JobRun)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.phase_id == 'phase_id_value'
    assert response.job_id == 'job_id_value'
    assert response.state == cloud_deploy.JobRun.State.IN_PROGRESS
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_get_job_run_async_from_dict():
    await test_get_job_run_async(request_type=dict)

def test_get_job_run_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.GetJobRunRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_job_run), '__call__') as call:
        call.return_value = cloud_deploy.JobRun()
        client.get_job_run(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_job_run_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.GetJobRunRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_job_run), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.JobRun())
        await client.get_job_run(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_job_run_flattened():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_job_run), '__call__') as call:
        call.return_value = cloud_deploy.JobRun()
        client.get_job_run(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_job_run_flattened_error():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_job_run(cloud_deploy.GetJobRunRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_job_run_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_job_run), '__call__') as call:
        call.return_value = cloud_deploy.JobRun()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.JobRun())
        response = await client.get_job_run(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_job_run_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_job_run(cloud_deploy.GetJobRunRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.TerminateJobRunRequest, dict])
def test_terminate_job_run(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.terminate_job_run), '__call__') as call:
        call.return_value = cloud_deploy.TerminateJobRunResponse()
        response = client.terminate_job_run(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.TerminateJobRunRequest()
    assert isinstance(response, cloud_deploy.TerminateJobRunResponse)

def test_terminate_job_run_empty_call():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.terminate_job_run), '__call__') as call:
        client.terminate_job_run()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.TerminateJobRunRequest()

@pytest.mark.asyncio
async def test_terminate_job_run_async(transport: str='grpc_asyncio', request_type=cloud_deploy.TerminateJobRunRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.terminate_job_run), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.TerminateJobRunResponse())
        response = await client.terminate_job_run(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.TerminateJobRunRequest()
    assert isinstance(response, cloud_deploy.TerminateJobRunResponse)

@pytest.mark.asyncio
async def test_terminate_job_run_async_from_dict():
    await test_terminate_job_run_async(request_type=dict)

def test_terminate_job_run_field_headers():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.TerminateJobRunRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.terminate_job_run), '__call__') as call:
        call.return_value = cloud_deploy.TerminateJobRunResponse()
        client.terminate_job_run(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_terminate_job_run_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.TerminateJobRunRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.terminate_job_run), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.TerminateJobRunResponse())
        await client.terminate_job_run(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_terminate_job_run_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.terminate_job_run), '__call__') as call:
        call.return_value = cloud_deploy.TerminateJobRunResponse()
        client.terminate_job_run(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_terminate_job_run_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.terminate_job_run(cloud_deploy.TerminateJobRunRequest(), name='name_value')

@pytest.mark.asyncio
async def test_terminate_job_run_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.terminate_job_run), '__call__') as call:
        call.return_value = cloud_deploy.TerminateJobRunResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.TerminateJobRunResponse())
        response = await client.terminate_job_run(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_terminate_job_run_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.terminate_job_run(cloud_deploy.TerminateJobRunRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.GetConfigRequest, dict])
def test_get_config(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_config), '__call__') as call:
        call.return_value = cloud_deploy.Config(name='name_value', default_skaffold_version='default_skaffold_version_value')
        response = client.get_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetConfigRequest()
    assert isinstance(response, cloud_deploy.Config)
    assert response.name == 'name_value'
    assert response.default_skaffold_version == 'default_skaffold_version_value'

def test_get_config_empty_call():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_config), '__call__') as call:
        client.get_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetConfigRequest()

@pytest.mark.asyncio
async def test_get_config_async(transport: str='grpc_asyncio', request_type=cloud_deploy.GetConfigRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.Config(name='name_value', default_skaffold_version='default_skaffold_version_value'))
        response = await client.get_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetConfigRequest()
    assert isinstance(response, cloud_deploy.Config)
    assert response.name == 'name_value'
    assert response.default_skaffold_version == 'default_skaffold_version_value'

@pytest.mark.asyncio
async def test_get_config_async_from_dict():
    await test_get_config_async(request_type=dict)

def test_get_config_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.GetConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_config), '__call__') as call:
        call.return_value = cloud_deploy.Config()
        client.get_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_config_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.GetConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.Config())
        await client.get_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_config_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_config), '__call__') as call:
        call.return_value = cloud_deploy.Config()
        client.get_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_config_flattened_error():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_config(cloud_deploy.GetConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_config_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_config), '__call__') as call:
        call.return_value = cloud_deploy.Config()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.Config())
        response = await client.get_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_config_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_config(cloud_deploy.GetConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.CreateAutomationRequest, dict])
def test_create_automation(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_automation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_automation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CreateAutomationRequest()
    assert isinstance(response, future.Future)

def test_create_automation_empty_call():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_automation), '__call__') as call:
        client.create_automation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CreateAutomationRequest()

@pytest.mark.asyncio
async def test_create_automation_async(transport: str='grpc_asyncio', request_type=cloud_deploy.CreateAutomationRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_automation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_automation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CreateAutomationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_automation_async_from_dict():
    await test_create_automation_async(request_type=dict)

def test_create_automation_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.CreateAutomationRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_automation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_automation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_automation_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.CreateAutomationRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_automation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_automation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_automation_flattened():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_automation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_automation(parent='parent_value', automation=cloud_deploy.Automation(name='name_value'), automation_id='automation_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].automation
        mock_val = cloud_deploy.Automation(name='name_value')
        assert arg == mock_val
        arg = args[0].automation_id
        mock_val = 'automation_id_value'
        assert arg == mock_val

def test_create_automation_flattened_error():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_automation(cloud_deploy.CreateAutomationRequest(), parent='parent_value', automation=cloud_deploy.Automation(name='name_value'), automation_id='automation_id_value')

@pytest.mark.asyncio
async def test_create_automation_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_automation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_automation(parent='parent_value', automation=cloud_deploy.Automation(name='name_value'), automation_id='automation_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].automation
        mock_val = cloud_deploy.Automation(name='name_value')
        assert arg == mock_val
        arg = args[0].automation_id
        mock_val = 'automation_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_automation_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_automation(cloud_deploy.CreateAutomationRequest(), parent='parent_value', automation=cloud_deploy.Automation(name='name_value'), automation_id='automation_id_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.UpdateAutomationRequest, dict])
def test_update_automation(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_automation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_automation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.UpdateAutomationRequest()
    assert isinstance(response, future.Future)

def test_update_automation_empty_call():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_automation), '__call__') as call:
        client.update_automation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.UpdateAutomationRequest()

@pytest.mark.asyncio
async def test_update_automation_async(transport: str='grpc_asyncio', request_type=cloud_deploy.UpdateAutomationRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_automation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_automation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.UpdateAutomationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_automation_async_from_dict():
    await test_update_automation_async(request_type=dict)

def test_update_automation_field_headers():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.UpdateAutomationRequest()
    request.automation.name = 'name_value'
    with mock.patch.object(type(client.transport.update_automation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_automation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'automation.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_automation_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.UpdateAutomationRequest()
    request.automation.name = 'name_value'
    with mock.patch.object(type(client.transport.update_automation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_automation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'automation.name=name_value') in kw['metadata']

def test_update_automation_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_automation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_automation(automation=cloud_deploy.Automation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].automation
        mock_val = cloud_deploy.Automation(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_automation_flattened_error():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_automation(cloud_deploy.UpdateAutomationRequest(), automation=cloud_deploy.Automation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_automation_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_automation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_automation(automation=cloud_deploy.Automation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].automation
        mock_val = cloud_deploy.Automation(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_automation_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_automation(cloud_deploy.UpdateAutomationRequest(), automation=cloud_deploy.Automation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [cloud_deploy.DeleteAutomationRequest, dict])
def test_delete_automation(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_automation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_automation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.DeleteAutomationRequest()
    assert isinstance(response, future.Future)

def test_delete_automation_empty_call():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_automation), '__call__') as call:
        client.delete_automation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.DeleteAutomationRequest()

@pytest.mark.asyncio
async def test_delete_automation_async(transport: str='grpc_asyncio', request_type=cloud_deploy.DeleteAutomationRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_automation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_automation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.DeleteAutomationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_automation_async_from_dict():
    await test_delete_automation_async(request_type=dict)

def test_delete_automation_field_headers():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.DeleteAutomationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_automation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_automation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_automation_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.DeleteAutomationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_automation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_automation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_automation_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_automation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_automation(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_automation_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_automation(cloud_deploy.DeleteAutomationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_automation_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_automation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_automation(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_automation_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_automation(cloud_deploy.DeleteAutomationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.GetAutomationRequest, dict])
def test_get_automation(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_automation), '__call__') as call:
        call.return_value = cloud_deploy.Automation(name='name_value', uid='uid_value', description='description_value', etag='etag_value', suspended=True, service_account='service_account_value')
        response = client.get_automation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetAutomationRequest()
    assert isinstance(response, cloud_deploy.Automation)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.etag == 'etag_value'
    assert response.suspended is True
    assert response.service_account == 'service_account_value'

def test_get_automation_empty_call():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_automation), '__call__') as call:
        client.get_automation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetAutomationRequest()

@pytest.mark.asyncio
async def test_get_automation_async(transport: str='grpc_asyncio', request_type=cloud_deploy.GetAutomationRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_automation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.Automation(name='name_value', uid='uid_value', description='description_value', etag='etag_value', suspended=True, service_account='service_account_value'))
        response = await client.get_automation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetAutomationRequest()
    assert isinstance(response, cloud_deploy.Automation)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.etag == 'etag_value'
    assert response.suspended is True
    assert response.service_account == 'service_account_value'

@pytest.mark.asyncio
async def test_get_automation_async_from_dict():
    await test_get_automation_async(request_type=dict)

def test_get_automation_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.GetAutomationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_automation), '__call__') as call:
        call.return_value = cloud_deploy.Automation()
        client.get_automation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_automation_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.GetAutomationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_automation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.Automation())
        await client.get_automation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_automation_flattened():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_automation), '__call__') as call:
        call.return_value = cloud_deploy.Automation()
        client.get_automation(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_automation_flattened_error():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_automation(cloud_deploy.GetAutomationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_automation_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_automation), '__call__') as call:
        call.return_value = cloud_deploy.Automation()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.Automation())
        response = await client.get_automation(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_automation_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_automation(cloud_deploy.GetAutomationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.ListAutomationsRequest, dict])
def test_list_automations(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_automations), '__call__') as call:
        call.return_value = cloud_deploy.ListAutomationsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_automations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListAutomationsRequest()
    assert isinstance(response, pagers.ListAutomationsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_automations_empty_call():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_automations), '__call__') as call:
        client.list_automations()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListAutomationsRequest()

@pytest.mark.asyncio
async def test_list_automations_async(transport: str='grpc_asyncio', request_type=cloud_deploy.ListAutomationsRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_automations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListAutomationsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_automations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListAutomationsRequest()
    assert isinstance(response, pagers.ListAutomationsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_automations_async_from_dict():
    await test_list_automations_async(request_type=dict)

def test_list_automations_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.ListAutomationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_automations), '__call__') as call:
        call.return_value = cloud_deploy.ListAutomationsResponse()
        client.list_automations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_automations_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.ListAutomationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_automations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListAutomationsResponse())
        await client.list_automations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_automations_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_automations), '__call__') as call:
        call.return_value = cloud_deploy.ListAutomationsResponse()
        client.list_automations(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_automations_flattened_error():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_automations(cloud_deploy.ListAutomationsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_automations_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_automations), '__call__') as call:
        call.return_value = cloud_deploy.ListAutomationsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListAutomationsResponse())
        response = await client.list_automations(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_automations_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_automations(cloud_deploy.ListAutomationsRequest(), parent='parent_value')

def test_list_automations_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_automations), '__call__') as call:
        call.side_effect = (cloud_deploy.ListAutomationsResponse(automations=[cloud_deploy.Automation(), cloud_deploy.Automation(), cloud_deploy.Automation()], next_page_token='abc'), cloud_deploy.ListAutomationsResponse(automations=[], next_page_token='def'), cloud_deploy.ListAutomationsResponse(automations=[cloud_deploy.Automation()], next_page_token='ghi'), cloud_deploy.ListAutomationsResponse(automations=[cloud_deploy.Automation(), cloud_deploy.Automation()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_automations(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_deploy.Automation) for i in results))

def test_list_automations_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_automations), '__call__') as call:
        call.side_effect = (cloud_deploy.ListAutomationsResponse(automations=[cloud_deploy.Automation(), cloud_deploy.Automation(), cloud_deploy.Automation()], next_page_token='abc'), cloud_deploy.ListAutomationsResponse(automations=[], next_page_token='def'), cloud_deploy.ListAutomationsResponse(automations=[cloud_deploy.Automation()], next_page_token='ghi'), cloud_deploy.ListAutomationsResponse(automations=[cloud_deploy.Automation(), cloud_deploy.Automation()]), RuntimeError)
        pages = list(client.list_automations(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_automations_async_pager():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_automations), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_deploy.ListAutomationsResponse(automations=[cloud_deploy.Automation(), cloud_deploy.Automation(), cloud_deploy.Automation()], next_page_token='abc'), cloud_deploy.ListAutomationsResponse(automations=[], next_page_token='def'), cloud_deploy.ListAutomationsResponse(automations=[cloud_deploy.Automation()], next_page_token='ghi'), cloud_deploy.ListAutomationsResponse(automations=[cloud_deploy.Automation(), cloud_deploy.Automation()]), RuntimeError)
        async_pager = await client.list_automations(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, cloud_deploy.Automation) for i in responses))

@pytest.mark.asyncio
async def test_list_automations_async_pages():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_automations), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_deploy.ListAutomationsResponse(automations=[cloud_deploy.Automation(), cloud_deploy.Automation(), cloud_deploy.Automation()], next_page_token='abc'), cloud_deploy.ListAutomationsResponse(automations=[], next_page_token='def'), cloud_deploy.ListAutomationsResponse(automations=[cloud_deploy.Automation()], next_page_token='ghi'), cloud_deploy.ListAutomationsResponse(automations=[cloud_deploy.Automation(), cloud_deploy.Automation()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_automations(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_deploy.GetAutomationRunRequest, dict])
def test_get_automation_run(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_automation_run), '__call__') as call:
        call.return_value = cloud_deploy.AutomationRun(name='name_value', etag='etag_value', service_account='service_account_value', target_id='target_id_value', state=cloud_deploy.AutomationRun.State.SUCCEEDED, state_description='state_description_value', rule_id='rule_id_value', automation_id='automation_id_value')
        response = client.get_automation_run(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetAutomationRunRequest()
    assert isinstance(response, cloud_deploy.AutomationRun)
    assert response.name == 'name_value'
    assert response.etag == 'etag_value'
    assert response.service_account == 'service_account_value'
    assert response.target_id == 'target_id_value'
    assert response.state == cloud_deploy.AutomationRun.State.SUCCEEDED
    assert response.state_description == 'state_description_value'
    assert response.rule_id == 'rule_id_value'
    assert response.automation_id == 'automation_id_value'

def test_get_automation_run_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_automation_run), '__call__') as call:
        client.get_automation_run()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetAutomationRunRequest()

@pytest.mark.asyncio
async def test_get_automation_run_async(transport: str='grpc_asyncio', request_type=cloud_deploy.GetAutomationRunRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_automation_run), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.AutomationRun(name='name_value', etag='etag_value', service_account='service_account_value', target_id='target_id_value', state=cloud_deploy.AutomationRun.State.SUCCEEDED, state_description='state_description_value', rule_id='rule_id_value', automation_id='automation_id_value'))
        response = await client.get_automation_run(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.GetAutomationRunRequest()
    assert isinstance(response, cloud_deploy.AutomationRun)
    assert response.name == 'name_value'
    assert response.etag == 'etag_value'
    assert response.service_account == 'service_account_value'
    assert response.target_id == 'target_id_value'
    assert response.state == cloud_deploy.AutomationRun.State.SUCCEEDED
    assert response.state_description == 'state_description_value'
    assert response.rule_id == 'rule_id_value'
    assert response.automation_id == 'automation_id_value'

@pytest.mark.asyncio
async def test_get_automation_run_async_from_dict():
    await test_get_automation_run_async(request_type=dict)

def test_get_automation_run_field_headers():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.GetAutomationRunRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_automation_run), '__call__') as call:
        call.return_value = cloud_deploy.AutomationRun()
        client.get_automation_run(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_automation_run_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.GetAutomationRunRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_automation_run), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.AutomationRun())
        await client.get_automation_run(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_automation_run_flattened():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_automation_run), '__call__') as call:
        call.return_value = cloud_deploy.AutomationRun()
        client.get_automation_run(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_automation_run_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_automation_run(cloud_deploy.GetAutomationRunRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_automation_run_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_automation_run), '__call__') as call:
        call.return_value = cloud_deploy.AutomationRun()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.AutomationRun())
        response = await client.get_automation_run(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_automation_run_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_automation_run(cloud_deploy.GetAutomationRunRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.ListAutomationRunsRequest, dict])
def test_list_automation_runs(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_automation_runs), '__call__') as call:
        call.return_value = cloud_deploy.ListAutomationRunsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_automation_runs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListAutomationRunsRequest()
    assert isinstance(response, pagers.ListAutomationRunsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_automation_runs_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_automation_runs), '__call__') as call:
        client.list_automation_runs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListAutomationRunsRequest()

@pytest.mark.asyncio
async def test_list_automation_runs_async(transport: str='grpc_asyncio', request_type=cloud_deploy.ListAutomationRunsRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_automation_runs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListAutomationRunsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_automation_runs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.ListAutomationRunsRequest()
    assert isinstance(response, pagers.ListAutomationRunsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_automation_runs_async_from_dict():
    await test_list_automation_runs_async(request_type=dict)

def test_list_automation_runs_field_headers():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.ListAutomationRunsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_automation_runs), '__call__') as call:
        call.return_value = cloud_deploy.ListAutomationRunsResponse()
        client.list_automation_runs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_automation_runs_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.ListAutomationRunsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_automation_runs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListAutomationRunsResponse())
        await client.list_automation_runs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_automation_runs_flattened():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_automation_runs), '__call__') as call:
        call.return_value = cloud_deploy.ListAutomationRunsResponse()
        client.list_automation_runs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_automation_runs_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_automation_runs(cloud_deploy.ListAutomationRunsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_automation_runs_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_automation_runs), '__call__') as call:
        call.return_value = cloud_deploy.ListAutomationRunsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.ListAutomationRunsResponse())
        response = await client.list_automation_runs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_automation_runs_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_automation_runs(cloud_deploy.ListAutomationRunsRequest(), parent='parent_value')

def test_list_automation_runs_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_automation_runs), '__call__') as call:
        call.side_effect = (cloud_deploy.ListAutomationRunsResponse(automation_runs=[cloud_deploy.AutomationRun(), cloud_deploy.AutomationRun(), cloud_deploy.AutomationRun()], next_page_token='abc'), cloud_deploy.ListAutomationRunsResponse(automation_runs=[], next_page_token='def'), cloud_deploy.ListAutomationRunsResponse(automation_runs=[cloud_deploy.AutomationRun()], next_page_token='ghi'), cloud_deploy.ListAutomationRunsResponse(automation_runs=[cloud_deploy.AutomationRun(), cloud_deploy.AutomationRun()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_automation_runs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_deploy.AutomationRun) for i in results))

def test_list_automation_runs_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_automation_runs), '__call__') as call:
        call.side_effect = (cloud_deploy.ListAutomationRunsResponse(automation_runs=[cloud_deploy.AutomationRun(), cloud_deploy.AutomationRun(), cloud_deploy.AutomationRun()], next_page_token='abc'), cloud_deploy.ListAutomationRunsResponse(automation_runs=[], next_page_token='def'), cloud_deploy.ListAutomationRunsResponse(automation_runs=[cloud_deploy.AutomationRun()], next_page_token='ghi'), cloud_deploy.ListAutomationRunsResponse(automation_runs=[cloud_deploy.AutomationRun(), cloud_deploy.AutomationRun()]), RuntimeError)
        pages = list(client.list_automation_runs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_automation_runs_async_pager():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_automation_runs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_deploy.ListAutomationRunsResponse(automation_runs=[cloud_deploy.AutomationRun(), cloud_deploy.AutomationRun(), cloud_deploy.AutomationRun()], next_page_token='abc'), cloud_deploy.ListAutomationRunsResponse(automation_runs=[], next_page_token='def'), cloud_deploy.ListAutomationRunsResponse(automation_runs=[cloud_deploy.AutomationRun()], next_page_token='ghi'), cloud_deploy.ListAutomationRunsResponse(automation_runs=[cloud_deploy.AutomationRun(), cloud_deploy.AutomationRun()]), RuntimeError)
        async_pager = await client.list_automation_runs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, cloud_deploy.AutomationRun) for i in responses))

@pytest.mark.asyncio
async def test_list_automation_runs_async_pages():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_automation_runs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_deploy.ListAutomationRunsResponse(automation_runs=[cloud_deploy.AutomationRun(), cloud_deploy.AutomationRun(), cloud_deploy.AutomationRun()], next_page_token='abc'), cloud_deploy.ListAutomationRunsResponse(automation_runs=[], next_page_token='def'), cloud_deploy.ListAutomationRunsResponse(automation_runs=[cloud_deploy.AutomationRun()], next_page_token='ghi'), cloud_deploy.ListAutomationRunsResponse(automation_runs=[cloud_deploy.AutomationRun(), cloud_deploy.AutomationRun()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_automation_runs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_deploy.CancelAutomationRunRequest, dict])
def test_cancel_automation_run(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.cancel_automation_run), '__call__') as call:
        call.return_value = cloud_deploy.CancelAutomationRunResponse()
        response = client.cancel_automation_run(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CancelAutomationRunRequest()
    assert isinstance(response, cloud_deploy.CancelAutomationRunResponse)

def test_cancel_automation_run_empty_call():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.cancel_automation_run), '__call__') as call:
        client.cancel_automation_run()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CancelAutomationRunRequest()

@pytest.mark.asyncio
async def test_cancel_automation_run_async(transport: str='grpc_asyncio', request_type=cloud_deploy.CancelAutomationRunRequest):
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.cancel_automation_run), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.CancelAutomationRunResponse())
        response = await client.cancel_automation_run(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_deploy.CancelAutomationRunRequest()
    assert isinstance(response, cloud_deploy.CancelAutomationRunResponse)

@pytest.mark.asyncio
async def test_cancel_automation_run_async_from_dict():
    await test_cancel_automation_run_async(request_type=dict)

def test_cancel_automation_run_field_headers():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.CancelAutomationRunRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.cancel_automation_run), '__call__') as call:
        call.return_value = cloud_deploy.CancelAutomationRunResponse()
        client.cancel_automation_run(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_cancel_automation_run_field_headers_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_deploy.CancelAutomationRunRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.cancel_automation_run), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.CancelAutomationRunResponse())
        await client.cancel_automation_run(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_cancel_automation_run_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_automation_run), '__call__') as call:
        call.return_value = cloud_deploy.CancelAutomationRunResponse()
        client.cancel_automation_run(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_cancel_automation_run_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.cancel_automation_run(cloud_deploy.CancelAutomationRunRequest(), name='name_value')

@pytest.mark.asyncio
async def test_cancel_automation_run_flattened_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_automation_run), '__call__') as call:
        call.return_value = cloud_deploy.CancelAutomationRunResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_deploy.CancelAutomationRunResponse())
        response = await client.cancel_automation_run(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_cancel_automation_run_flattened_error_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.cancel_automation_run(cloud_deploy.CancelAutomationRunRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_deploy.ListDeliveryPipelinesRequest, dict])
def test_list_delivery_pipelines_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.ListDeliveryPipelinesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.ListDeliveryPipelinesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_delivery_pipelines(request)
    assert isinstance(response, pagers.ListDeliveryPipelinesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_delivery_pipelines_rest_required_fields(request_type=cloud_deploy.ListDeliveryPipelinesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_delivery_pipelines._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_delivery_pipelines._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.ListDeliveryPipelinesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.ListDeliveryPipelinesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_delivery_pipelines(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_delivery_pipelines_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_delivery_pipelines._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_delivery_pipelines_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_list_delivery_pipelines') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_list_delivery_pipelines') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.ListDeliveryPipelinesRequest.pb(cloud_deploy.ListDeliveryPipelinesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.ListDeliveryPipelinesResponse.to_json(cloud_deploy.ListDeliveryPipelinesResponse())
        request = cloud_deploy.ListDeliveryPipelinesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.ListDeliveryPipelinesResponse()
        client.list_delivery_pipelines(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_delivery_pipelines_rest_bad_request(transport: str='rest', request_type=cloud_deploy.ListDeliveryPipelinesRequest):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_delivery_pipelines(request)

def test_list_delivery_pipelines_rest_flattened():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.ListDeliveryPipelinesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.ListDeliveryPipelinesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_delivery_pipelines(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/deliveryPipelines' % client.transport._host, args[1])

def test_list_delivery_pipelines_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_delivery_pipelines(cloud_deploy.ListDeliveryPipelinesRequest(), parent='parent_value')

def test_list_delivery_pipelines_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[cloud_deploy.DeliveryPipeline(), cloud_deploy.DeliveryPipeline(), cloud_deploy.DeliveryPipeline()], next_page_token='abc'), cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[], next_page_token='def'), cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[cloud_deploy.DeliveryPipeline()], next_page_token='ghi'), cloud_deploy.ListDeliveryPipelinesResponse(delivery_pipelines=[cloud_deploy.DeliveryPipeline(), cloud_deploy.DeliveryPipeline()]))
        response = response + response
        response = tuple((cloud_deploy.ListDeliveryPipelinesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_delivery_pipelines(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_deploy.DeliveryPipeline) for i in results))
        pages = list(client.list_delivery_pipelines(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_deploy.GetDeliveryPipelineRequest, dict])
def test_get_delivery_pipeline_rest(request_type):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.DeliveryPipeline(name='name_value', uid='uid_value', description='description_value', etag='etag_value', suspended=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.DeliveryPipeline.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_delivery_pipeline(request)
    assert isinstance(response, cloud_deploy.DeliveryPipeline)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.etag == 'etag_value'
    assert response.suspended is True

def test_get_delivery_pipeline_rest_required_fields(request_type=cloud_deploy.GetDeliveryPipelineRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_delivery_pipeline._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_delivery_pipeline._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.DeliveryPipeline()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.DeliveryPipeline.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_delivery_pipeline(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_delivery_pipeline_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_delivery_pipeline._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_delivery_pipeline_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_get_delivery_pipeline') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_get_delivery_pipeline') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.GetDeliveryPipelineRequest.pb(cloud_deploy.GetDeliveryPipelineRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.DeliveryPipeline.to_json(cloud_deploy.DeliveryPipeline())
        request = cloud_deploy.GetDeliveryPipelineRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.DeliveryPipeline()
        client.get_delivery_pipeline(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_delivery_pipeline_rest_bad_request(transport: str='rest', request_type=cloud_deploy.GetDeliveryPipelineRequest):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_delivery_pipeline(request)

def test_get_delivery_pipeline_rest_flattened():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.DeliveryPipeline()
        sample_request = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.DeliveryPipeline.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_delivery_pipeline(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/deliveryPipelines/*}' % client.transport._host, args[1])

def test_get_delivery_pipeline_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_delivery_pipeline(cloud_deploy.GetDeliveryPipelineRequest(), name='name_value')

def test_get_delivery_pipeline_rest_error():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.CreateDeliveryPipelineRequest, dict])
def test_create_delivery_pipeline_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['delivery_pipeline'] = {'name': 'name_value', 'uid': 'uid_value', 'description': 'description_value', 'annotations': {}, 'labels': {}, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'serial_pipeline': {'stages': [{'target_id': 'target_id_value', 'profiles': ['profiles_value1', 'profiles_value2'], 'strategy': {'standard': {'verify': True, 'predeploy': {'actions': ['actions_value1', 'actions_value2']}, 'postdeploy': {'actions': ['actions_value1', 'actions_value2']}}, 'canary': {'runtime_config': {'kubernetes': {'gateway_service_mesh': {'http_route': 'http_route_value', 'service': 'service_value', 'deployment': 'deployment_value', 'route_update_wait_time': {'seconds': 751, 'nanos': 543}}, 'service_networking': {'service': 'service_value', 'deployment': 'deployment_value', 'disable_pod_overprovisioning': True}}, 'cloud_run': {'automatic_traffic_control': True}}, 'canary_deployment': {'percentages': [1170, 1171], 'verify': True, 'predeploy': {}, 'postdeploy': {}}, 'custom_canary_deployment': {'phase_configs': [{'phase_id': 'phase_id_value', 'percentage': 1054, 'profiles': ['profiles_value1', 'profiles_value2'], 'verify': True, 'predeploy': {}, 'postdeploy': {}}]}}}, 'deploy_parameters': [{'values': {}, 'match_target_labels': {}}]}]}, 'condition': {'pipeline_ready_condition': {'status': True, 'update_time': {}}, 'targets_present_condition': {'status': True, 'missing_targets': ['missing_targets_value1', 'missing_targets_value2'], 'update_time': {}}, 'targets_type_condition': {'status': True, 'error_details': 'error_details_value'}}, 'etag': 'etag_value', 'suspended': True}
    test_field = cloud_deploy.CreateDeliveryPipelineRequest.meta.fields['delivery_pipeline']

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
    for (field, value) in request_init['delivery_pipeline'].items():
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
                for i in range(0, len(request_init['delivery_pipeline'][field])):
                    del request_init['delivery_pipeline'][field][i][subfield]
            else:
                del request_init['delivery_pipeline'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_delivery_pipeline(request)
    assert response.operation.name == 'operations/spam'

def test_create_delivery_pipeline_rest_required_fields(request_type=cloud_deploy.CreateDeliveryPipelineRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['delivery_pipeline_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'deliveryPipelineId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_delivery_pipeline._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'deliveryPipelineId' in jsonified_request
    assert jsonified_request['deliveryPipelineId'] == request_init['delivery_pipeline_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['deliveryPipelineId'] = 'delivery_pipeline_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_delivery_pipeline._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('delivery_pipeline_id', 'request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'deliveryPipelineId' in jsonified_request
    assert jsonified_request['deliveryPipelineId'] == 'delivery_pipeline_id_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_delivery_pipeline(request)
            expected_params = [('deliveryPipelineId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_delivery_pipeline_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_delivery_pipeline._get_unset_required_fields({})
    assert set(unset_fields) == set(('deliveryPipelineId', 'requestId', 'validateOnly')) & set(('parent', 'deliveryPipelineId', 'deliveryPipeline'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_delivery_pipeline_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudDeployRestInterceptor, 'post_create_delivery_pipeline') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_create_delivery_pipeline') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.CreateDeliveryPipelineRequest.pb(cloud_deploy.CreateDeliveryPipelineRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_deploy.CreateDeliveryPipelineRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_delivery_pipeline(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_delivery_pipeline_rest_bad_request(transport: str='rest', request_type=cloud_deploy.CreateDeliveryPipelineRequest):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_delivery_pipeline(request)

def test_create_delivery_pipeline_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', delivery_pipeline=cloud_deploy.DeliveryPipeline(name='name_value'), delivery_pipeline_id='delivery_pipeline_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_delivery_pipeline(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/deliveryPipelines' % client.transport._host, args[1])

def test_create_delivery_pipeline_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_delivery_pipeline(cloud_deploy.CreateDeliveryPipelineRequest(), parent='parent_value', delivery_pipeline=cloud_deploy.DeliveryPipeline(name='name_value'), delivery_pipeline_id='delivery_pipeline_id_value')

def test_create_delivery_pipeline_rest_error():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.UpdateDeliveryPipelineRequest, dict])
def test_update_delivery_pipeline_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'delivery_pipeline': {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}}
    request_init['delivery_pipeline'] = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3', 'uid': 'uid_value', 'description': 'description_value', 'annotations': {}, 'labels': {}, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'serial_pipeline': {'stages': [{'target_id': 'target_id_value', 'profiles': ['profiles_value1', 'profiles_value2'], 'strategy': {'standard': {'verify': True, 'predeploy': {'actions': ['actions_value1', 'actions_value2']}, 'postdeploy': {'actions': ['actions_value1', 'actions_value2']}}, 'canary': {'runtime_config': {'kubernetes': {'gateway_service_mesh': {'http_route': 'http_route_value', 'service': 'service_value', 'deployment': 'deployment_value', 'route_update_wait_time': {'seconds': 751, 'nanos': 543}}, 'service_networking': {'service': 'service_value', 'deployment': 'deployment_value', 'disable_pod_overprovisioning': True}}, 'cloud_run': {'automatic_traffic_control': True}}, 'canary_deployment': {'percentages': [1170, 1171], 'verify': True, 'predeploy': {}, 'postdeploy': {}}, 'custom_canary_deployment': {'phase_configs': [{'phase_id': 'phase_id_value', 'percentage': 1054, 'profiles': ['profiles_value1', 'profiles_value2'], 'verify': True, 'predeploy': {}, 'postdeploy': {}}]}}}, 'deploy_parameters': [{'values': {}, 'match_target_labels': {}}]}]}, 'condition': {'pipeline_ready_condition': {'status': True, 'update_time': {}}, 'targets_present_condition': {'status': True, 'missing_targets': ['missing_targets_value1', 'missing_targets_value2'], 'update_time': {}}, 'targets_type_condition': {'status': True, 'error_details': 'error_details_value'}}, 'etag': 'etag_value', 'suspended': True}
    test_field = cloud_deploy.UpdateDeliveryPipelineRequest.meta.fields['delivery_pipeline']

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
    for (field, value) in request_init['delivery_pipeline'].items():
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
                for i in range(0, len(request_init['delivery_pipeline'][field])):
                    del request_init['delivery_pipeline'][field][i][subfield]
            else:
                del request_init['delivery_pipeline'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_delivery_pipeline(request)
    assert response.operation.name == 'operations/spam'

def test_update_delivery_pipeline_rest_required_fields(request_type=cloud_deploy.UpdateDeliveryPipelineRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_delivery_pipeline._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_delivery_pipeline._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'request_id', 'update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_delivery_pipeline(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_delivery_pipeline_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_delivery_pipeline._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'requestId', 'updateMask', 'validateOnly')) & set(('updateMask', 'deliveryPipeline'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_delivery_pipeline_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudDeployRestInterceptor, 'post_update_delivery_pipeline') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_update_delivery_pipeline') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.UpdateDeliveryPipelineRequest.pb(cloud_deploy.UpdateDeliveryPipelineRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_deploy.UpdateDeliveryPipelineRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_delivery_pipeline(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_delivery_pipeline_rest_bad_request(transport: str='rest', request_type=cloud_deploy.UpdateDeliveryPipelineRequest):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'delivery_pipeline': {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_delivery_pipeline(request)

def test_update_delivery_pipeline_rest_flattened():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'delivery_pipeline': {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}}
        mock_args = dict(delivery_pipeline=cloud_deploy.DeliveryPipeline(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_delivery_pipeline(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{delivery_pipeline.name=projects/*/locations/*/deliveryPipelines/*}' % client.transport._host, args[1])

def test_update_delivery_pipeline_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_delivery_pipeline(cloud_deploy.UpdateDeliveryPipelineRequest(), delivery_pipeline=cloud_deploy.DeliveryPipeline(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_delivery_pipeline_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.DeleteDeliveryPipelineRequest, dict])
def test_delete_delivery_pipeline_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_delivery_pipeline(request)
    assert response.operation.name == 'operations/spam'

def test_delete_delivery_pipeline_rest_required_fields(request_type=cloud_deploy.DeleteDeliveryPipelineRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_delivery_pipeline._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_delivery_pipeline._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'etag', 'force', 'request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_delivery_pipeline(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_delivery_pipeline_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_delivery_pipeline._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'etag', 'force', 'requestId', 'validateOnly')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_delivery_pipeline_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudDeployRestInterceptor, 'post_delete_delivery_pipeline') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_delete_delivery_pipeline') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.DeleteDeliveryPipelineRequest.pb(cloud_deploy.DeleteDeliveryPipelineRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_deploy.DeleteDeliveryPipelineRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_delivery_pipeline(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_delivery_pipeline_rest_bad_request(transport: str='rest', request_type=cloud_deploy.DeleteDeliveryPipelineRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_delivery_pipeline(request)

def test_delete_delivery_pipeline_rest_flattened():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_delivery_pipeline(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/deliveryPipelines/*}' % client.transport._host, args[1])

def test_delete_delivery_pipeline_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_delivery_pipeline(cloud_deploy.DeleteDeliveryPipelineRequest(), name='name_value')

def test_delete_delivery_pipeline_rest_error():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.ListTargetsRequest, dict])
def test_list_targets_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.ListTargetsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.ListTargetsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_targets(request)
    assert isinstance(response, pagers.ListTargetsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_targets_rest_required_fields(request_type=cloud_deploy.ListTargetsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_targets._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_targets._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.ListTargetsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.ListTargetsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_targets(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_targets_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_targets._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_targets_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_list_targets') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_list_targets') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.ListTargetsRequest.pb(cloud_deploy.ListTargetsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.ListTargetsResponse.to_json(cloud_deploy.ListTargetsResponse())
        request = cloud_deploy.ListTargetsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.ListTargetsResponse()
        client.list_targets(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_targets_rest_bad_request(transport: str='rest', request_type=cloud_deploy.ListTargetsRequest):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_targets(request)

def test_list_targets_rest_flattened():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.ListTargetsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.ListTargetsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_targets(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/targets' % client.transport._host, args[1])

def test_list_targets_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_targets(cloud_deploy.ListTargetsRequest(), parent='parent_value')

def test_list_targets_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (cloud_deploy.ListTargetsResponse(targets=[cloud_deploy.Target(), cloud_deploy.Target(), cloud_deploy.Target()], next_page_token='abc'), cloud_deploy.ListTargetsResponse(targets=[], next_page_token='def'), cloud_deploy.ListTargetsResponse(targets=[cloud_deploy.Target()], next_page_token='ghi'), cloud_deploy.ListTargetsResponse(targets=[cloud_deploy.Target(), cloud_deploy.Target()]))
        response = response + response
        response = tuple((cloud_deploy.ListTargetsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_targets(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_deploy.Target) for i in results))
        pages = list(client.list_targets(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_deploy.RollbackTargetRequest, dict])
def test_rollback_target_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.RollbackTargetResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.RollbackTargetResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.rollback_target(request)
    assert isinstance(response, cloud_deploy.RollbackTargetResponse)

def test_rollback_target_rest_required_fields(request_type=cloud_deploy.RollbackTargetRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['target_id'] = ''
    request_init['rollout_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).rollback_target._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['targetId'] = 'target_id_value'
    jsonified_request['rolloutId'] = 'rollout_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).rollback_target._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'targetId' in jsonified_request
    assert jsonified_request['targetId'] == 'target_id_value'
    assert 'rolloutId' in jsonified_request
    assert jsonified_request['rolloutId'] == 'rollout_id_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.RollbackTargetResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.RollbackTargetResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.rollback_target(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_rollback_target_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.rollback_target._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'targetId', 'rolloutId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_rollback_target_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_rollback_target') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_rollback_target') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.RollbackTargetRequest.pb(cloud_deploy.RollbackTargetRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.RollbackTargetResponse.to_json(cloud_deploy.RollbackTargetResponse())
        request = cloud_deploy.RollbackTargetRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.RollbackTargetResponse()
        client.rollback_target(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_rollback_target_rest_bad_request(transport: str='rest', request_type=cloud_deploy.RollbackTargetRequest):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.rollback_target(request)

def test_rollback_target_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.RollbackTargetResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
        mock_args = dict(name='name_value', target_id='target_id_value', rollout_id='rollout_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.RollbackTargetResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.rollback_target(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/deliveryPipelines/*}:rollbackTarget' % client.transport._host, args[1])

def test_rollback_target_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.rollback_target(cloud_deploy.RollbackTargetRequest(), name='name_value', target_id='target_id_value', rollout_id='rollout_id_value')

def test_rollback_target_rest_error():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.GetTargetRequest, dict])
def test_get_target_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/targets/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.Target(name='name_value', target_id='target_id_value', uid='uid_value', description='description_value', require_approval=True, etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.Target.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_target(request)
    assert isinstance(response, cloud_deploy.Target)
    assert response.name == 'name_value'
    assert response.target_id == 'target_id_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.require_approval is True
    assert response.etag == 'etag_value'

def test_get_target_rest_required_fields(request_type=cloud_deploy.GetTargetRequest):
    if False:
        return 10
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_target._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_target._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.Target()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.Target.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_target(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_target_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_target._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_target_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_get_target') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_get_target') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.GetTargetRequest.pb(cloud_deploy.GetTargetRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.Target.to_json(cloud_deploy.Target())
        request = cloud_deploy.GetTargetRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.Target()
        client.get_target(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_target_rest_bad_request(transport: str='rest', request_type=cloud_deploy.GetTargetRequest):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/targets/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_target(request)

def test_get_target_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.Target()
        sample_request = {'name': 'projects/sample1/locations/sample2/targets/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.Target.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_target(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/targets/*}' % client.transport._host, args[1])

def test_get_target_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_target(cloud_deploy.GetTargetRequest(), name='name_value')

def test_get_target_rest_error():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.CreateTargetRequest, dict])
def test_create_target_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['target'] = {'name': 'name_value', 'target_id': 'target_id_value', 'uid': 'uid_value', 'description': 'description_value', 'annotations': {}, 'labels': {}, 'require_approval': True, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'gke': {'cluster': 'cluster_value', 'internal_ip': True}, 'anthos_cluster': {'membership': 'membership_value'}, 'run': {'location': 'location_value'}, 'multi_target': {'target_ids': ['target_ids_value1', 'target_ids_value2']}, 'etag': 'etag_value', 'execution_configs': [{'usages': [1], 'default_pool': {'service_account': 'service_account_value', 'artifact_storage': 'artifact_storage_value'}, 'private_pool': {'worker_pool': 'worker_pool_value', 'service_account': 'service_account_value', 'artifact_storage': 'artifact_storage_value'}, 'worker_pool': 'worker_pool_value', 'service_account': 'service_account_value', 'artifact_storage': 'artifact_storage_value', 'execution_timeout': {'seconds': 751, 'nanos': 543}}], 'deploy_parameters': {}}
    test_field = cloud_deploy.CreateTargetRequest.meta.fields['target']

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
    for (field, value) in request_init['target'].items():
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
                for i in range(0, len(request_init['target'][field])):
                    del request_init['target'][field][i][subfield]
            else:
                del request_init['target'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_target(request)
    assert response.operation.name == 'operations/spam'

def test_create_target_rest_required_fields(request_type=cloud_deploy.CreateTargetRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['target_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'targetId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_target._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'targetId' in jsonified_request
    assert jsonified_request['targetId'] == request_init['target_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['targetId'] = 'target_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_target._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'target_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'targetId' in jsonified_request
    assert jsonified_request['targetId'] == 'target_id_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_target(request)
            expected_params = [('targetId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_target_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_target._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'targetId', 'validateOnly')) & set(('parent', 'targetId', 'target'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_target_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudDeployRestInterceptor, 'post_create_target') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_create_target') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.CreateTargetRequest.pb(cloud_deploy.CreateTargetRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_deploy.CreateTargetRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_target(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_target_rest_bad_request(transport: str='rest', request_type=cloud_deploy.CreateTargetRequest):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_target(request)

def test_create_target_rest_flattened():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', target=cloud_deploy.Target(name='name_value'), target_id='target_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_target(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/targets' % client.transport._host, args[1])

def test_create_target_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_target(cloud_deploy.CreateTargetRequest(), parent='parent_value', target=cloud_deploy.Target(name='name_value'), target_id='target_id_value')

def test_create_target_rest_error():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.UpdateTargetRequest, dict])
def test_update_target_rest(request_type):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'target': {'name': 'projects/sample1/locations/sample2/targets/sample3'}}
    request_init['target'] = {'name': 'projects/sample1/locations/sample2/targets/sample3', 'target_id': 'target_id_value', 'uid': 'uid_value', 'description': 'description_value', 'annotations': {}, 'labels': {}, 'require_approval': True, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'gke': {'cluster': 'cluster_value', 'internal_ip': True}, 'anthos_cluster': {'membership': 'membership_value'}, 'run': {'location': 'location_value'}, 'multi_target': {'target_ids': ['target_ids_value1', 'target_ids_value2']}, 'etag': 'etag_value', 'execution_configs': [{'usages': [1], 'default_pool': {'service_account': 'service_account_value', 'artifact_storage': 'artifact_storage_value'}, 'private_pool': {'worker_pool': 'worker_pool_value', 'service_account': 'service_account_value', 'artifact_storage': 'artifact_storage_value'}, 'worker_pool': 'worker_pool_value', 'service_account': 'service_account_value', 'artifact_storage': 'artifact_storage_value', 'execution_timeout': {'seconds': 751, 'nanos': 543}}], 'deploy_parameters': {}}
    test_field = cloud_deploy.UpdateTargetRequest.meta.fields['target']

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
    for (field, value) in request_init['target'].items():
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
                for i in range(0, len(request_init['target'][field])):
                    del request_init['target'][field][i][subfield]
            else:
                del request_init['target'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_target(request)
    assert response.operation.name == 'operations/spam'

def test_update_target_rest_required_fields(request_type=cloud_deploy.UpdateTargetRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_target._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_target._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'request_id', 'update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_target(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_target_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_target._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'requestId', 'updateMask', 'validateOnly')) & set(('updateMask', 'target'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_target_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudDeployRestInterceptor, 'post_update_target') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_update_target') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.UpdateTargetRequest.pb(cloud_deploy.UpdateTargetRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_deploy.UpdateTargetRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_target(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_target_rest_bad_request(transport: str='rest', request_type=cloud_deploy.UpdateTargetRequest):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'target': {'name': 'projects/sample1/locations/sample2/targets/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_target(request)

def test_update_target_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'target': {'name': 'projects/sample1/locations/sample2/targets/sample3'}}
        mock_args = dict(target=cloud_deploy.Target(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_target(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{target.name=projects/*/locations/*/targets/*}' % client.transport._host, args[1])

def test_update_target_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_target(cloud_deploy.UpdateTargetRequest(), target=cloud_deploy.Target(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_target_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.DeleteTargetRequest, dict])
def test_delete_target_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/targets/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_target(request)
    assert response.operation.name == 'operations/spam'

def test_delete_target_rest_required_fields(request_type=cloud_deploy.DeleteTargetRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_target._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_target._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'etag', 'request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_target(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_target_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_target._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'etag', 'requestId', 'validateOnly')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_target_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudDeployRestInterceptor, 'post_delete_target') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_delete_target') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.DeleteTargetRequest.pb(cloud_deploy.DeleteTargetRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_deploy.DeleteTargetRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_target(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_target_rest_bad_request(transport: str='rest', request_type=cloud_deploy.DeleteTargetRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/targets/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_target(request)

def test_delete_target_rest_flattened():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/targets/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_target(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/targets/*}' % client.transport._host, args[1])

def test_delete_target_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_target(cloud_deploy.DeleteTargetRequest(), name='name_value')

def test_delete_target_rest_error():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.ListReleasesRequest, dict])
def test_list_releases_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.ListReleasesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.ListReleasesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_releases(request)
    assert isinstance(response, pagers.ListReleasesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_releases_rest_required_fields(request_type=cloud_deploy.ListReleasesRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_releases._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_releases._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.ListReleasesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.ListReleasesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_releases(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_releases_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_releases._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_releases_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_list_releases') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_list_releases') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.ListReleasesRequest.pb(cloud_deploy.ListReleasesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.ListReleasesResponse.to_json(cloud_deploy.ListReleasesResponse())
        request = cloud_deploy.ListReleasesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.ListReleasesResponse()
        client.list_releases(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_releases_rest_bad_request(transport: str='rest', request_type=cloud_deploy.ListReleasesRequest):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_releases(request)

def test_list_releases_rest_flattened():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.ListReleasesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.ListReleasesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_releases(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/deliveryPipelines/*}/releases' % client.transport._host, args[1])

def test_list_releases_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_releases(cloud_deploy.ListReleasesRequest(), parent='parent_value')

def test_list_releases_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (cloud_deploy.ListReleasesResponse(releases=[cloud_deploy.Release(), cloud_deploy.Release(), cloud_deploy.Release()], next_page_token='abc'), cloud_deploy.ListReleasesResponse(releases=[], next_page_token='def'), cloud_deploy.ListReleasesResponse(releases=[cloud_deploy.Release()], next_page_token='ghi'), cloud_deploy.ListReleasesResponse(releases=[cloud_deploy.Release(), cloud_deploy.Release()]))
        response = response + response
        response = tuple((cloud_deploy.ListReleasesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
        pager = client.list_releases(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_deploy.Release) for i in results))
        pages = list(client.list_releases(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_deploy.GetReleaseRequest, dict])
def test_get_release_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.Release(name='name_value', uid='uid_value', description='description_value', abandoned=True, skaffold_config_uri='skaffold_config_uri_value', skaffold_config_path='skaffold_config_path_value', render_state=cloud_deploy.Release.RenderState.SUCCEEDED, etag='etag_value', skaffold_version='skaffold_version_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.Release.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_release(request)
    assert isinstance(response, cloud_deploy.Release)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.abandoned is True
    assert response.skaffold_config_uri == 'skaffold_config_uri_value'
    assert response.skaffold_config_path == 'skaffold_config_path_value'
    assert response.render_state == cloud_deploy.Release.RenderState.SUCCEEDED
    assert response.etag == 'etag_value'
    assert response.skaffold_version == 'skaffold_version_value'

def test_get_release_rest_required_fields(request_type=cloud_deploy.GetReleaseRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_release._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_release._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.Release()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.Release.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_release(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_release_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_release._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_release_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_get_release') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_get_release') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.GetReleaseRequest.pb(cloud_deploy.GetReleaseRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.Release.to_json(cloud_deploy.Release())
        request = cloud_deploy.GetReleaseRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.Release()
        client.get_release(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_release_rest_bad_request(transport: str='rest', request_type=cloud_deploy.GetReleaseRequest):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_release(request)

def test_get_release_rest_flattened():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.Release()
        sample_request = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.Release.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_release(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/deliveryPipelines/*/releases/*}' % client.transport._host, args[1])

def test_get_release_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_release(cloud_deploy.GetReleaseRequest(), name='name_value')

def test_get_release_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.CreateReleaseRequest, dict])
def test_create_release_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
    request_init['release'] = {'name': 'name_value', 'uid': 'uid_value', 'description': 'description_value', 'annotations': {}, 'labels': {}, 'abandoned': True, 'create_time': {'seconds': 751, 'nanos': 543}, 'render_start_time': {}, 'render_end_time': {}, 'skaffold_config_uri': 'skaffold_config_uri_value', 'skaffold_config_path': 'skaffold_config_path_value', 'build_artifacts': [{'image': 'image_value', 'tag': 'tag_value'}], 'delivery_pipeline_snapshot': {'name': 'name_value', 'uid': 'uid_value', 'description': 'description_value', 'annotations': {}, 'labels': {}, 'create_time': {}, 'update_time': {}, 'serial_pipeline': {'stages': [{'target_id': 'target_id_value', 'profiles': ['profiles_value1', 'profiles_value2'], 'strategy': {'standard': {'verify': True, 'predeploy': {'actions': ['actions_value1', 'actions_value2']}, 'postdeploy': {'actions': ['actions_value1', 'actions_value2']}}, 'canary': {'runtime_config': {'kubernetes': {'gateway_service_mesh': {'http_route': 'http_route_value', 'service': 'service_value', 'deployment': 'deployment_value', 'route_update_wait_time': {'seconds': 751, 'nanos': 543}}, 'service_networking': {'service': 'service_value', 'deployment': 'deployment_value', 'disable_pod_overprovisioning': True}}, 'cloud_run': {'automatic_traffic_control': True}}, 'canary_deployment': {'percentages': [1170, 1171], 'verify': True, 'predeploy': {}, 'postdeploy': {}}, 'custom_canary_deployment': {'phase_configs': [{'phase_id': 'phase_id_value', 'percentage': 1054, 'profiles': ['profiles_value1', 'profiles_value2'], 'verify': True, 'predeploy': {}, 'postdeploy': {}}]}}}, 'deploy_parameters': [{'values': {}, 'match_target_labels': {}}]}]}, 'condition': {'pipeline_ready_condition': {'status': True, 'update_time': {}}, 'targets_present_condition': {'status': True, 'missing_targets': ['missing_targets_value1', 'missing_targets_value2'], 'update_time': {}}, 'targets_type_condition': {'status': True, 'error_details': 'error_details_value'}}, 'etag': 'etag_value', 'suspended': True}, 'target_snapshots': [{'name': 'name_value', 'target_id': 'target_id_value', 'uid': 'uid_value', 'description': 'description_value', 'annotations': {}, 'labels': {}, 'require_approval': True, 'create_time': {}, 'update_time': {}, 'gke': {'cluster': 'cluster_value', 'internal_ip': True}, 'anthos_cluster': {'membership': 'membership_value'}, 'run': {'location': 'location_value'}, 'multi_target': {'target_ids': ['target_ids_value1', 'target_ids_value2']}, 'etag': 'etag_value', 'execution_configs': [{'usages': [1], 'default_pool': {'service_account': 'service_account_value', 'artifact_storage': 'artifact_storage_value'}, 'private_pool': {'worker_pool': 'worker_pool_value', 'service_account': 'service_account_value', 'artifact_storage': 'artifact_storage_value'}, 'worker_pool': 'worker_pool_value', 'service_account': 'service_account_value', 'artifact_storage': 'artifact_storage_value', 'execution_timeout': {}}], 'deploy_parameters': {}}], 'render_state': 1, 'etag': 'etag_value', 'skaffold_version': 'skaffold_version_value', 'target_artifacts': {}, 'target_renders': {}, 'condition': {'release_ready_condition': {'status': True}, 'skaffold_supported_condition': {'status': True, 'skaffold_support_state': 1, 'maintenance_mode_time': {}, 'support_expiration_time': {}}}, 'deploy_parameters': {}}
    test_field = cloud_deploy.CreateReleaseRequest.meta.fields['release']

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
    for (field, value) in request_init['release'].items():
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
                for i in range(0, len(request_init['release'][field])):
                    del request_init['release'][field][i][subfield]
            else:
                del request_init['release'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_release(request)
    assert response.operation.name == 'operations/spam'

def test_create_release_rest_required_fields(request_type=cloud_deploy.CreateReleaseRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['release_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'releaseId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_release._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'releaseId' in jsonified_request
    assert jsonified_request['releaseId'] == request_init['release_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['releaseId'] = 'release_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_release._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('release_id', 'request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'releaseId' in jsonified_request
    assert jsonified_request['releaseId'] == 'release_id_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_release(request)
            expected_params = [('releaseId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_release_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_release._get_unset_required_fields({})
    assert set(unset_fields) == set(('releaseId', 'requestId', 'validateOnly')) & set(('parent', 'releaseId', 'release'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_release_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudDeployRestInterceptor, 'post_create_release') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_create_release') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.CreateReleaseRequest.pb(cloud_deploy.CreateReleaseRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_deploy.CreateReleaseRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_release(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_release_rest_bad_request(transport: str='rest', request_type=cloud_deploy.CreateReleaseRequest):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_release(request)

def test_create_release_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
        mock_args = dict(parent='parent_value', release=cloud_deploy.Release(name='name_value'), release_id='release_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_release(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/deliveryPipelines/*}/releases' % client.transport._host, args[1])

def test_create_release_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_release(cloud_deploy.CreateReleaseRequest(), parent='parent_value', release=cloud_deploy.Release(name='name_value'), release_id='release_id_value')

def test_create_release_rest_error():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.AbandonReleaseRequest, dict])
def test_abandon_release_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.AbandonReleaseResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.AbandonReleaseResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.abandon_release(request)
    assert isinstance(response, cloud_deploy.AbandonReleaseResponse)

def test_abandon_release_rest_required_fields(request_type=cloud_deploy.AbandonReleaseRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).abandon_release._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).abandon_release._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.AbandonReleaseResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.AbandonReleaseResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.abandon_release(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_abandon_release_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.abandon_release._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_abandon_release_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_abandon_release') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_abandon_release') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.AbandonReleaseRequest.pb(cloud_deploy.AbandonReleaseRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.AbandonReleaseResponse.to_json(cloud_deploy.AbandonReleaseResponse())
        request = cloud_deploy.AbandonReleaseRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.AbandonReleaseResponse()
        client.abandon_release(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_abandon_release_rest_bad_request(transport: str='rest', request_type=cloud_deploy.AbandonReleaseRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.abandon_release(request)

def test_abandon_release_rest_flattened():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.AbandonReleaseResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.AbandonReleaseResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.abandon_release(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/deliveryPipelines/*/releases/*}:abandon' % client.transport._host, args[1])

def test_abandon_release_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.abandon_release(cloud_deploy.AbandonReleaseRequest(), name='name_value')

def test_abandon_release_rest_error():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.ApproveRolloutRequest, dict])
def test_approve_rollout_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.ApproveRolloutResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.ApproveRolloutResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.approve_rollout(request)
    assert isinstance(response, cloud_deploy.ApproveRolloutResponse)

def test_approve_rollout_rest_required_fields(request_type=cloud_deploy.ApproveRolloutRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['approved'] = False
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).approve_rollout._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['approved'] = True
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).approve_rollout._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'approved' in jsonified_request
    assert jsonified_request['approved'] == True
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.ApproveRolloutResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.ApproveRolloutResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.approve_rollout(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_approve_rollout_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.approve_rollout._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'approved'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_approve_rollout_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_approve_rollout') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_approve_rollout') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.ApproveRolloutRequest.pb(cloud_deploy.ApproveRolloutRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.ApproveRolloutResponse.to_json(cloud_deploy.ApproveRolloutResponse())
        request = cloud_deploy.ApproveRolloutRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.ApproveRolloutResponse()
        client.approve_rollout(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_approve_rollout_rest_bad_request(transport: str='rest', request_type=cloud_deploy.ApproveRolloutRequest):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.approve_rollout(request)

def test_approve_rollout_rest_flattened():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.ApproveRolloutResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.ApproveRolloutResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.approve_rollout(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/deliveryPipelines/*/releases/*/rollouts/*}:approve' % client.transport._host, args[1])

def test_approve_rollout_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.approve_rollout(cloud_deploy.ApproveRolloutRequest(), name='name_value')

def test_approve_rollout_rest_error():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.AdvanceRolloutRequest, dict])
def test_advance_rollout_rest(request_type):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.AdvanceRolloutResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.AdvanceRolloutResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.advance_rollout(request)
    assert isinstance(response, cloud_deploy.AdvanceRolloutResponse)

def test_advance_rollout_rest_required_fields(request_type=cloud_deploy.AdvanceRolloutRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['phase_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).advance_rollout._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['phaseId'] = 'phase_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).advance_rollout._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'phaseId' in jsonified_request
    assert jsonified_request['phaseId'] == 'phase_id_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.AdvanceRolloutResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.AdvanceRolloutResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.advance_rollout(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_advance_rollout_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.advance_rollout._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'phaseId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_advance_rollout_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_advance_rollout') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_advance_rollout') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.AdvanceRolloutRequest.pb(cloud_deploy.AdvanceRolloutRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.AdvanceRolloutResponse.to_json(cloud_deploy.AdvanceRolloutResponse())
        request = cloud_deploy.AdvanceRolloutRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.AdvanceRolloutResponse()
        client.advance_rollout(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_advance_rollout_rest_bad_request(transport: str='rest', request_type=cloud_deploy.AdvanceRolloutRequest):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.advance_rollout(request)

def test_advance_rollout_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.AdvanceRolloutResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
        mock_args = dict(name='name_value', phase_id='phase_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.AdvanceRolloutResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.advance_rollout(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/deliveryPipelines/*/releases/*/rollouts/*}:advance' % client.transport._host, args[1])

def test_advance_rollout_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.advance_rollout(cloud_deploy.AdvanceRolloutRequest(), name='name_value', phase_id='phase_id_value')

def test_advance_rollout_rest_error():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.CancelRolloutRequest, dict])
def test_cancel_rollout_rest(request_type):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.CancelRolloutResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.CancelRolloutResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.cancel_rollout(request)
    assert isinstance(response, cloud_deploy.CancelRolloutResponse)

def test_cancel_rollout_rest_required_fields(request_type=cloud_deploy.CancelRolloutRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).cancel_rollout._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).cancel_rollout._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.CancelRolloutResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.CancelRolloutResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.cancel_rollout(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_cancel_rollout_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.cancel_rollout._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_cancel_rollout_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_cancel_rollout') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_cancel_rollout') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.CancelRolloutRequest.pb(cloud_deploy.CancelRolloutRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.CancelRolloutResponse.to_json(cloud_deploy.CancelRolloutResponse())
        request = cloud_deploy.CancelRolloutRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.CancelRolloutResponse()
        client.cancel_rollout(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_cancel_rollout_rest_bad_request(transport: str='rest', request_type=cloud_deploy.CancelRolloutRequest):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.cancel_rollout(request)

def test_cancel_rollout_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.CancelRolloutResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.CancelRolloutResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.cancel_rollout(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/deliveryPipelines/*/releases/*/rollouts/*}:cancel' % client.transport._host, args[1])

def test_cancel_rollout_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.cancel_rollout(cloud_deploy.CancelRolloutRequest(), name='name_value')

def test_cancel_rollout_rest_error():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.ListRolloutsRequest, dict])
def test_list_rollouts_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.ListRolloutsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.ListRolloutsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_rollouts(request)
    assert isinstance(response, pagers.ListRolloutsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_rollouts_rest_required_fields(request_type=cloud_deploy.ListRolloutsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_rollouts._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_rollouts._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.ListRolloutsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.ListRolloutsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_rollouts(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_rollouts_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_rollouts._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_rollouts_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_list_rollouts') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_list_rollouts') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.ListRolloutsRequest.pb(cloud_deploy.ListRolloutsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.ListRolloutsResponse.to_json(cloud_deploy.ListRolloutsResponse())
        request = cloud_deploy.ListRolloutsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.ListRolloutsResponse()
        client.list_rollouts(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_rollouts_rest_bad_request(transport: str='rest', request_type=cloud_deploy.ListRolloutsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_rollouts(request)

def test_list_rollouts_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.ListRolloutsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.ListRolloutsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_rollouts(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/deliveryPipelines/*/releases/*}/rollouts' % client.transport._host, args[1])

def test_list_rollouts_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_rollouts(cloud_deploy.ListRolloutsRequest(), parent='parent_value')

def test_list_rollouts_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (cloud_deploy.ListRolloutsResponse(rollouts=[cloud_deploy.Rollout(), cloud_deploy.Rollout(), cloud_deploy.Rollout()], next_page_token='abc'), cloud_deploy.ListRolloutsResponse(rollouts=[], next_page_token='def'), cloud_deploy.ListRolloutsResponse(rollouts=[cloud_deploy.Rollout()], next_page_token='ghi'), cloud_deploy.ListRolloutsResponse(rollouts=[cloud_deploy.Rollout(), cloud_deploy.Rollout()]))
        response = response + response
        response = tuple((cloud_deploy.ListRolloutsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4'}
        pager = client.list_rollouts(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_deploy.Rollout) for i in results))
        pages = list(client.list_rollouts(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_deploy.GetRolloutRequest, dict])
def test_get_rollout_rest(request_type):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.Rollout(name='name_value', uid='uid_value', description='description_value', target_id='target_id_value', approval_state=cloud_deploy.Rollout.ApprovalState.NEEDS_APPROVAL, state=cloud_deploy.Rollout.State.SUCCEEDED, failure_reason='failure_reason_value', deploying_build='deploying_build_value', etag='etag_value', deploy_failure_cause=cloud_deploy.Rollout.FailureCause.CLOUD_BUILD_UNAVAILABLE, controller_rollout='controller_rollout_value', rollback_of_rollout='rollback_of_rollout_value', rolled_back_by_rollouts=['rolled_back_by_rollouts_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.Rollout.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_rollout(request)
    assert isinstance(response, cloud_deploy.Rollout)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.target_id == 'target_id_value'
    assert response.approval_state == cloud_deploy.Rollout.ApprovalState.NEEDS_APPROVAL
    assert response.state == cloud_deploy.Rollout.State.SUCCEEDED
    assert response.failure_reason == 'failure_reason_value'
    assert response.deploying_build == 'deploying_build_value'
    assert response.etag == 'etag_value'
    assert response.deploy_failure_cause == cloud_deploy.Rollout.FailureCause.CLOUD_BUILD_UNAVAILABLE
    assert response.controller_rollout == 'controller_rollout_value'
    assert response.rollback_of_rollout == 'rollback_of_rollout_value'
    assert response.rolled_back_by_rollouts == ['rolled_back_by_rollouts_value']

def test_get_rollout_rest_required_fields(request_type=cloud_deploy.GetRolloutRequest):
    if False:
        return 10
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_rollout._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_rollout._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.Rollout()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.Rollout.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_rollout(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_rollout_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_rollout._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_rollout_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_get_rollout') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_get_rollout') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.GetRolloutRequest.pb(cloud_deploy.GetRolloutRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.Rollout.to_json(cloud_deploy.Rollout())
        request = cloud_deploy.GetRolloutRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.Rollout()
        client.get_rollout(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_rollout_rest_bad_request(transport: str='rest', request_type=cloud_deploy.GetRolloutRequest):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_rollout(request)

def test_get_rollout_rest_flattened():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.Rollout()
        sample_request = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.Rollout.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_rollout(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/deliveryPipelines/*/releases/*/rollouts/*}' % client.transport._host, args[1])

def test_get_rollout_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_rollout(cloud_deploy.GetRolloutRequest(), name='name_value')

def test_get_rollout_rest_error():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.CreateRolloutRequest, dict])
def test_create_rollout_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4'}
    request_init['rollout'] = {'name': 'name_value', 'uid': 'uid_value', 'description': 'description_value', 'annotations': {}, 'labels': {}, 'create_time': {'seconds': 751, 'nanos': 543}, 'approve_time': {}, 'enqueue_time': {}, 'deploy_start_time': {}, 'deploy_end_time': {}, 'target_id': 'target_id_value', 'approval_state': 1, 'state': 1, 'failure_reason': 'failure_reason_value', 'deploying_build': 'deploying_build_value', 'etag': 'etag_value', 'deploy_failure_cause': 1, 'phases': [{'id': 'id_value', 'state': 1, 'skip_message': 'skip_message_value', 'deployment_jobs': {'deploy_job': {'id': 'id_value', 'state': 1, 'skip_message': 'skip_message_value', 'job_run': 'job_run_value', 'deploy_job': {}, 'verify_job': {}, 'predeploy_job': {'actions': ['actions_value1', 'actions_value2']}, 'postdeploy_job': {'actions': ['actions_value1', 'actions_value2']}, 'create_child_rollout_job': {}, 'advance_child_rollout_job': {}}, 'verify_job': {}, 'predeploy_job': {}, 'postdeploy_job': {}}, 'child_rollout_jobs': {'create_rollout_jobs': {}, 'advance_rollout_jobs': {}}}], 'metadata': {'cloud_run': {'service': 'service_value', 'service_urls': ['service_urls_value1', 'service_urls_value2'], 'revision': 'revision_value', 'job': 'job_value'}, 'automation': {'promote_automation_run': 'promote_automation_run_value', 'advance_automation_runs': ['advance_automation_runs_value1', 'advance_automation_runs_value2'], 'repair_automation_runs': ['repair_automation_runs_value1', 'repair_automation_runs_value2']}}, 'controller_rollout': 'controller_rollout_value', 'rollback_of_rollout': 'rollback_of_rollout_value', 'rolled_back_by_rollouts': ['rolled_back_by_rollouts_value1', 'rolled_back_by_rollouts_value2']}
    test_field = cloud_deploy.CreateRolloutRequest.meta.fields['rollout']

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
    for (field, value) in request_init['rollout'].items():
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
                for i in range(0, len(request_init['rollout'][field])):
                    del request_init['rollout'][field][i][subfield]
            else:
                del request_init['rollout'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_rollout(request)
    assert response.operation.name == 'operations/spam'

def test_create_rollout_rest_required_fields(request_type=cloud_deploy.CreateRolloutRequest):
    if False:
        return 10
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['rollout_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'rolloutId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_rollout._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'rolloutId' in jsonified_request
    assert jsonified_request['rolloutId'] == request_init['rollout_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['rolloutId'] = 'rollout_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_rollout._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'rollout_id', 'starting_phase_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'rolloutId' in jsonified_request
    assert jsonified_request['rolloutId'] == 'rollout_id_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_rollout(request)
            expected_params = [('rolloutId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_rollout_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_rollout._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'rolloutId', 'startingPhaseId', 'validateOnly')) & set(('parent', 'rolloutId', 'rollout'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_rollout_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudDeployRestInterceptor, 'post_create_rollout') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_create_rollout') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.CreateRolloutRequest.pb(cloud_deploy.CreateRolloutRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_deploy.CreateRolloutRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_rollout(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_rollout_rest_bad_request(transport: str='rest', request_type=cloud_deploy.CreateRolloutRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_rollout(request)

def test_create_rollout_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4'}
        mock_args = dict(parent='parent_value', rollout=cloud_deploy.Rollout(name='name_value'), rollout_id='rollout_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_rollout(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/deliveryPipelines/*/releases/*}/rollouts' % client.transport._host, args[1])

def test_create_rollout_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_rollout(cloud_deploy.CreateRolloutRequest(), parent='parent_value', rollout=cloud_deploy.Rollout(name='name_value'), rollout_id='rollout_id_value')

def test_create_rollout_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.IgnoreJobRequest, dict])
def test_ignore_job_rest(request_type):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'rollout': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.IgnoreJobResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.IgnoreJobResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.ignore_job(request)
    assert isinstance(response, cloud_deploy.IgnoreJobResponse)

def test_ignore_job_rest_required_fields(request_type=cloud_deploy.IgnoreJobRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['rollout'] = ''
    request_init['phase_id'] = ''
    request_init['job_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).ignore_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['rollout'] = 'rollout_value'
    jsonified_request['phaseId'] = 'phase_id_value'
    jsonified_request['jobId'] = 'job_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).ignore_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'rollout' in jsonified_request
    assert jsonified_request['rollout'] == 'rollout_value'
    assert 'phaseId' in jsonified_request
    assert jsonified_request['phaseId'] == 'phase_id_value'
    assert 'jobId' in jsonified_request
    assert jsonified_request['jobId'] == 'job_id_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.IgnoreJobResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.IgnoreJobResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.ignore_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_ignore_job_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.ignore_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('rollout', 'phaseId', 'jobId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_ignore_job_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_ignore_job') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_ignore_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.IgnoreJobRequest.pb(cloud_deploy.IgnoreJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.IgnoreJobResponse.to_json(cloud_deploy.IgnoreJobResponse())
        request = cloud_deploy.IgnoreJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.IgnoreJobResponse()
        client.ignore_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_ignore_job_rest_bad_request(transport: str='rest', request_type=cloud_deploy.IgnoreJobRequest):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'rollout': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.ignore_job(request)

def test_ignore_job_rest_flattened():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.IgnoreJobResponse()
        sample_request = {'rollout': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
        mock_args = dict(rollout='rollout_value', phase_id='phase_id_value', job_id='job_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.IgnoreJobResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.ignore_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{rollout=projects/*/locations/*/deliveryPipelines/*/releases/*/rollouts/*}:ignoreJob' % client.transport._host, args[1])

def test_ignore_job_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.ignore_job(cloud_deploy.IgnoreJobRequest(), rollout='rollout_value', phase_id='phase_id_value', job_id='job_id_value')

def test_ignore_job_rest_error():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.RetryJobRequest, dict])
def test_retry_job_rest(request_type):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'rollout': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.RetryJobResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.RetryJobResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.retry_job(request)
    assert isinstance(response, cloud_deploy.RetryJobResponse)

def test_retry_job_rest_required_fields(request_type=cloud_deploy.RetryJobRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['rollout'] = ''
    request_init['phase_id'] = ''
    request_init['job_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).retry_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['rollout'] = 'rollout_value'
    jsonified_request['phaseId'] = 'phase_id_value'
    jsonified_request['jobId'] = 'job_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).retry_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'rollout' in jsonified_request
    assert jsonified_request['rollout'] == 'rollout_value'
    assert 'phaseId' in jsonified_request
    assert jsonified_request['phaseId'] == 'phase_id_value'
    assert 'jobId' in jsonified_request
    assert jsonified_request['jobId'] == 'job_id_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.RetryJobResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.RetryJobResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.retry_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_retry_job_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.retry_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('rollout', 'phaseId', 'jobId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_retry_job_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_retry_job') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_retry_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.RetryJobRequest.pb(cloud_deploy.RetryJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.RetryJobResponse.to_json(cloud_deploy.RetryJobResponse())
        request = cloud_deploy.RetryJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.RetryJobResponse()
        client.retry_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_retry_job_rest_bad_request(transport: str='rest', request_type=cloud_deploy.RetryJobRequest):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'rollout': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.retry_job(request)

def test_retry_job_rest_flattened():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.RetryJobResponse()
        sample_request = {'rollout': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
        mock_args = dict(rollout='rollout_value', phase_id='phase_id_value', job_id='job_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.RetryJobResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.retry_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{rollout=projects/*/locations/*/deliveryPipelines/*/releases/*/rollouts/*}:retryJob' % client.transport._host, args[1])

def test_retry_job_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.retry_job(cloud_deploy.RetryJobRequest(), rollout='rollout_value', phase_id='phase_id_value', job_id='job_id_value')

def test_retry_job_rest_error():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.ListJobRunsRequest, dict])
def test_list_job_runs_rest(request_type):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.ListJobRunsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.ListJobRunsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_job_runs(request)
    assert isinstance(response, pagers.ListJobRunsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_job_runs_rest_required_fields(request_type=cloud_deploy.ListJobRunsRequest):
    if False:
        return 10
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_job_runs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_job_runs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.ListJobRunsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.ListJobRunsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_job_runs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_job_runs_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_job_runs._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_job_runs_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_list_job_runs') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_list_job_runs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.ListJobRunsRequest.pb(cloud_deploy.ListJobRunsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.ListJobRunsResponse.to_json(cloud_deploy.ListJobRunsResponse())
        request = cloud_deploy.ListJobRunsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.ListJobRunsResponse()
        client.list_job_runs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_job_runs_rest_bad_request(transport: str='rest', request_type=cloud_deploy.ListJobRunsRequest):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_job_runs(request)

def test_list_job_runs_rest_flattened():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.ListJobRunsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.ListJobRunsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_job_runs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/deliveryPipelines/*/releases/*/rollouts/*}/jobRuns' % client.transport._host, args[1])

def test_list_job_runs_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_job_runs(cloud_deploy.ListJobRunsRequest(), parent='parent_value')

def test_list_job_runs_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (cloud_deploy.ListJobRunsResponse(job_runs=[cloud_deploy.JobRun(), cloud_deploy.JobRun(), cloud_deploy.JobRun()], next_page_token='abc'), cloud_deploy.ListJobRunsResponse(job_runs=[], next_page_token='def'), cloud_deploy.ListJobRunsResponse(job_runs=[cloud_deploy.JobRun()], next_page_token='ghi'), cloud_deploy.ListJobRunsResponse(job_runs=[cloud_deploy.JobRun(), cloud_deploy.JobRun()]))
        response = response + response
        response = tuple((cloud_deploy.ListJobRunsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5'}
        pager = client.list_job_runs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_deploy.JobRun) for i in results))
        pages = list(client.list_job_runs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_deploy.GetJobRunRequest, dict])
def test_get_job_run_rest(request_type):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5/jobRuns/sample6'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.JobRun(name='name_value', uid='uid_value', phase_id='phase_id_value', job_id='job_id_value', state=cloud_deploy.JobRun.State.IN_PROGRESS, etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.JobRun.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_job_run(request)
    assert isinstance(response, cloud_deploy.JobRun)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.phase_id == 'phase_id_value'
    assert response.job_id == 'job_id_value'
    assert response.state == cloud_deploy.JobRun.State.IN_PROGRESS
    assert response.etag == 'etag_value'

def test_get_job_run_rest_required_fields(request_type=cloud_deploy.GetJobRunRequest):
    if False:
        return 10
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_job_run._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_job_run._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.JobRun()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.JobRun.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_job_run(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_job_run_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_job_run._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_job_run_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_get_job_run') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_get_job_run') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.GetJobRunRequest.pb(cloud_deploy.GetJobRunRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.JobRun.to_json(cloud_deploy.JobRun())
        request = cloud_deploy.GetJobRunRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.JobRun()
        client.get_job_run(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_job_run_rest_bad_request(transport: str='rest', request_type=cloud_deploy.GetJobRunRequest):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5/jobRuns/sample6'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_job_run(request)

def test_get_job_run_rest_flattened():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.JobRun()
        sample_request = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5/jobRuns/sample6'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.JobRun.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_job_run(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/deliveryPipelines/*/releases/*/rollouts/*/jobRuns/*}' % client.transport._host, args[1])

def test_get_job_run_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_job_run(cloud_deploy.GetJobRunRequest(), name='name_value')

def test_get_job_run_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.TerminateJobRunRequest, dict])
def test_terminate_job_run_rest(request_type):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5/jobRuns/sample6'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.TerminateJobRunResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.TerminateJobRunResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.terminate_job_run(request)
    assert isinstance(response, cloud_deploy.TerminateJobRunResponse)

def test_terminate_job_run_rest_required_fields(request_type=cloud_deploy.TerminateJobRunRequest):
    if False:
        return 10
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).terminate_job_run._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).terminate_job_run._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.TerminateJobRunResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.TerminateJobRunResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.terminate_job_run(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_terminate_job_run_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.terminate_job_run._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_terminate_job_run_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_terminate_job_run') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_terminate_job_run') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.TerminateJobRunRequest.pb(cloud_deploy.TerminateJobRunRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.TerminateJobRunResponse.to_json(cloud_deploy.TerminateJobRunResponse())
        request = cloud_deploy.TerminateJobRunRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.TerminateJobRunResponse()
        client.terminate_job_run(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_terminate_job_run_rest_bad_request(transport: str='rest', request_type=cloud_deploy.TerminateJobRunRequest):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5/jobRuns/sample6'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.terminate_job_run(request)

def test_terminate_job_run_rest_flattened():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.TerminateJobRunResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/releases/sample4/rollouts/sample5/jobRuns/sample6'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.TerminateJobRunResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.terminate_job_run(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/deliveryPipelines/*/releases/*/rollouts/*/jobRuns/*}:terminate' % client.transport._host, args[1])

def test_terminate_job_run_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.terminate_job_run(cloud_deploy.TerminateJobRunRequest(), name='name_value')

def test_terminate_job_run_rest_error():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.GetConfigRequest, dict])
def test_get_config_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/config'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.Config(name='name_value', default_skaffold_version='default_skaffold_version_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.Config.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_config(request)
    assert isinstance(response, cloud_deploy.Config)
    assert response.name == 'name_value'
    assert response.default_skaffold_version == 'default_skaffold_version_value'

def test_get_config_rest_required_fields(request_type=cloud_deploy.GetConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.Config()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.Config.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_config_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_config_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_get_config') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_get_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.GetConfigRequest.pb(cloud_deploy.GetConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.Config.to_json(cloud_deploy.Config())
        request = cloud_deploy.GetConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.Config()
        client.get_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_config_rest_bad_request(transport: str='rest', request_type=cloud_deploy.GetConfigRequest):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/config'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_config(request)

def test_get_config_rest_flattened():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.Config()
        sample_request = {'name': 'projects/sample1/locations/sample2/config'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.Config.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/config}' % client.transport._host, args[1])

def test_get_config_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_config(cloud_deploy.GetConfigRequest(), name='name_value')

def test_get_config_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.CreateAutomationRequest, dict])
def test_create_automation_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
    request_init['automation'] = {'name': 'name_value', 'uid': 'uid_value', 'description': 'description_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'annotations': {}, 'labels': {}, 'etag': 'etag_value', 'suspended': True, 'service_account': 'service_account_value', 'selector': {'targets': [{'id': 'id_value', 'labels': {}}]}, 'rules': [{'promote_release_rule': {'id': 'id_value', 'wait': {'seconds': 751, 'nanos': 543}, 'destination_target_id': 'destination_target_id_value', 'condition': {'targets_present_condition': {'status': True, 'missing_targets': ['missing_targets_value1', 'missing_targets_value2'], 'update_time': {}}}, 'destination_phase': 'destination_phase_value'}, 'advance_rollout_rule': {'id': 'id_value', 'source_phases': ['source_phases_value1', 'source_phases_value2'], 'wait': {}, 'condition': {}}, 'repair_rollout_rule': {'id': 'id_value', 'source_phases': ['source_phases_value1', 'source_phases_value2'], 'jobs': ['jobs_value1', 'jobs_value2'], 'repair_modes': [{'retry': {'attempts': 882, 'wait': {}, 'backoff_mode': 1}, 'rollback': {'destination_phase': 'destination_phase_value'}}], 'condition': {}}}]}
    test_field = cloud_deploy.CreateAutomationRequest.meta.fields['automation']

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
    for (field, value) in request_init['automation'].items():
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
                for i in range(0, len(request_init['automation'][field])):
                    del request_init['automation'][field][i][subfield]
            else:
                del request_init['automation'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_automation(request)
    assert response.operation.name == 'operations/spam'

def test_create_automation_rest_required_fields(request_type=cloud_deploy.CreateAutomationRequest):
    if False:
        return 10
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['automation_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'automationId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_automation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'automationId' in jsonified_request
    assert jsonified_request['automationId'] == request_init['automation_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['automationId'] = 'automation_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_automation._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('automation_id', 'request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'automationId' in jsonified_request
    assert jsonified_request['automationId'] == 'automation_id_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_automation(request)
            expected_params = [('automationId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_automation_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_automation._get_unset_required_fields({})
    assert set(unset_fields) == set(('automationId', 'requestId', 'validateOnly')) & set(('parent', 'automationId', 'automation'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_automation_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudDeployRestInterceptor, 'post_create_automation') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_create_automation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.CreateAutomationRequest.pb(cloud_deploy.CreateAutomationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_deploy.CreateAutomationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_automation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_automation_rest_bad_request(transport: str='rest', request_type=cloud_deploy.CreateAutomationRequest):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_automation(request)

def test_create_automation_rest_flattened():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
        mock_args = dict(parent='parent_value', automation=cloud_deploy.Automation(name='name_value'), automation_id='automation_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_automation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/deliveryPipelines/*}/automations' % client.transport._host, args[1])

def test_create_automation_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_automation(cloud_deploy.CreateAutomationRequest(), parent='parent_value', automation=cloud_deploy.Automation(name='name_value'), automation_id='automation_id_value')

def test_create_automation_rest_error():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.UpdateAutomationRequest, dict])
def test_update_automation_rest(request_type):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'automation': {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/automations/sample4'}}
    request_init['automation'] = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/automations/sample4', 'uid': 'uid_value', 'description': 'description_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'annotations': {}, 'labels': {}, 'etag': 'etag_value', 'suspended': True, 'service_account': 'service_account_value', 'selector': {'targets': [{'id': 'id_value', 'labels': {}}]}, 'rules': [{'promote_release_rule': {'id': 'id_value', 'wait': {'seconds': 751, 'nanos': 543}, 'destination_target_id': 'destination_target_id_value', 'condition': {'targets_present_condition': {'status': True, 'missing_targets': ['missing_targets_value1', 'missing_targets_value2'], 'update_time': {}}}, 'destination_phase': 'destination_phase_value'}, 'advance_rollout_rule': {'id': 'id_value', 'source_phases': ['source_phases_value1', 'source_phases_value2'], 'wait': {}, 'condition': {}}, 'repair_rollout_rule': {'id': 'id_value', 'source_phases': ['source_phases_value1', 'source_phases_value2'], 'jobs': ['jobs_value1', 'jobs_value2'], 'repair_modes': [{'retry': {'attempts': 882, 'wait': {}, 'backoff_mode': 1}, 'rollback': {'destination_phase': 'destination_phase_value'}}], 'condition': {}}}]}
    test_field = cloud_deploy.UpdateAutomationRequest.meta.fields['automation']

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
    for (field, value) in request_init['automation'].items():
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
                for i in range(0, len(request_init['automation'][field])):
                    del request_init['automation'][field][i][subfield]
            else:
                del request_init['automation'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_automation(request)
    assert response.operation.name == 'operations/spam'

def test_update_automation_rest_required_fields(request_type=cloud_deploy.UpdateAutomationRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_automation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_automation._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'request_id', 'update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_automation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_automation_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_automation._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'requestId', 'updateMask', 'validateOnly')) & set(('updateMask', 'automation'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_automation_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudDeployRestInterceptor, 'post_update_automation') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_update_automation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.UpdateAutomationRequest.pb(cloud_deploy.UpdateAutomationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_deploy.UpdateAutomationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_automation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_automation_rest_bad_request(transport: str='rest', request_type=cloud_deploy.UpdateAutomationRequest):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'automation': {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/automations/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_automation(request)

def test_update_automation_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'automation': {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/automations/sample4'}}
        mock_args = dict(automation=cloud_deploy.Automation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_automation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{automation.name=projects/*/locations/*/deliveryPipelines/*/automations/*}' % client.transport._host, args[1])

def test_update_automation_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_automation(cloud_deploy.UpdateAutomationRequest(), automation=cloud_deploy.Automation(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_automation_rest_error():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.DeleteAutomationRequest, dict])
def test_delete_automation_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/automations/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_automation(request)
    assert response.operation.name == 'operations/spam'

def test_delete_automation_rest_required_fields(request_type=cloud_deploy.DeleteAutomationRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_automation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_automation._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'etag', 'request_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_automation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_automation_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_automation._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'etag', 'requestId', 'validateOnly')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_automation_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudDeployRestInterceptor, 'post_delete_automation') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_delete_automation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.DeleteAutomationRequest.pb(cloud_deploy.DeleteAutomationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = cloud_deploy.DeleteAutomationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_automation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_automation_rest_bad_request(transport: str='rest', request_type=cloud_deploy.DeleteAutomationRequest):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/automations/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_automation(request)

def test_delete_automation_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/automations/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_automation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/deliveryPipelines/*/automations/*}' % client.transport._host, args[1])

def test_delete_automation_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_automation(cloud_deploy.DeleteAutomationRequest(), name='name_value')

def test_delete_automation_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.GetAutomationRequest, dict])
def test_get_automation_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/automations/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.Automation(name='name_value', uid='uid_value', description='description_value', etag='etag_value', suspended=True, service_account='service_account_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.Automation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_automation(request)
    assert isinstance(response, cloud_deploy.Automation)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.description == 'description_value'
    assert response.etag == 'etag_value'
    assert response.suspended is True
    assert response.service_account == 'service_account_value'

def test_get_automation_rest_required_fields(request_type=cloud_deploy.GetAutomationRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_automation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_automation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.Automation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.Automation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_automation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_automation_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_automation._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_automation_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_get_automation') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_get_automation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.GetAutomationRequest.pb(cloud_deploy.GetAutomationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.Automation.to_json(cloud_deploy.Automation())
        request = cloud_deploy.GetAutomationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.Automation()
        client.get_automation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_automation_rest_bad_request(transport: str='rest', request_type=cloud_deploy.GetAutomationRequest):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/automations/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_automation(request)

def test_get_automation_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.Automation()
        sample_request = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/automations/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.Automation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_automation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/deliveryPipelines/*/automations/*}' % client.transport._host, args[1])

def test_get_automation_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_automation(cloud_deploy.GetAutomationRequest(), name='name_value')

def test_get_automation_rest_error():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.ListAutomationsRequest, dict])
def test_list_automations_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.ListAutomationsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.ListAutomationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_automations(request)
    assert isinstance(response, pagers.ListAutomationsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_automations_rest_required_fields(request_type=cloud_deploy.ListAutomationsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_automations._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_automations._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.ListAutomationsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.ListAutomationsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_automations(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_automations_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_automations._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_automations_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_list_automations') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_list_automations') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.ListAutomationsRequest.pb(cloud_deploy.ListAutomationsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.ListAutomationsResponse.to_json(cloud_deploy.ListAutomationsResponse())
        request = cloud_deploy.ListAutomationsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.ListAutomationsResponse()
        client.list_automations(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_automations_rest_bad_request(transport: str='rest', request_type=cloud_deploy.ListAutomationsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_automations(request)

def test_list_automations_rest_flattened():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.ListAutomationsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.ListAutomationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_automations(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/deliveryPipelines/*}/automations' % client.transport._host, args[1])

def test_list_automations_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_automations(cloud_deploy.ListAutomationsRequest(), parent='parent_value')

def test_list_automations_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (cloud_deploy.ListAutomationsResponse(automations=[cloud_deploy.Automation(), cloud_deploy.Automation(), cloud_deploy.Automation()], next_page_token='abc'), cloud_deploy.ListAutomationsResponse(automations=[], next_page_token='def'), cloud_deploy.ListAutomationsResponse(automations=[cloud_deploy.Automation()], next_page_token='ghi'), cloud_deploy.ListAutomationsResponse(automations=[cloud_deploy.Automation(), cloud_deploy.Automation()]))
        response = response + response
        response = tuple((cloud_deploy.ListAutomationsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
        pager = client.list_automations(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_deploy.Automation) for i in results))
        pages = list(client.list_automations(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_deploy.GetAutomationRunRequest, dict])
def test_get_automation_run_rest(request_type):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/automationRuns/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.AutomationRun(name='name_value', etag='etag_value', service_account='service_account_value', target_id='target_id_value', state=cloud_deploy.AutomationRun.State.SUCCEEDED, state_description='state_description_value', rule_id='rule_id_value', automation_id='automation_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.AutomationRun.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_automation_run(request)
    assert isinstance(response, cloud_deploy.AutomationRun)
    assert response.name == 'name_value'
    assert response.etag == 'etag_value'
    assert response.service_account == 'service_account_value'
    assert response.target_id == 'target_id_value'
    assert response.state == cloud_deploy.AutomationRun.State.SUCCEEDED
    assert response.state_description == 'state_description_value'
    assert response.rule_id == 'rule_id_value'
    assert response.automation_id == 'automation_id_value'

def test_get_automation_run_rest_required_fields(request_type=cloud_deploy.GetAutomationRunRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_automation_run._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_automation_run._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.AutomationRun()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.AutomationRun.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_automation_run(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_automation_run_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_automation_run._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_automation_run_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_get_automation_run') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_get_automation_run') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.GetAutomationRunRequest.pb(cloud_deploy.GetAutomationRunRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.AutomationRun.to_json(cloud_deploy.AutomationRun())
        request = cloud_deploy.GetAutomationRunRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.AutomationRun()
        client.get_automation_run(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_automation_run_rest_bad_request(transport: str='rest', request_type=cloud_deploy.GetAutomationRunRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/automationRuns/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_automation_run(request)

def test_get_automation_run_rest_flattened():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.AutomationRun()
        sample_request = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/automationRuns/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.AutomationRun.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_automation_run(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/deliveryPipelines/*/automationRuns/*}' % client.transport._host, args[1])

def test_get_automation_run_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_automation_run(cloud_deploy.GetAutomationRunRequest(), name='name_value')

def test_get_automation_run_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_deploy.ListAutomationRunsRequest, dict])
def test_list_automation_runs_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.ListAutomationRunsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.ListAutomationRunsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_automation_runs(request)
    assert isinstance(response, pagers.ListAutomationRunsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_automation_runs_rest_required_fields(request_type=cloud_deploy.ListAutomationRunsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_automation_runs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_automation_runs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.ListAutomationRunsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.ListAutomationRunsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_automation_runs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_automation_runs_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_automation_runs._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_automation_runs_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_list_automation_runs') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_list_automation_runs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.ListAutomationRunsRequest.pb(cloud_deploy.ListAutomationRunsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.ListAutomationRunsResponse.to_json(cloud_deploy.ListAutomationRunsResponse())
        request = cloud_deploy.ListAutomationRunsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.ListAutomationRunsResponse()
        client.list_automation_runs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_automation_runs_rest_bad_request(transport: str='rest', request_type=cloud_deploy.ListAutomationRunsRequest):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_automation_runs(request)

def test_list_automation_runs_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.ListAutomationRunsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.ListAutomationRunsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_automation_runs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/deliveryPipelines/*}/automationRuns' % client.transport._host, args[1])

def test_list_automation_runs_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_automation_runs(cloud_deploy.ListAutomationRunsRequest(), parent='parent_value')

def test_list_automation_runs_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (cloud_deploy.ListAutomationRunsResponse(automation_runs=[cloud_deploy.AutomationRun(), cloud_deploy.AutomationRun(), cloud_deploy.AutomationRun()], next_page_token='abc'), cloud_deploy.ListAutomationRunsResponse(automation_runs=[], next_page_token='def'), cloud_deploy.ListAutomationRunsResponse(automation_runs=[cloud_deploy.AutomationRun()], next_page_token='ghi'), cloud_deploy.ListAutomationRunsResponse(automation_runs=[cloud_deploy.AutomationRun(), cloud_deploy.AutomationRun()]))
        response = response + response
        response = tuple((cloud_deploy.ListAutomationRunsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
        pager = client.list_automation_runs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_deploy.AutomationRun) for i in results))
        pages = list(client.list_automation_runs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_deploy.CancelAutomationRunRequest, dict])
def test_cancel_automation_run_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/automationRuns/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.CancelAutomationRunResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.CancelAutomationRunResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.cancel_automation_run(request)
    assert isinstance(response, cloud_deploy.CancelAutomationRunResponse)

def test_cancel_automation_run_rest_required_fields(request_type=cloud_deploy.CancelAutomationRunRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudDeployRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).cancel_automation_run._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).cancel_automation_run._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_deploy.CancelAutomationRunResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_deploy.CancelAutomationRunResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.cancel_automation_run(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_cancel_automation_run_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.cancel_automation_run._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_cancel_automation_run_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudDeployRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudDeployRestInterceptor())
    client = CloudDeployClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudDeployRestInterceptor, 'post_cancel_automation_run') as post, mock.patch.object(transports.CloudDeployRestInterceptor, 'pre_cancel_automation_run') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_deploy.CancelAutomationRunRequest.pb(cloud_deploy.CancelAutomationRunRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_deploy.CancelAutomationRunResponse.to_json(cloud_deploy.CancelAutomationRunResponse())
        request = cloud_deploy.CancelAutomationRunRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_deploy.CancelAutomationRunResponse()
        client.cancel_automation_run(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_cancel_automation_run_rest_bad_request(transport: str='rest', request_type=cloud_deploy.CancelAutomationRunRequest):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/automationRuns/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.cancel_automation_run(request)

def test_cancel_automation_run_rest_flattened():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_deploy.CancelAutomationRunResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/deliveryPipelines/sample3/automationRuns/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_deploy.CancelAutomationRunResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.cancel_automation_run(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/deliveryPipelines/*/automationRuns/*}:cancel' % client.transport._host, args[1])

def test_cancel_automation_run_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.cancel_automation_run(cloud_deploy.CancelAutomationRunRequest(), name='name_value')

def test_cancel_automation_run_rest_error():
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudDeployGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.CloudDeployGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudDeployClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.CloudDeployGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CloudDeployClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CloudDeployClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.CloudDeployGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudDeployClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.CloudDeployGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = CloudDeployClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.CloudDeployGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.CloudDeployGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.CloudDeployGrpcTransport, transports.CloudDeployGrpcAsyncIOTransport, transports.CloudDeployRestTransport])
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
        for i in range(10):
            print('nop')
    transport = CloudDeployClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.CloudDeployGrpcTransport)

def test_cloud_deploy_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.CloudDeployTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_cloud_deploy_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.deploy_v1.services.cloud_deploy.transports.CloudDeployTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.CloudDeployTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_delivery_pipelines', 'get_delivery_pipeline', 'create_delivery_pipeline', 'update_delivery_pipeline', 'delete_delivery_pipeline', 'list_targets', 'rollback_target', 'get_target', 'create_target', 'update_target', 'delete_target', 'list_releases', 'get_release', 'create_release', 'abandon_release', 'approve_rollout', 'advance_rollout', 'cancel_rollout', 'list_rollouts', 'get_rollout', 'create_rollout', 'ignore_job', 'retry_job', 'list_job_runs', 'get_job_run', 'terminate_job_run', 'get_config', 'create_automation', 'update_automation', 'delete_automation', 'get_automation', 'list_automations', 'get_automation_run', 'list_automation_runs', 'cancel_automation_run', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_cloud_deploy_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.deploy_v1.services.cloud_deploy.transports.CloudDeployTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CloudDeployTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_cloud_deploy_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.deploy_v1.services.cloud_deploy.transports.CloudDeployTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CloudDeployTransport()
        adc.assert_called_once()

def test_cloud_deploy_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        CloudDeployClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.CloudDeployGrpcTransport, transports.CloudDeployGrpcAsyncIOTransport])
def test_cloud_deploy_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.CloudDeployGrpcTransport, transports.CloudDeployGrpcAsyncIOTransport, transports.CloudDeployRestTransport])
def test_cloud_deploy_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.CloudDeployGrpcTransport, grpc_helpers), (transports.CloudDeployGrpcAsyncIOTransport, grpc_helpers_async)])
def test_cloud_deploy_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('clouddeploy.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='clouddeploy.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.CloudDeployGrpcTransport, transports.CloudDeployGrpcAsyncIOTransport])
def test_cloud_deploy_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_cloud_deploy_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.CloudDeployRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_cloud_deploy_rest_lro_client():
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_cloud_deploy_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='clouddeploy.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('clouddeploy.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://clouddeploy.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_cloud_deploy_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='clouddeploy.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('clouddeploy.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://clouddeploy.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_cloud_deploy_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = CloudDeployClient(credentials=creds1, transport=transport_name)
    client2 = CloudDeployClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_delivery_pipelines._session
    session2 = client2.transport.list_delivery_pipelines._session
    assert session1 != session2
    session1 = client1.transport.get_delivery_pipeline._session
    session2 = client2.transport.get_delivery_pipeline._session
    assert session1 != session2
    session1 = client1.transport.create_delivery_pipeline._session
    session2 = client2.transport.create_delivery_pipeline._session
    assert session1 != session2
    session1 = client1.transport.update_delivery_pipeline._session
    session2 = client2.transport.update_delivery_pipeline._session
    assert session1 != session2
    session1 = client1.transport.delete_delivery_pipeline._session
    session2 = client2.transport.delete_delivery_pipeline._session
    assert session1 != session2
    session1 = client1.transport.list_targets._session
    session2 = client2.transport.list_targets._session
    assert session1 != session2
    session1 = client1.transport.rollback_target._session
    session2 = client2.transport.rollback_target._session
    assert session1 != session2
    session1 = client1.transport.get_target._session
    session2 = client2.transport.get_target._session
    assert session1 != session2
    session1 = client1.transport.create_target._session
    session2 = client2.transport.create_target._session
    assert session1 != session2
    session1 = client1.transport.update_target._session
    session2 = client2.transport.update_target._session
    assert session1 != session2
    session1 = client1.transport.delete_target._session
    session2 = client2.transport.delete_target._session
    assert session1 != session2
    session1 = client1.transport.list_releases._session
    session2 = client2.transport.list_releases._session
    assert session1 != session2
    session1 = client1.transport.get_release._session
    session2 = client2.transport.get_release._session
    assert session1 != session2
    session1 = client1.transport.create_release._session
    session2 = client2.transport.create_release._session
    assert session1 != session2
    session1 = client1.transport.abandon_release._session
    session2 = client2.transport.abandon_release._session
    assert session1 != session2
    session1 = client1.transport.approve_rollout._session
    session2 = client2.transport.approve_rollout._session
    assert session1 != session2
    session1 = client1.transport.advance_rollout._session
    session2 = client2.transport.advance_rollout._session
    assert session1 != session2
    session1 = client1.transport.cancel_rollout._session
    session2 = client2.transport.cancel_rollout._session
    assert session1 != session2
    session1 = client1.transport.list_rollouts._session
    session2 = client2.transport.list_rollouts._session
    assert session1 != session2
    session1 = client1.transport.get_rollout._session
    session2 = client2.transport.get_rollout._session
    assert session1 != session2
    session1 = client1.transport.create_rollout._session
    session2 = client2.transport.create_rollout._session
    assert session1 != session2
    session1 = client1.transport.ignore_job._session
    session2 = client2.transport.ignore_job._session
    assert session1 != session2
    session1 = client1.transport.retry_job._session
    session2 = client2.transport.retry_job._session
    assert session1 != session2
    session1 = client1.transport.list_job_runs._session
    session2 = client2.transport.list_job_runs._session
    assert session1 != session2
    session1 = client1.transport.get_job_run._session
    session2 = client2.transport.get_job_run._session
    assert session1 != session2
    session1 = client1.transport.terminate_job_run._session
    session2 = client2.transport.terminate_job_run._session
    assert session1 != session2
    session1 = client1.transport.get_config._session
    session2 = client2.transport.get_config._session
    assert session1 != session2
    session1 = client1.transport.create_automation._session
    session2 = client2.transport.create_automation._session
    assert session1 != session2
    session1 = client1.transport.update_automation._session
    session2 = client2.transport.update_automation._session
    assert session1 != session2
    session1 = client1.transport.delete_automation._session
    session2 = client2.transport.delete_automation._session
    assert session1 != session2
    session1 = client1.transport.get_automation._session
    session2 = client2.transport.get_automation._session
    assert session1 != session2
    session1 = client1.transport.list_automations._session
    session2 = client2.transport.list_automations._session
    assert session1 != session2
    session1 = client1.transport.get_automation_run._session
    session2 = client2.transport.get_automation_run._session
    assert session1 != session2
    session1 = client1.transport.list_automation_runs._session
    session2 = client2.transport.list_automation_runs._session
    assert session1 != session2
    session1 = client1.transport.cancel_automation_run._session
    session2 = client2.transport.cancel_automation_run._session
    assert session1 != session2

def test_cloud_deploy_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CloudDeployGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_cloud_deploy_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CloudDeployGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.CloudDeployGrpcTransport, transports.CloudDeployGrpcAsyncIOTransport])
def test_cloud_deploy_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.CloudDeployGrpcTransport, transports.CloudDeployGrpcAsyncIOTransport])
def test_cloud_deploy_transport_channel_mtls_with_adc(transport_class):
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

def test_cloud_deploy_grpc_lro_client():
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_cloud_deploy_grpc_lro_async_client():
    if False:
        while True:
            i = 10
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_automation_path():
    if False:
        return 10
    project = 'squid'
    location = 'clam'
    delivery_pipeline = 'whelk'
    automation = 'octopus'
    expected = 'projects/{project}/locations/{location}/deliveryPipelines/{delivery_pipeline}/automations/{automation}'.format(project=project, location=location, delivery_pipeline=delivery_pipeline, automation=automation)
    actual = CloudDeployClient.automation_path(project, location, delivery_pipeline, automation)
    assert expected == actual

def test_parse_automation_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'oyster', 'location': 'nudibranch', 'delivery_pipeline': 'cuttlefish', 'automation': 'mussel'}
    path = CloudDeployClient.automation_path(**expected)
    actual = CloudDeployClient.parse_automation_path(path)
    assert expected == actual

def test_automation_run_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    location = 'nautilus'
    delivery_pipeline = 'scallop'
    automation_run = 'abalone'
    expected = 'projects/{project}/locations/{location}/deliveryPipelines/{delivery_pipeline}/automationRuns/{automation_run}'.format(project=project, location=location, delivery_pipeline=delivery_pipeline, automation_run=automation_run)
    actual = CloudDeployClient.automation_run_path(project, location, delivery_pipeline, automation_run)
    assert expected == actual

def test_parse_automation_run_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'squid', 'location': 'clam', 'delivery_pipeline': 'whelk', 'automation_run': 'octopus'}
    path = CloudDeployClient.automation_run_path(**expected)
    actual = CloudDeployClient.parse_automation_run_path(path)
    assert expected == actual

def test_build_path():
    if False:
        i = 10
        return i + 15
    project = 'oyster'
    location = 'nudibranch'
    build = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/builds/{build}'.format(project=project, location=location, build=build)
    actual = CloudDeployClient.build_path(project, location, build)
    assert expected == actual

def test_parse_build_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'mussel', 'location': 'winkle', 'build': 'nautilus'}
    path = CloudDeployClient.build_path(**expected)
    actual = CloudDeployClient.parse_build_path(path)
    assert expected == actual

def test_cluster_path():
    if False:
        return 10
    project = 'scallop'
    location = 'abalone'
    cluster = 'squid'
    expected = 'projects/{project}/locations/{location}/clusters/{cluster}'.format(project=project, location=location, cluster=cluster)
    actual = CloudDeployClient.cluster_path(project, location, cluster)
    assert expected == actual

def test_parse_cluster_path():
    if False:
        print('Hello World!')
    expected = {'project': 'clam', 'location': 'whelk', 'cluster': 'octopus'}
    path = CloudDeployClient.cluster_path(**expected)
    actual = CloudDeployClient.parse_cluster_path(path)
    assert expected == actual

def test_config_path():
    if False:
        return 10
    project = 'oyster'
    location = 'nudibranch'
    expected = 'projects/{project}/locations/{location}/config'.format(project=project, location=location)
    actual = CloudDeployClient.config_path(project, location)
    assert expected == actual

def test_parse_config_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'cuttlefish', 'location': 'mussel'}
    path = CloudDeployClient.config_path(**expected)
    actual = CloudDeployClient.parse_config_path(path)
    assert expected == actual

def test_delivery_pipeline_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    location = 'nautilus'
    delivery_pipeline = 'scallop'
    expected = 'projects/{project}/locations/{location}/deliveryPipelines/{delivery_pipeline}'.format(project=project, location=location, delivery_pipeline=delivery_pipeline)
    actual = CloudDeployClient.delivery_pipeline_path(project, location, delivery_pipeline)
    assert expected == actual

def test_parse_delivery_pipeline_path():
    if False:
        print('Hello World!')
    expected = {'project': 'abalone', 'location': 'squid', 'delivery_pipeline': 'clam'}
    path = CloudDeployClient.delivery_pipeline_path(**expected)
    actual = CloudDeployClient.parse_delivery_pipeline_path(path)
    assert expected == actual

def test_job_path():
    if False:
        while True:
            i = 10
    project = 'whelk'
    location = 'octopus'
    job = 'oyster'
    expected = 'projects/{project}/locations/{location}/jobs/{job}'.format(project=project, location=location, job=job)
    actual = CloudDeployClient.job_path(project, location, job)
    assert expected == actual

def test_parse_job_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'job': 'mussel'}
    path = CloudDeployClient.job_path(**expected)
    actual = CloudDeployClient.parse_job_path(path)
    assert expected == actual

def test_job_run_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    location = 'nautilus'
    delivery_pipeline = 'scallop'
    release = 'abalone'
    rollout = 'squid'
    job_run = 'clam'
    expected = 'projects/{project}/locations/{location}/deliveryPipelines/{delivery_pipeline}/releases/{release}/rollouts/{rollout}/jobRuns/{job_run}'.format(project=project, location=location, delivery_pipeline=delivery_pipeline, release=release, rollout=rollout, job_run=job_run)
    actual = CloudDeployClient.job_run_path(project, location, delivery_pipeline, release, rollout, job_run)
    assert expected == actual

def test_parse_job_run_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'whelk', 'location': 'octopus', 'delivery_pipeline': 'oyster', 'release': 'nudibranch', 'rollout': 'cuttlefish', 'job_run': 'mussel'}
    path = CloudDeployClient.job_run_path(**expected)
    actual = CloudDeployClient.parse_job_run_path(path)
    assert expected == actual

def test_membership_path():
    if False:
        i = 10
        return i + 15
    project = 'winkle'
    location = 'nautilus'
    membership = 'scallop'
    expected = 'projects/{project}/locations/{location}/memberships/{membership}'.format(project=project, location=location, membership=membership)
    actual = CloudDeployClient.membership_path(project, location, membership)
    assert expected == actual

def test_parse_membership_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'abalone', 'location': 'squid', 'membership': 'clam'}
    path = CloudDeployClient.membership_path(**expected)
    actual = CloudDeployClient.parse_membership_path(path)
    assert expected == actual

def test_release_path():
    if False:
        i = 10
        return i + 15
    project = 'whelk'
    location = 'octopus'
    delivery_pipeline = 'oyster'
    release = 'nudibranch'
    expected = 'projects/{project}/locations/{location}/deliveryPipelines/{delivery_pipeline}/releases/{release}'.format(project=project, location=location, delivery_pipeline=delivery_pipeline, release=release)
    actual = CloudDeployClient.release_path(project, location, delivery_pipeline, release)
    assert expected == actual

def test_parse_release_path():
    if False:
        return 10
    expected = {'project': 'cuttlefish', 'location': 'mussel', 'delivery_pipeline': 'winkle', 'release': 'nautilus'}
    path = CloudDeployClient.release_path(**expected)
    actual = CloudDeployClient.parse_release_path(path)
    assert expected == actual

def test_rollout_path():
    if False:
        i = 10
        return i + 15
    project = 'scallop'
    location = 'abalone'
    delivery_pipeline = 'squid'
    release = 'clam'
    rollout = 'whelk'
    expected = 'projects/{project}/locations/{location}/deliveryPipelines/{delivery_pipeline}/releases/{release}/rollouts/{rollout}'.format(project=project, location=location, delivery_pipeline=delivery_pipeline, release=release, rollout=rollout)
    actual = CloudDeployClient.rollout_path(project, location, delivery_pipeline, release, rollout)
    assert expected == actual

def test_parse_rollout_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus', 'location': 'oyster', 'delivery_pipeline': 'nudibranch', 'release': 'cuttlefish', 'rollout': 'mussel'}
    path = CloudDeployClient.rollout_path(**expected)
    actual = CloudDeployClient.parse_rollout_path(path)
    assert expected == actual

def test_service_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    location = 'nautilus'
    service = 'scallop'
    expected = 'projects/{project}/locations/{location}/services/{service}'.format(project=project, location=location, service=service)
    actual = CloudDeployClient.service_path(project, location, service)
    assert expected == actual

def test_parse_service_path():
    if False:
        print('Hello World!')
    expected = {'project': 'abalone', 'location': 'squid', 'service': 'clam'}
    path = CloudDeployClient.service_path(**expected)
    actual = CloudDeployClient.parse_service_path(path)
    assert expected == actual

def test_target_path():
    if False:
        i = 10
        return i + 15
    project = 'whelk'
    location = 'octopus'
    target = 'oyster'
    expected = 'projects/{project}/locations/{location}/targets/{target}'.format(project=project, location=location, target=target)
    actual = CloudDeployClient.target_path(project, location, target)
    assert expected == actual

def test_parse_target_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'target': 'mussel'}
    path = CloudDeployClient.target_path(**expected)
    actual = CloudDeployClient.parse_target_path(path)
    assert expected == actual

def test_worker_pool_path():
    if False:
        i = 10
        return i + 15
    project = 'winkle'
    location = 'nautilus'
    worker_pool = 'scallop'
    expected = 'projects/{project}/locations/{location}/workerPools/{worker_pool}'.format(project=project, location=location, worker_pool=worker_pool)
    actual = CloudDeployClient.worker_pool_path(project, location, worker_pool)
    assert expected == actual

def test_parse_worker_pool_path():
    if False:
        print('Hello World!')
    expected = {'project': 'abalone', 'location': 'squid', 'worker_pool': 'clam'}
    path = CloudDeployClient.worker_pool_path(**expected)
    actual = CloudDeployClient.parse_worker_pool_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = CloudDeployClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'octopus'}
    path = CloudDeployClient.common_billing_account_path(**expected)
    actual = CloudDeployClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = CloudDeployClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'nudibranch'}
    path = CloudDeployClient.common_folder_path(**expected)
    actual = CloudDeployClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = CloudDeployClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'mussel'}
    path = CloudDeployClient.common_organization_path(**expected)
    actual = CloudDeployClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = CloudDeployClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'nautilus'}
    path = CloudDeployClient.common_project_path(**expected)
    actual = CloudDeployClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = CloudDeployClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'squid', 'location': 'clam'}
    path = CloudDeployClient.common_location_path(**expected)
    actual = CloudDeployClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.CloudDeployTransport, '_prep_wrapped_messages') as prep:
        client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.CloudDeployTransport, '_prep_wrapped_messages') as prep:
        transport_class = CloudDeployClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
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
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
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
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}, request)
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
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/deliveryPipelines/sample3'}
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
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        return 10
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = CloudDeployAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = CloudDeployClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(CloudDeployClient, transports.CloudDeployGrpcTransport), (CloudDeployAsyncClient, transports.CloudDeployGrpcAsyncIOTransport)])
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