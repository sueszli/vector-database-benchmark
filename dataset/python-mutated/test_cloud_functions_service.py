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
from google.type import expr_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.functions_v1.services.cloud_functions_service import CloudFunctionsServiceAsyncClient, CloudFunctionsServiceClient, pagers, transports
from google.cloud.functions_v1.types import functions, operations

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        return 10
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        print('Hello World!')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert CloudFunctionsServiceClient._get_default_mtls_endpoint(None) is None
    assert CloudFunctionsServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert CloudFunctionsServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert CloudFunctionsServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert CloudFunctionsServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert CloudFunctionsServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(CloudFunctionsServiceClient, 'grpc'), (CloudFunctionsServiceAsyncClient, 'grpc_asyncio'), (CloudFunctionsServiceClient, 'rest')])
def test_cloud_functions_service_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('cloudfunctions.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudfunctions.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.CloudFunctionsServiceGrpcTransport, 'grpc'), (transports.CloudFunctionsServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.CloudFunctionsServiceRestTransport, 'rest')])
def test_cloud_functions_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(CloudFunctionsServiceClient, 'grpc'), (CloudFunctionsServiceAsyncClient, 'grpc_asyncio'), (CloudFunctionsServiceClient, 'rest')])
def test_cloud_functions_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('cloudfunctions.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudfunctions.googleapis.com')

def test_cloud_functions_service_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = CloudFunctionsServiceClient.get_transport_class()
    available_transports = [transports.CloudFunctionsServiceGrpcTransport, transports.CloudFunctionsServiceRestTransport]
    assert transport in available_transports
    transport = CloudFunctionsServiceClient.get_transport_class('grpc')
    assert transport == transports.CloudFunctionsServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CloudFunctionsServiceClient, transports.CloudFunctionsServiceGrpcTransport, 'grpc'), (CloudFunctionsServiceAsyncClient, transports.CloudFunctionsServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (CloudFunctionsServiceClient, transports.CloudFunctionsServiceRestTransport, 'rest')])
@mock.patch.object(CloudFunctionsServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudFunctionsServiceClient))
@mock.patch.object(CloudFunctionsServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudFunctionsServiceAsyncClient))
def test_cloud_functions_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(CloudFunctionsServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(CloudFunctionsServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(CloudFunctionsServiceClient, transports.CloudFunctionsServiceGrpcTransport, 'grpc', 'true'), (CloudFunctionsServiceAsyncClient, transports.CloudFunctionsServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (CloudFunctionsServiceClient, transports.CloudFunctionsServiceGrpcTransport, 'grpc', 'false'), (CloudFunctionsServiceAsyncClient, transports.CloudFunctionsServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (CloudFunctionsServiceClient, transports.CloudFunctionsServiceRestTransport, 'rest', 'true'), (CloudFunctionsServiceClient, transports.CloudFunctionsServiceRestTransport, 'rest', 'false')])
@mock.patch.object(CloudFunctionsServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudFunctionsServiceClient))
@mock.patch.object(CloudFunctionsServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudFunctionsServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_cloud_functions_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [CloudFunctionsServiceClient, CloudFunctionsServiceAsyncClient])
@mock.patch.object(CloudFunctionsServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudFunctionsServiceClient))
@mock.patch.object(CloudFunctionsServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudFunctionsServiceAsyncClient))
def test_cloud_functions_service_client_get_mtls_endpoint_and_cert_source(client_class):
    if False:
        return 10
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CloudFunctionsServiceClient, transports.CloudFunctionsServiceGrpcTransport, 'grpc'), (CloudFunctionsServiceAsyncClient, transports.CloudFunctionsServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (CloudFunctionsServiceClient, transports.CloudFunctionsServiceRestTransport, 'rest')])
def test_cloud_functions_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CloudFunctionsServiceClient, transports.CloudFunctionsServiceGrpcTransport, 'grpc', grpc_helpers), (CloudFunctionsServiceAsyncClient, transports.CloudFunctionsServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (CloudFunctionsServiceClient, transports.CloudFunctionsServiceRestTransport, 'rest', None)])
def test_cloud_functions_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_cloud_functions_service_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.functions_v1.services.cloud_functions_service.transports.CloudFunctionsServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = CloudFunctionsServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CloudFunctionsServiceClient, transports.CloudFunctionsServiceGrpcTransport, 'grpc', grpc_helpers), (CloudFunctionsServiceAsyncClient, transports.CloudFunctionsServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_cloud_functions_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('cloudfunctions.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='cloudfunctions.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [functions.ListFunctionsRequest, dict])
def test_list_functions(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_functions), '__call__') as call:
        call.return_value = functions.ListFunctionsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_functions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.ListFunctionsRequest()
    assert isinstance(response, pagers.ListFunctionsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_functions_empty_call():
    if False:
        print('Hello World!')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_functions), '__call__') as call:
        client.list_functions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.ListFunctionsRequest()

@pytest.mark.asyncio
async def test_list_functions_async(transport: str='grpc_asyncio', request_type=functions.ListFunctionsRequest):
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_functions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.ListFunctionsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_functions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.ListFunctionsRequest()
    assert isinstance(response, pagers.ListFunctionsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_functions_async_from_dict():
    await test_list_functions_async(request_type=dict)

def test_list_functions_field_headers():
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.ListFunctionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_functions), '__call__') as call:
        call.return_value = functions.ListFunctionsResponse()
        client.list_functions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_functions_field_headers_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.ListFunctionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_functions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.ListFunctionsResponse())
        await client.list_functions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_functions_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_functions), '__call__') as call:
        call.side_effect = (functions.ListFunctionsResponse(functions=[functions.CloudFunction(), functions.CloudFunction(), functions.CloudFunction()], next_page_token='abc'), functions.ListFunctionsResponse(functions=[], next_page_token='def'), functions.ListFunctionsResponse(functions=[functions.CloudFunction()], next_page_token='ghi'), functions.ListFunctionsResponse(functions=[functions.CloudFunction(), functions.CloudFunction()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_functions(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, functions.CloudFunction) for i in results))

def test_list_functions_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_functions), '__call__') as call:
        call.side_effect = (functions.ListFunctionsResponse(functions=[functions.CloudFunction(), functions.CloudFunction(), functions.CloudFunction()], next_page_token='abc'), functions.ListFunctionsResponse(functions=[], next_page_token='def'), functions.ListFunctionsResponse(functions=[functions.CloudFunction()], next_page_token='ghi'), functions.ListFunctionsResponse(functions=[functions.CloudFunction(), functions.CloudFunction()]), RuntimeError)
        pages = list(client.list_functions(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_functions_async_pager():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_functions), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (functions.ListFunctionsResponse(functions=[functions.CloudFunction(), functions.CloudFunction(), functions.CloudFunction()], next_page_token='abc'), functions.ListFunctionsResponse(functions=[], next_page_token='def'), functions.ListFunctionsResponse(functions=[functions.CloudFunction()], next_page_token='ghi'), functions.ListFunctionsResponse(functions=[functions.CloudFunction(), functions.CloudFunction()]), RuntimeError)
        async_pager = await client.list_functions(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, functions.CloudFunction) for i in responses))

@pytest.mark.asyncio
async def test_list_functions_async_pages():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_functions), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (functions.ListFunctionsResponse(functions=[functions.CloudFunction(), functions.CloudFunction(), functions.CloudFunction()], next_page_token='abc'), functions.ListFunctionsResponse(functions=[], next_page_token='def'), functions.ListFunctionsResponse(functions=[functions.CloudFunction()], next_page_token='ghi'), functions.ListFunctionsResponse(functions=[functions.CloudFunction(), functions.CloudFunction()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_functions(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [functions.GetFunctionRequest, dict])
def test_get_function(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_function), '__call__') as call:
        call.return_value = functions.CloudFunction(name='name_value', description='description_value', status=functions.CloudFunctionStatus.ACTIVE, entry_point='entry_point_value', runtime='runtime_value', available_memory_mb=1991, service_account_email='service_account_email_value', version_id=1074, network='network_value', max_instances=1389, min_instances=1387, vpc_connector='vpc_connector_value', vpc_connector_egress_settings=functions.CloudFunction.VpcConnectorEgressSettings.PRIVATE_RANGES_ONLY, ingress_settings=functions.CloudFunction.IngressSettings.ALLOW_ALL, kms_key_name='kms_key_name_value', build_worker_pool='build_worker_pool_value', build_id='build_id_value', build_name='build_name_value', source_token='source_token_value', docker_repository='docker_repository_value', docker_registry=functions.CloudFunction.DockerRegistry.CONTAINER_REGISTRY, source_archive_url='source_archive_url_value')
        response = client.get_function(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.GetFunctionRequest()
    assert isinstance(response, functions.CloudFunction)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.status == functions.CloudFunctionStatus.ACTIVE
    assert response.entry_point == 'entry_point_value'
    assert response.runtime == 'runtime_value'
    assert response.available_memory_mb == 1991
    assert response.service_account_email == 'service_account_email_value'
    assert response.version_id == 1074
    assert response.network == 'network_value'
    assert response.max_instances == 1389
    assert response.min_instances == 1387
    assert response.vpc_connector == 'vpc_connector_value'
    assert response.vpc_connector_egress_settings == functions.CloudFunction.VpcConnectorEgressSettings.PRIVATE_RANGES_ONLY
    assert response.ingress_settings == functions.CloudFunction.IngressSettings.ALLOW_ALL
    assert response.kms_key_name == 'kms_key_name_value'
    assert response.build_worker_pool == 'build_worker_pool_value'
    assert response.build_id == 'build_id_value'
    assert response.build_name == 'build_name_value'
    assert response.source_token == 'source_token_value'
    assert response.docker_repository == 'docker_repository_value'
    assert response.docker_registry == functions.CloudFunction.DockerRegistry.CONTAINER_REGISTRY

def test_get_function_empty_call():
    if False:
        print('Hello World!')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_function), '__call__') as call:
        client.get_function()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.GetFunctionRequest()

@pytest.mark.asyncio
async def test_get_function_async(transport: str='grpc_asyncio', request_type=functions.GetFunctionRequest):
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_function), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.CloudFunction(name='name_value', description='description_value', status=functions.CloudFunctionStatus.ACTIVE, entry_point='entry_point_value', runtime='runtime_value', available_memory_mb=1991, service_account_email='service_account_email_value', version_id=1074, network='network_value', max_instances=1389, min_instances=1387, vpc_connector='vpc_connector_value', vpc_connector_egress_settings=functions.CloudFunction.VpcConnectorEgressSettings.PRIVATE_RANGES_ONLY, ingress_settings=functions.CloudFunction.IngressSettings.ALLOW_ALL, kms_key_name='kms_key_name_value', build_worker_pool='build_worker_pool_value', build_id='build_id_value', build_name='build_name_value', source_token='source_token_value', docker_repository='docker_repository_value', docker_registry=functions.CloudFunction.DockerRegistry.CONTAINER_REGISTRY))
        response = await client.get_function(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.GetFunctionRequest()
    assert isinstance(response, functions.CloudFunction)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.status == functions.CloudFunctionStatus.ACTIVE
    assert response.entry_point == 'entry_point_value'
    assert response.runtime == 'runtime_value'
    assert response.available_memory_mb == 1991
    assert response.service_account_email == 'service_account_email_value'
    assert response.version_id == 1074
    assert response.network == 'network_value'
    assert response.max_instances == 1389
    assert response.min_instances == 1387
    assert response.vpc_connector == 'vpc_connector_value'
    assert response.vpc_connector_egress_settings == functions.CloudFunction.VpcConnectorEgressSettings.PRIVATE_RANGES_ONLY
    assert response.ingress_settings == functions.CloudFunction.IngressSettings.ALLOW_ALL
    assert response.kms_key_name == 'kms_key_name_value'
    assert response.build_worker_pool == 'build_worker_pool_value'
    assert response.build_id == 'build_id_value'
    assert response.build_name == 'build_name_value'
    assert response.source_token == 'source_token_value'
    assert response.docker_repository == 'docker_repository_value'
    assert response.docker_registry == functions.CloudFunction.DockerRegistry.CONTAINER_REGISTRY

@pytest.mark.asyncio
async def test_get_function_async_from_dict():
    await test_get_function_async(request_type=dict)

def test_get_function_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.GetFunctionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_function), '__call__') as call:
        call.return_value = functions.CloudFunction()
        client.get_function(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_function_field_headers_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.GetFunctionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_function), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.CloudFunction())
        await client.get_function(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_function_flattened():
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_function), '__call__') as call:
        call.return_value = functions.CloudFunction()
        client.get_function(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_function_flattened_error():
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_function(functions.GetFunctionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_function_flattened_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_function), '__call__') as call:
        call.return_value = functions.CloudFunction()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.CloudFunction())
        response = await client.get_function(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_function_flattened_error_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_function(functions.GetFunctionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [functions.CreateFunctionRequest, dict])
def test_create_function(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_function), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_function(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.CreateFunctionRequest()
    assert isinstance(response, future.Future)

def test_create_function_empty_call():
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_function), '__call__') as call:
        client.create_function()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.CreateFunctionRequest()

@pytest.mark.asyncio
async def test_create_function_async(transport: str='grpc_asyncio', request_type=functions.CreateFunctionRequest):
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_function), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_function(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.CreateFunctionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_function_async_from_dict():
    await test_create_function_async(request_type=dict)

def test_create_function_field_headers():
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.CreateFunctionRequest()
    request.location = 'location_value'
    with mock.patch.object(type(client.transport.create_function), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_function(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'location=location_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_function_field_headers_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.CreateFunctionRequest()
    request.location = 'location_value'
    with mock.patch.object(type(client.transport.create_function), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_function(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'location=location_value') in kw['metadata']

def test_create_function_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_function), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_function(location='location_value', function=functions.CloudFunction(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].location
        mock_val = 'location_value'
        assert arg == mock_val
        arg = args[0].function
        mock_val = functions.CloudFunction(name='name_value')
        assert arg == mock_val

def test_create_function_flattened_error():
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_function(functions.CreateFunctionRequest(), location='location_value', function=functions.CloudFunction(name='name_value'))

@pytest.mark.asyncio
async def test_create_function_flattened_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_function), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_function(location='location_value', function=functions.CloudFunction(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].location
        mock_val = 'location_value'
        assert arg == mock_val
        arg = args[0].function
        mock_val = functions.CloudFunction(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_function_flattened_error_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_function(functions.CreateFunctionRequest(), location='location_value', function=functions.CloudFunction(name='name_value'))

@pytest.mark.parametrize('request_type', [functions.UpdateFunctionRequest, dict])
def test_update_function(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_function), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_function(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.UpdateFunctionRequest()
    assert isinstance(response, future.Future)

def test_update_function_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_function), '__call__') as call:
        client.update_function()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.UpdateFunctionRequest()

@pytest.mark.asyncio
async def test_update_function_async(transport: str='grpc_asyncio', request_type=functions.UpdateFunctionRequest):
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_function), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_function(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.UpdateFunctionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_function_async_from_dict():
    await test_update_function_async(request_type=dict)

def test_update_function_field_headers():
    if False:
        while True:
            i = 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.UpdateFunctionRequest()
    request.function.name = 'name_value'
    with mock.patch.object(type(client.transport.update_function), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_function(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'function.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_function_field_headers_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.UpdateFunctionRequest()
    request.function.name = 'name_value'
    with mock.patch.object(type(client.transport.update_function), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_function(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'function.name=name_value') in kw['metadata']

def test_update_function_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_function), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_function(function=functions.CloudFunction(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].function
        mock_val = functions.CloudFunction(name='name_value')
        assert arg == mock_val

def test_update_function_flattened_error():
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_function(functions.UpdateFunctionRequest(), function=functions.CloudFunction(name='name_value'))

@pytest.mark.asyncio
async def test_update_function_flattened_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_function), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_function(function=functions.CloudFunction(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].function
        mock_val = functions.CloudFunction(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_function_flattened_error_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_function(functions.UpdateFunctionRequest(), function=functions.CloudFunction(name='name_value'))

@pytest.mark.parametrize('request_type', [functions.DeleteFunctionRequest, dict])
def test_delete_function(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_function), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_function(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.DeleteFunctionRequest()
    assert isinstance(response, future.Future)

def test_delete_function_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_function), '__call__') as call:
        client.delete_function()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.DeleteFunctionRequest()

@pytest.mark.asyncio
async def test_delete_function_async(transport: str='grpc_asyncio', request_type=functions.DeleteFunctionRequest):
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_function), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_function(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.DeleteFunctionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_function_async_from_dict():
    await test_delete_function_async(request_type=dict)

def test_delete_function_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.DeleteFunctionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_function), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_function(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_function_field_headers_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.DeleteFunctionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_function), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_function(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_function_flattened():
    if False:
        while True:
            i = 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_function), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_function(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_function_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_function(functions.DeleteFunctionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_function_flattened_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_function), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_function(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_function_flattened_error_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_function(functions.DeleteFunctionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [functions.CallFunctionRequest, dict])
def test_call_function(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.call_function), '__call__') as call:
        call.return_value = functions.CallFunctionResponse(execution_id='execution_id_value', result='result_value', error='error_value')
        response = client.call_function(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.CallFunctionRequest()
    assert isinstance(response, functions.CallFunctionResponse)
    assert response.execution_id == 'execution_id_value'
    assert response.result == 'result_value'
    assert response.error == 'error_value'

def test_call_function_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.call_function), '__call__') as call:
        client.call_function()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.CallFunctionRequest()

@pytest.mark.asyncio
async def test_call_function_async(transport: str='grpc_asyncio', request_type=functions.CallFunctionRequest):
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.call_function), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.CallFunctionResponse(execution_id='execution_id_value', result='result_value', error='error_value'))
        response = await client.call_function(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.CallFunctionRequest()
    assert isinstance(response, functions.CallFunctionResponse)
    assert response.execution_id == 'execution_id_value'
    assert response.result == 'result_value'
    assert response.error == 'error_value'

@pytest.mark.asyncio
async def test_call_function_async_from_dict():
    await test_call_function_async(request_type=dict)

def test_call_function_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.CallFunctionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.call_function), '__call__') as call:
        call.return_value = functions.CallFunctionResponse()
        client.call_function(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_call_function_field_headers_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.CallFunctionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.call_function), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.CallFunctionResponse())
        await client.call_function(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_call_function_flattened():
    if False:
        while True:
            i = 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.call_function), '__call__') as call:
        call.return_value = functions.CallFunctionResponse()
        client.call_function(name='name_value', data='data_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].data
        mock_val = 'data_value'
        assert arg == mock_val

def test_call_function_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.call_function(functions.CallFunctionRequest(), name='name_value', data='data_value')

@pytest.mark.asyncio
async def test_call_function_flattened_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.call_function), '__call__') as call:
        call.return_value = functions.CallFunctionResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.CallFunctionResponse())
        response = await client.call_function(name='name_value', data='data_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].data
        mock_val = 'data_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_call_function_flattened_error_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.call_function(functions.CallFunctionRequest(), name='name_value', data='data_value')

@pytest.mark.parametrize('request_type', [functions.GenerateUploadUrlRequest, dict])
def test_generate_upload_url(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_upload_url), '__call__') as call:
        call.return_value = functions.GenerateUploadUrlResponse(upload_url='upload_url_value')
        response = client.generate_upload_url(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.GenerateUploadUrlRequest()
    assert isinstance(response, functions.GenerateUploadUrlResponse)
    assert response.upload_url == 'upload_url_value'

def test_generate_upload_url_empty_call():
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.generate_upload_url), '__call__') as call:
        client.generate_upload_url()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.GenerateUploadUrlRequest()

@pytest.mark.asyncio
async def test_generate_upload_url_async(transport: str='grpc_asyncio', request_type=functions.GenerateUploadUrlRequest):
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_upload_url), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.GenerateUploadUrlResponse(upload_url='upload_url_value'))
        response = await client.generate_upload_url(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.GenerateUploadUrlRequest()
    assert isinstance(response, functions.GenerateUploadUrlResponse)
    assert response.upload_url == 'upload_url_value'

@pytest.mark.asyncio
async def test_generate_upload_url_async_from_dict():
    await test_generate_upload_url_async(request_type=dict)

def test_generate_upload_url_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.GenerateUploadUrlRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.generate_upload_url), '__call__') as call:
        call.return_value = functions.GenerateUploadUrlResponse()
        client.generate_upload_url(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_generate_upload_url_field_headers_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.GenerateUploadUrlRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.generate_upload_url), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.GenerateUploadUrlResponse())
        await client.generate_upload_url(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [functions.GenerateDownloadUrlRequest, dict])
def test_generate_download_url(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_download_url), '__call__') as call:
        call.return_value = functions.GenerateDownloadUrlResponse(download_url='download_url_value')
        response = client.generate_download_url(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.GenerateDownloadUrlRequest()
    assert isinstance(response, functions.GenerateDownloadUrlResponse)
    assert response.download_url == 'download_url_value'

def test_generate_download_url_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.generate_download_url), '__call__') as call:
        client.generate_download_url()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.GenerateDownloadUrlRequest()

@pytest.mark.asyncio
async def test_generate_download_url_async(transport: str='grpc_asyncio', request_type=functions.GenerateDownloadUrlRequest):
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_download_url), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.GenerateDownloadUrlResponse(download_url='download_url_value'))
        response = await client.generate_download_url(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.GenerateDownloadUrlRequest()
    assert isinstance(response, functions.GenerateDownloadUrlResponse)
    assert response.download_url == 'download_url_value'

@pytest.mark.asyncio
async def test_generate_download_url_async_from_dict():
    await test_generate_download_url_async(request_type=dict)

def test_generate_download_url_field_headers():
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.GenerateDownloadUrlRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.generate_download_url), '__call__') as call:
        call.return_value = functions.GenerateDownloadUrlResponse()
        client.generate_download_url(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_generate_download_url_field_headers_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.GenerateDownloadUrlRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.generate_download_url), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.GenerateDownloadUrlResponse())
        await client.generate_download_url(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_empty_call():
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        client.set_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()

@pytest.mark.asyncio
async def test_set_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.SetIamPolicyRequest):
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.set_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_set_iam_policy_async_from_dict():
    await test_set_iam_policy_async(request_type=dict)

def test_set_iam_policy_field_headers():
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_set_iam_policy_field_headers_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.set_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_set_iam_policy_from_dict_foreign():
    if False:
        while True:
            i = 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774), 'update_mask': field_mask_pb2.FieldMask(paths=['paths_value'])})
        call.assert_called()

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        client.get_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()

@pytest.mark.asyncio
async def test_get_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.GetIamPolicyRequest):
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_get_iam_policy_async_from_dict():
    await test_get_iam_policy_async(request_type=dict)

def test_get_iam_policy_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_iam_policy_field_headers_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_get_iam_policy_from_dict_foreign():
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response = client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_empty_call():
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        client.test_iam_permissions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()

@pytest.mark.asyncio
async def test_test_iam_permissions_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value']))
        response = await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

@pytest.mark.asyncio
async def test_test_iam_permissions_async_from_dict():
    await test_test_iam_permissions_async(request_type=dict)

def test_test_iam_permissions_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_test_iam_permissions_field_headers_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_test_iam_permissions_from_dict_foreign():
    if False:
        print('Hello World!')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.parametrize('request_type', [functions.ListFunctionsRequest, dict])
def test_list_functions_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = functions.ListFunctionsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = functions.ListFunctionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_functions(request)
    assert isinstance(response, pagers.ListFunctionsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_functions_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudFunctionsServiceRestInterceptor())
    client = CloudFunctionsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'post_list_functions') as post, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'pre_list_functions') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = functions.ListFunctionsRequest.pb(functions.ListFunctionsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = functions.ListFunctionsResponse.to_json(functions.ListFunctionsResponse())
        request = functions.ListFunctionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = functions.ListFunctionsResponse()
        client.list_functions(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_functions_rest_bad_request(transport: str='rest', request_type=functions.ListFunctionsRequest):
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_functions(request)

def test_list_functions_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (functions.ListFunctionsResponse(functions=[functions.CloudFunction(), functions.CloudFunction(), functions.CloudFunction()], next_page_token='abc'), functions.ListFunctionsResponse(functions=[], next_page_token='def'), functions.ListFunctionsResponse(functions=[functions.CloudFunction()], next_page_token='ghi'), functions.ListFunctionsResponse(functions=[functions.CloudFunction(), functions.CloudFunction()]))
        response = response + response
        response = tuple((functions.ListFunctionsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_functions(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, functions.CloudFunction) for i in results))
        pages = list(client.list_functions(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [functions.GetFunctionRequest, dict])
def test_get_function_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/functions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = functions.CloudFunction(name='name_value', description='description_value', status=functions.CloudFunctionStatus.ACTIVE, entry_point='entry_point_value', runtime='runtime_value', available_memory_mb=1991, service_account_email='service_account_email_value', version_id=1074, network='network_value', max_instances=1389, min_instances=1387, vpc_connector='vpc_connector_value', vpc_connector_egress_settings=functions.CloudFunction.VpcConnectorEgressSettings.PRIVATE_RANGES_ONLY, ingress_settings=functions.CloudFunction.IngressSettings.ALLOW_ALL, kms_key_name='kms_key_name_value', build_worker_pool='build_worker_pool_value', build_id='build_id_value', build_name='build_name_value', source_token='source_token_value', docker_repository='docker_repository_value', docker_registry=functions.CloudFunction.DockerRegistry.CONTAINER_REGISTRY, source_archive_url='source_archive_url_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = functions.CloudFunction.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_function(request)
    assert isinstance(response, functions.CloudFunction)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.status == functions.CloudFunctionStatus.ACTIVE
    assert response.entry_point == 'entry_point_value'
    assert response.runtime == 'runtime_value'
    assert response.available_memory_mb == 1991
    assert response.service_account_email == 'service_account_email_value'
    assert response.version_id == 1074
    assert response.network == 'network_value'
    assert response.max_instances == 1389
    assert response.min_instances == 1387
    assert response.vpc_connector == 'vpc_connector_value'
    assert response.vpc_connector_egress_settings == functions.CloudFunction.VpcConnectorEgressSettings.PRIVATE_RANGES_ONLY
    assert response.ingress_settings == functions.CloudFunction.IngressSettings.ALLOW_ALL
    assert response.kms_key_name == 'kms_key_name_value'
    assert response.build_worker_pool == 'build_worker_pool_value'
    assert response.build_id == 'build_id_value'
    assert response.build_name == 'build_name_value'
    assert response.source_token == 'source_token_value'
    assert response.docker_repository == 'docker_repository_value'
    assert response.docker_registry == functions.CloudFunction.DockerRegistry.CONTAINER_REGISTRY

def test_get_function_rest_required_fields(request_type=functions.GetFunctionRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudFunctionsServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_function._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_function._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = functions.CloudFunction()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = functions.CloudFunction.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_function(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_function_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_function._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_function_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudFunctionsServiceRestInterceptor())
    client = CloudFunctionsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'post_get_function') as post, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'pre_get_function') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = functions.GetFunctionRequest.pb(functions.GetFunctionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = functions.CloudFunction.to_json(functions.CloudFunction())
        request = functions.GetFunctionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = functions.CloudFunction()
        client.get_function(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_function_rest_bad_request(transport: str='rest', request_type=functions.GetFunctionRequest):
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/functions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_function(request)

def test_get_function_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = functions.CloudFunction()
        sample_request = {'name': 'projects/sample1/locations/sample2/functions/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = functions.CloudFunction.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_function(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/functions/*}' % client.transport._host, args[1])

def test_get_function_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_function(functions.GetFunctionRequest(), name='name_value')

def test_get_function_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [functions.CreateFunctionRequest, dict])
def test_create_function_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'location': 'projects/sample1/locations/sample2'}
    request_init['function'] = {'name': 'name_value', 'description': 'description_value', 'source_archive_url': 'source_archive_url_value', 'source_repository': {'url': 'url_value', 'deployed_url': 'deployed_url_value'}, 'source_upload_url': 'source_upload_url_value', 'https_trigger': {'url': 'url_value', 'security_level': 1}, 'event_trigger': {'event_type': 'event_type_value', 'resource': 'resource_value', 'service': 'service_value', 'failure_policy': {'retry': {}}}, 'status': 1, 'entry_point': 'entry_point_value', 'runtime': 'runtime_value', 'timeout': {'seconds': 751, 'nanos': 543}, 'available_memory_mb': 1991, 'service_account_email': 'service_account_email_value', 'update_time': {'seconds': 751, 'nanos': 543}, 'version_id': 1074, 'labels': {}, 'environment_variables': {}, 'build_environment_variables': {}, 'network': 'network_value', 'max_instances': 1389, 'min_instances': 1387, 'vpc_connector': 'vpc_connector_value', 'vpc_connector_egress_settings': 1, 'ingress_settings': 1, 'kms_key_name': 'kms_key_name_value', 'build_worker_pool': 'build_worker_pool_value', 'build_id': 'build_id_value', 'build_name': 'build_name_value', 'secret_environment_variables': [{'key': 'key_value', 'project_id': 'project_id_value', 'secret': 'secret_value', 'version': 'version_value'}], 'secret_volumes': [{'mount_path': 'mount_path_value', 'project_id': 'project_id_value', 'secret': 'secret_value', 'versions': [{'version': 'version_value', 'path': 'path_value'}]}], 'source_token': 'source_token_value', 'docker_repository': 'docker_repository_value', 'docker_registry': 1}
    test_field = functions.CreateFunctionRequest.meta.fields['function']

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
    for (field, value) in request_init['function'].items():
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
                for i in range(0, len(request_init['function'][field])):
                    del request_init['function'][field][i][subfield]
            else:
                del request_init['function'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_function(request)
    assert response.operation.name == 'operations/spam'

def test_create_function_rest_required_fields(request_type=functions.CreateFunctionRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudFunctionsServiceRestTransport
    request_init = {}
    request_init['location'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_function._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['location'] = 'location_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_function._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'location' in jsonified_request
    assert jsonified_request['location'] == 'location_value'
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_function(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_function_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_function._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('location', 'function'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_function_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudFunctionsServiceRestInterceptor())
    client = CloudFunctionsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'post_create_function') as post, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'pre_create_function') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = functions.CreateFunctionRequest.pb(functions.CreateFunctionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = functions.CreateFunctionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_function(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_function_rest_bad_request(transport: str='rest', request_type=functions.CreateFunctionRequest):
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'location': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_function(request)

def test_create_function_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'location': 'projects/sample1/locations/sample2'}
        mock_args = dict(location='location_value', function=functions.CloudFunction(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_function(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{location=projects/*/locations/*}/functions' % client.transport._host, args[1])

def test_create_function_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_function(functions.CreateFunctionRequest(), location='location_value', function=functions.CloudFunction(name='name_value'))

def test_create_function_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [functions.UpdateFunctionRequest, dict])
def test_update_function_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'function': {'name': 'projects/sample1/locations/sample2/functions/sample3'}}
    request_init['function'] = {'name': 'projects/sample1/locations/sample2/functions/sample3', 'description': 'description_value', 'source_archive_url': 'source_archive_url_value', 'source_repository': {'url': 'url_value', 'deployed_url': 'deployed_url_value'}, 'source_upload_url': 'source_upload_url_value', 'https_trigger': {'url': 'url_value', 'security_level': 1}, 'event_trigger': {'event_type': 'event_type_value', 'resource': 'resource_value', 'service': 'service_value', 'failure_policy': {'retry': {}}}, 'status': 1, 'entry_point': 'entry_point_value', 'runtime': 'runtime_value', 'timeout': {'seconds': 751, 'nanos': 543}, 'available_memory_mb': 1991, 'service_account_email': 'service_account_email_value', 'update_time': {'seconds': 751, 'nanos': 543}, 'version_id': 1074, 'labels': {}, 'environment_variables': {}, 'build_environment_variables': {}, 'network': 'network_value', 'max_instances': 1389, 'min_instances': 1387, 'vpc_connector': 'vpc_connector_value', 'vpc_connector_egress_settings': 1, 'ingress_settings': 1, 'kms_key_name': 'kms_key_name_value', 'build_worker_pool': 'build_worker_pool_value', 'build_id': 'build_id_value', 'build_name': 'build_name_value', 'secret_environment_variables': [{'key': 'key_value', 'project_id': 'project_id_value', 'secret': 'secret_value', 'version': 'version_value'}], 'secret_volumes': [{'mount_path': 'mount_path_value', 'project_id': 'project_id_value', 'secret': 'secret_value', 'versions': [{'version': 'version_value', 'path': 'path_value'}]}], 'source_token': 'source_token_value', 'docker_repository': 'docker_repository_value', 'docker_registry': 1}
    test_field = functions.UpdateFunctionRequest.meta.fields['function']

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
    for (field, value) in request_init['function'].items():
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
                for i in range(0, len(request_init['function'][field])):
                    del request_init['function'][field][i][subfield]
            else:
                del request_init['function'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_function(request)
    assert response.operation.name == 'operations/spam'

def test_update_function_rest_required_fields(request_type=functions.UpdateFunctionRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudFunctionsServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_function._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_function._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_function(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_function_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_function._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('function',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_function_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudFunctionsServiceRestInterceptor())
    client = CloudFunctionsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'post_update_function') as post, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'pre_update_function') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = functions.UpdateFunctionRequest.pb(functions.UpdateFunctionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = functions.UpdateFunctionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_function(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_function_rest_bad_request(transport: str='rest', request_type=functions.UpdateFunctionRequest):
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'function': {'name': 'projects/sample1/locations/sample2/functions/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_function(request)

def test_update_function_rest_flattened():
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'function': {'name': 'projects/sample1/locations/sample2/functions/sample3'}}
        mock_args = dict(function=functions.CloudFunction(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_function(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{function.name=projects/*/locations/*/functions/*}' % client.transport._host, args[1])

def test_update_function_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_function(functions.UpdateFunctionRequest(), function=functions.CloudFunction(name='name_value'))

def test_update_function_rest_error():
    if False:
        while True:
            i = 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [functions.DeleteFunctionRequest, dict])
def test_delete_function_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/functions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_function(request)
    assert response.operation.name == 'operations/spam'

def test_delete_function_rest_required_fields(request_type=functions.DeleteFunctionRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CloudFunctionsServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_function._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_function._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_function(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_function_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_function._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_function_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudFunctionsServiceRestInterceptor())
    client = CloudFunctionsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'post_delete_function') as post, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'pre_delete_function') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = functions.DeleteFunctionRequest.pb(functions.DeleteFunctionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = functions.DeleteFunctionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_function(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_function_rest_bad_request(transport: str='rest', request_type=functions.DeleteFunctionRequest):
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/functions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_function(request)

def test_delete_function_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/functions/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_function(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/functions/*}' % client.transport._host, args[1])

def test_delete_function_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_function(functions.DeleteFunctionRequest(), name='name_value')

def test_delete_function_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [functions.CallFunctionRequest, dict])
def test_call_function_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/functions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = functions.CallFunctionResponse(execution_id='execution_id_value', result='result_value', error='error_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = functions.CallFunctionResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.call_function(request)
    assert isinstance(response, functions.CallFunctionResponse)
    assert response.execution_id == 'execution_id_value'
    assert response.result == 'result_value'
    assert response.error == 'error_value'

def test_call_function_rest_required_fields(request_type=functions.CallFunctionRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudFunctionsServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['data'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).call_function._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    jsonified_request['data'] = 'data_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).call_function._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'data' in jsonified_request
    assert jsonified_request['data'] == 'data_value'
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = functions.CallFunctionResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = functions.CallFunctionResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.call_function(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_call_function_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.call_function._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'data'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_call_function_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudFunctionsServiceRestInterceptor())
    client = CloudFunctionsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'post_call_function') as post, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'pre_call_function') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = functions.CallFunctionRequest.pb(functions.CallFunctionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = functions.CallFunctionResponse.to_json(functions.CallFunctionResponse())
        request = functions.CallFunctionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = functions.CallFunctionResponse()
        client.call_function(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_call_function_rest_bad_request(transport: str='rest', request_type=functions.CallFunctionRequest):
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/functions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.call_function(request)

def test_call_function_rest_flattened():
    if False:
        print('Hello World!')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = functions.CallFunctionResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/functions/sample3'}
        mock_args = dict(name='name_value', data='data_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = functions.CallFunctionResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.call_function(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/functions/*}:call' % client.transport._host, args[1])

def test_call_function_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.call_function(functions.CallFunctionRequest(), name='name_value', data='data_value')

def test_call_function_rest_error():
    if False:
        while True:
            i = 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [functions.GenerateUploadUrlRequest, dict])
def test_generate_upload_url_rest(request_type):
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = functions.GenerateUploadUrlResponse(upload_url='upload_url_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = functions.GenerateUploadUrlResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.generate_upload_url(request)
    assert isinstance(response, functions.GenerateUploadUrlResponse)
    assert response.upload_url == 'upload_url_value'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_generate_upload_url_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudFunctionsServiceRestInterceptor())
    client = CloudFunctionsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'post_generate_upload_url') as post, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'pre_generate_upload_url') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = functions.GenerateUploadUrlRequest.pb(functions.GenerateUploadUrlRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = functions.GenerateUploadUrlResponse.to_json(functions.GenerateUploadUrlResponse())
        request = functions.GenerateUploadUrlRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = functions.GenerateUploadUrlResponse()
        client.generate_upload_url(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_generate_upload_url_rest_bad_request(transport: str='rest', request_type=functions.GenerateUploadUrlRequest):
    if False:
        print('Hello World!')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.generate_upload_url(request)

def test_generate_upload_url_rest_error():
    if False:
        while True:
            i = 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [functions.GenerateDownloadUrlRequest, dict])
def test_generate_download_url_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/functions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = functions.GenerateDownloadUrlResponse(download_url='download_url_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = functions.GenerateDownloadUrlResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.generate_download_url(request)
    assert isinstance(response, functions.GenerateDownloadUrlResponse)
    assert response.download_url == 'download_url_value'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_generate_download_url_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudFunctionsServiceRestInterceptor())
    client = CloudFunctionsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'post_generate_download_url') as post, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'pre_generate_download_url') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = functions.GenerateDownloadUrlRequest.pb(functions.GenerateDownloadUrlRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = functions.GenerateDownloadUrlResponse.to_json(functions.GenerateDownloadUrlResponse())
        request = functions.GenerateDownloadUrlRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = functions.GenerateDownloadUrlResponse()
        client.generate_download_url(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_generate_download_url_rest_bad_request(transport: str='rest', request_type=functions.GenerateDownloadUrlRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/functions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.generate_download_url(request)

def test_generate_download_url_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/functions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_rest_required_fields(request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudFunctionsServiceRestTransport
    request_init = {}
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = policy_pb2.Policy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.set_iam_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_iam_policy_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'policy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_iam_policy_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudFunctionsServiceRestInterceptor())
    client = CloudFunctionsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'post_set_iam_policy') as post, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'pre_set_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.SetIamPolicyRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(policy_pb2.Policy())
        request = iam_policy_pb2.SetIamPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = policy_pb2.Policy()
        client.set_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1/locations/sample2/functions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

def test_set_iam_policy_rest_error():
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/functions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_rest_required_fields(request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CloudFunctionsServiceRestTransport
    request_init = {}
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('options',))
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = policy_pb2.Policy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_iam_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_iam_policy_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(('options',)) & set(('resource',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_iam_policy_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudFunctionsServiceRestInterceptor())
    client = CloudFunctionsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'post_get_iam_policy') as post, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'pre_get_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.GetIamPolicyRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(policy_pb2.Policy())
        request = iam_policy_pb2.GetIamPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = policy_pb2.Policy()
        client.get_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        while True:
            i = 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1/locations/sample2/functions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

def test_get_iam_policy_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/functions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.test_iam_permissions(request)
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_rest_required_fields(request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudFunctionsServiceRestTransport
    request_init = {}
    request_init['resource'] = ''
    request_init['permissions'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    jsonified_request['permissions'] = 'permissions_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    assert 'permissions' in jsonified_request
    assert jsonified_request['permissions'] == 'permissions_value'
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = iam_policy_pb2.TestIamPermissionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.test_iam_permissions(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_test_iam_permissions_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.test_iam_permissions._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'permissions'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_test_iam_permissions_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudFunctionsServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudFunctionsServiceRestInterceptor())
    client = CloudFunctionsServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'post_test_iam_permissions') as post, mock.patch.object(transports.CloudFunctionsServiceRestInterceptor, 'pre_test_iam_permissions') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.TestIamPermissionsRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(iam_policy_pb2.TestIamPermissionsResponse())
        request = iam_policy_pb2.TestIamPermissionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_test_iam_permissions_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        while True:
            i = 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1/locations/sample2/functions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

def test_test_iam_permissions_rest_error():
    if False:
        print('Hello World!')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudFunctionsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.CloudFunctionsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudFunctionsServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.CloudFunctionsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CloudFunctionsServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CloudFunctionsServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.CloudFunctionsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudFunctionsServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.CloudFunctionsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = CloudFunctionsServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudFunctionsServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.CloudFunctionsServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.CloudFunctionsServiceGrpcTransport, transports.CloudFunctionsServiceGrpcAsyncIOTransport, transports.CloudFunctionsServiceRestTransport])
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
        print('Hello World!')
    transport = CloudFunctionsServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.CloudFunctionsServiceGrpcTransport)

def test_cloud_functions_service_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.CloudFunctionsServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_cloud_functions_service_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.functions_v1.services.cloud_functions_service.transports.CloudFunctionsServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.CloudFunctionsServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_functions', 'get_function', 'create_function', 'update_function', 'delete_function', 'call_function', 'generate_upload_url', 'generate_download_url', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'list_locations', 'get_operation', 'list_operations')
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

def test_cloud_functions_service_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.functions_v1.services.cloud_functions_service.transports.CloudFunctionsServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CloudFunctionsServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_cloud_functions_service_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.functions_v1.services.cloud_functions_service.transports.CloudFunctionsServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CloudFunctionsServiceTransport()
        adc.assert_called_once()

def test_cloud_functions_service_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        CloudFunctionsServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.CloudFunctionsServiceGrpcTransport, transports.CloudFunctionsServiceGrpcAsyncIOTransport])
def test_cloud_functions_service_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.CloudFunctionsServiceGrpcTransport, transports.CloudFunctionsServiceGrpcAsyncIOTransport, transports.CloudFunctionsServiceRestTransport])
def test_cloud_functions_service_transport_auth_gdch_credentials(transport_class):
    if False:
        while True:
            i = 10
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.CloudFunctionsServiceGrpcTransport, grpc_helpers), (transports.CloudFunctionsServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_cloud_functions_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('cloudfunctions.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='cloudfunctions.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.CloudFunctionsServiceGrpcTransport, transports.CloudFunctionsServiceGrpcAsyncIOTransport])
def test_cloud_functions_service_grpc_transport_client_cert_source_for_mtls(transport_class):
    if False:
        while True:
            i = 10
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

def test_cloud_functions_service_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.CloudFunctionsServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_cloud_functions_service_rest_lro_client():
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_cloud_functions_service_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudfunctions.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('cloudfunctions.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudfunctions.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_cloud_functions_service_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudfunctions.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('cloudfunctions.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudfunctions.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_cloud_functions_service_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = CloudFunctionsServiceClient(credentials=creds1, transport=transport_name)
    client2 = CloudFunctionsServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_functions._session
    session2 = client2.transport.list_functions._session
    assert session1 != session2
    session1 = client1.transport.get_function._session
    session2 = client2.transport.get_function._session
    assert session1 != session2
    session1 = client1.transport.create_function._session
    session2 = client2.transport.create_function._session
    assert session1 != session2
    session1 = client1.transport.update_function._session
    session2 = client2.transport.update_function._session
    assert session1 != session2
    session1 = client1.transport.delete_function._session
    session2 = client2.transport.delete_function._session
    assert session1 != session2
    session1 = client1.transport.call_function._session
    session2 = client2.transport.call_function._session
    assert session1 != session2
    session1 = client1.transport.generate_upload_url._session
    session2 = client2.transport.generate_upload_url._session
    assert session1 != session2
    session1 = client1.transport.generate_download_url._session
    session2 = client2.transport.generate_download_url._session
    assert session1 != session2
    session1 = client1.transport.set_iam_policy._session
    session2 = client2.transport.set_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.get_iam_policy._session
    session2 = client2.transport.get_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.test_iam_permissions._session
    session2 = client2.transport.test_iam_permissions._session
    assert session1 != session2

def test_cloud_functions_service_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CloudFunctionsServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_cloud_functions_service_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CloudFunctionsServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.CloudFunctionsServiceGrpcTransport, transports.CloudFunctionsServiceGrpcAsyncIOTransport])
def test_cloud_functions_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.CloudFunctionsServiceGrpcTransport, transports.CloudFunctionsServiceGrpcAsyncIOTransport])
def test_cloud_functions_service_transport_channel_mtls_with_adc(transport_class):
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

def test_cloud_functions_service_grpc_lro_client():
    if False:
        while True:
            i = 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_cloud_functions_service_grpc_lro_async_client():
    if False:
        print('Hello World!')
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_cloud_function_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    function = 'whelk'
    expected = 'projects/{project}/locations/{location}/functions/{function}'.format(project=project, location=location, function=function)
    actual = CloudFunctionsServiceClient.cloud_function_path(project, location, function)
    assert expected == actual

def test_parse_cloud_function_path():
    if False:
        print('Hello World!')
    expected = {'project': 'octopus', 'location': 'oyster', 'function': 'nudibranch'}
    path = CloudFunctionsServiceClient.cloud_function_path(**expected)
    actual = CloudFunctionsServiceClient.parse_cloud_function_path(path)
    assert expected == actual

def test_crypto_key_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    location = 'mussel'
    key_ring = 'winkle'
    crypto_key = 'nautilus'
    expected = 'projects/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}'.format(project=project, location=location, key_ring=key_ring, crypto_key=crypto_key)
    actual = CloudFunctionsServiceClient.crypto_key_path(project, location, key_ring, crypto_key)
    assert expected == actual

def test_parse_crypto_key_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'scallop', 'location': 'abalone', 'key_ring': 'squid', 'crypto_key': 'clam'}
    path = CloudFunctionsServiceClient.crypto_key_path(**expected)
    actual = CloudFunctionsServiceClient.parse_crypto_key_path(path)
    assert expected == actual

def test_repository_path():
    if False:
        i = 10
        return i + 15
    project = 'whelk'
    location = 'octopus'
    repository = 'oyster'
    expected = 'projects/{project}/locations/{location}/repositories/{repository}'.format(project=project, location=location, repository=repository)
    actual = CloudFunctionsServiceClient.repository_path(project, location, repository)
    assert expected == actual

def test_parse_repository_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'repository': 'mussel'}
    path = CloudFunctionsServiceClient.repository_path(**expected)
    actual = CloudFunctionsServiceClient.parse_repository_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'winkle'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = CloudFunctionsServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'nautilus'}
    path = CloudFunctionsServiceClient.common_billing_account_path(**expected)
    actual = CloudFunctionsServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'scallop'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = CloudFunctionsServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'abalone'}
    path = CloudFunctionsServiceClient.common_folder_path(**expected)
    actual = CloudFunctionsServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'squid'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = CloudFunctionsServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'clam'}
    path = CloudFunctionsServiceClient.common_organization_path(**expected)
    actual = CloudFunctionsServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'whelk'
    expected = 'projects/{project}'.format(project=project)
    actual = CloudFunctionsServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'octopus'}
    path = CloudFunctionsServiceClient.common_project_path(**expected)
    actual = CloudFunctionsServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'oyster'
    location = 'nudibranch'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = CloudFunctionsServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'cuttlefish', 'location': 'mussel'}
    path = CloudFunctionsServiceClient.common_location_path(**expected)
    actual = CloudFunctionsServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.CloudFunctionsServiceTransport, '_prep_wrapped_messages') as prep:
        client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.CloudFunctionsServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = CloudFunctionsServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_list_locations_rest_bad_request(transport: str='rest', request_type=locations_pb2.ListLocationsRequest):
    if False:
        return 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        while True:
            i = 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'operations/sample1'}, request)
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
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'operations/sample1'}
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
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_operations(request)

@pytest.mark.parametrize('request_type', [operations_pb2.ListOperationsRequest, dict])
def test_list_operations_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
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

def test_get_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = CloudFunctionsServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        return 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = CloudFunctionsServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(CloudFunctionsServiceClient, transports.CloudFunctionsServiceGrpcTransport), (CloudFunctionsServiceAsyncClient, transports.CloudFunctionsServiceGrpcAsyncIOTransport)])
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