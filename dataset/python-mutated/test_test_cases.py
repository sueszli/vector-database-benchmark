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
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import any_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import struct_pb2
from google.protobuf import timestamp_pb2
from google.rpc import status_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.dialogflowcx_v3beta1.services.test_cases import TestCasesAsyncClient, TestCasesClient, pagers, transports
from google.cloud.dialogflowcx_v3beta1.types import advanced_settings, audio_config, data_store_connection, fulfillment, gcs, intent, page, response_message, session
from google.cloud.dialogflowcx_v3beta1.types import test_case
from google.cloud.dialogflowcx_v3beta1.types import test_case as gcdc_test_case

def client_cert_source_callback():
    if False:
        print('Hello World!')
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
    assert TestCasesClient._get_default_mtls_endpoint(None) is None
    assert TestCasesClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert TestCasesClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert TestCasesClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert TestCasesClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert TestCasesClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(TestCasesClient, 'grpc'), (TestCasesAsyncClient, 'grpc_asyncio'), (TestCasesClient, 'rest')])
def test_test_cases_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.TestCasesGrpcTransport, 'grpc'), (transports.TestCasesGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.TestCasesRestTransport, 'rest')])
def test_test_cases_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(TestCasesClient, 'grpc'), (TestCasesAsyncClient, 'grpc_asyncio'), (TestCasesClient, 'rest')])
def test_test_cases_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

def test_test_cases_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = TestCasesClient.get_transport_class()
    available_transports = [transports.TestCasesGrpcTransport, transports.TestCasesRestTransport]
    assert transport in available_transports
    transport = TestCasesClient.get_transport_class('grpc')
    assert transport == transports.TestCasesGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(TestCasesClient, transports.TestCasesGrpcTransport, 'grpc'), (TestCasesAsyncClient, transports.TestCasesGrpcAsyncIOTransport, 'grpc_asyncio'), (TestCasesClient, transports.TestCasesRestTransport, 'rest')])
@mock.patch.object(TestCasesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TestCasesClient))
@mock.patch.object(TestCasesAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TestCasesAsyncClient))
def test_test_cases_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(TestCasesClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(TestCasesClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(TestCasesClient, transports.TestCasesGrpcTransport, 'grpc', 'true'), (TestCasesAsyncClient, transports.TestCasesGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (TestCasesClient, transports.TestCasesGrpcTransport, 'grpc', 'false'), (TestCasesAsyncClient, transports.TestCasesGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (TestCasesClient, transports.TestCasesRestTransport, 'rest', 'true'), (TestCasesClient, transports.TestCasesRestTransport, 'rest', 'false')])
@mock.patch.object(TestCasesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TestCasesClient))
@mock.patch.object(TestCasesAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TestCasesAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_test_cases_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [TestCasesClient, TestCasesAsyncClient])
@mock.patch.object(TestCasesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TestCasesClient))
@mock.patch.object(TestCasesAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TestCasesAsyncClient))
def test_test_cases_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(TestCasesClient, transports.TestCasesGrpcTransport, 'grpc'), (TestCasesAsyncClient, transports.TestCasesGrpcAsyncIOTransport, 'grpc_asyncio'), (TestCasesClient, transports.TestCasesRestTransport, 'rest')])
def test_test_cases_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(TestCasesClient, transports.TestCasesGrpcTransport, 'grpc', grpc_helpers), (TestCasesAsyncClient, transports.TestCasesGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (TestCasesClient, transports.TestCasesRestTransport, 'rest', None)])
def test_test_cases_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_test_cases_client_client_options_from_dict():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.dialogflowcx_v3beta1.services.test_cases.transports.TestCasesGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = TestCasesClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(TestCasesClient, transports.TestCasesGrpcTransport, 'grpc', grpc_helpers), (TestCasesAsyncClient, transports.TestCasesGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_test_cases_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('dialogflow.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), scopes=None, default_host='dialogflow.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [test_case.ListTestCasesRequest, dict])
def test_list_test_cases(request_type, transport: str='grpc'):
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_test_cases), '__call__') as call:
        call.return_value = test_case.ListTestCasesResponse(next_page_token='next_page_token_value')
        response = client.list_test_cases(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.ListTestCasesRequest()
    assert isinstance(response, pagers.ListTestCasesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_test_cases_empty_call():
    if False:
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_test_cases), '__call__') as call:
        client.list_test_cases()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.ListTestCasesRequest()

@pytest.mark.asyncio
async def test_list_test_cases_async(transport: str='grpc_asyncio', request_type=test_case.ListTestCasesRequest):
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_test_cases), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(test_case.ListTestCasesResponse(next_page_token='next_page_token_value'))
        response = await client.list_test_cases(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.ListTestCasesRequest()
    assert isinstance(response, pagers.ListTestCasesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_test_cases_async_from_dict():
    await test_list_test_cases_async(request_type=dict)

def test_list_test_cases_field_headers():
    if False:
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.ListTestCasesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_test_cases), '__call__') as call:
        call.return_value = test_case.ListTestCasesResponse()
        client.list_test_cases(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_test_cases_field_headers_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.ListTestCasesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_test_cases), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(test_case.ListTestCasesResponse())
        await client.list_test_cases(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_test_cases_flattened():
    if False:
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_test_cases), '__call__') as call:
        call.return_value = test_case.ListTestCasesResponse()
        client.list_test_cases(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_test_cases_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_test_cases(test_case.ListTestCasesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_test_cases_flattened_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_test_cases), '__call__') as call:
        call.return_value = test_case.ListTestCasesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(test_case.ListTestCasesResponse())
        response = await client.list_test_cases(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_test_cases_flattened_error_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_test_cases(test_case.ListTestCasesRequest(), parent='parent_value')

def test_list_test_cases_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_test_cases), '__call__') as call:
        call.side_effect = (test_case.ListTestCasesResponse(test_cases=[test_case.TestCase(), test_case.TestCase(), test_case.TestCase()], next_page_token='abc'), test_case.ListTestCasesResponse(test_cases=[], next_page_token='def'), test_case.ListTestCasesResponse(test_cases=[test_case.TestCase()], next_page_token='ghi'), test_case.ListTestCasesResponse(test_cases=[test_case.TestCase(), test_case.TestCase()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_test_cases(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, test_case.TestCase) for i in results))

def test_list_test_cases_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_test_cases), '__call__') as call:
        call.side_effect = (test_case.ListTestCasesResponse(test_cases=[test_case.TestCase(), test_case.TestCase(), test_case.TestCase()], next_page_token='abc'), test_case.ListTestCasesResponse(test_cases=[], next_page_token='def'), test_case.ListTestCasesResponse(test_cases=[test_case.TestCase()], next_page_token='ghi'), test_case.ListTestCasesResponse(test_cases=[test_case.TestCase(), test_case.TestCase()]), RuntimeError)
        pages = list(client.list_test_cases(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_test_cases_async_pager():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_test_cases), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (test_case.ListTestCasesResponse(test_cases=[test_case.TestCase(), test_case.TestCase(), test_case.TestCase()], next_page_token='abc'), test_case.ListTestCasesResponse(test_cases=[], next_page_token='def'), test_case.ListTestCasesResponse(test_cases=[test_case.TestCase()], next_page_token='ghi'), test_case.ListTestCasesResponse(test_cases=[test_case.TestCase(), test_case.TestCase()]), RuntimeError)
        async_pager = await client.list_test_cases(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, test_case.TestCase) for i in responses))

@pytest.mark.asyncio
async def test_list_test_cases_async_pages():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_test_cases), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (test_case.ListTestCasesResponse(test_cases=[test_case.TestCase(), test_case.TestCase(), test_case.TestCase()], next_page_token='abc'), test_case.ListTestCasesResponse(test_cases=[], next_page_token='def'), test_case.ListTestCasesResponse(test_cases=[test_case.TestCase()], next_page_token='ghi'), test_case.ListTestCasesResponse(test_cases=[test_case.TestCase(), test_case.TestCase()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_test_cases(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [test_case.BatchDeleteTestCasesRequest, dict])
def test_batch_delete_test_cases(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_delete_test_cases), '__call__') as call:
        call.return_value = None
        response = client.batch_delete_test_cases(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.BatchDeleteTestCasesRequest()
    assert response is None

def test_batch_delete_test_cases_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_delete_test_cases), '__call__') as call:
        client.batch_delete_test_cases()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.BatchDeleteTestCasesRequest()

@pytest.mark.asyncio
async def test_batch_delete_test_cases_async(transport: str='grpc_asyncio', request_type=test_case.BatchDeleteTestCasesRequest):
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_delete_test_cases), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.batch_delete_test_cases(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.BatchDeleteTestCasesRequest()
    assert response is None

@pytest.mark.asyncio
async def test_batch_delete_test_cases_async_from_dict():
    await test_batch_delete_test_cases_async(request_type=dict)

def test_batch_delete_test_cases_field_headers():
    if False:
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.BatchDeleteTestCasesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_delete_test_cases), '__call__') as call:
        call.return_value = None
        client.batch_delete_test_cases(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_delete_test_cases_field_headers_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.BatchDeleteTestCasesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_delete_test_cases), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.batch_delete_test_cases(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_batch_delete_test_cases_flattened():
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_delete_test_cases), '__call__') as call:
        call.return_value = None
        client.batch_delete_test_cases(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_batch_delete_test_cases_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.batch_delete_test_cases(test_case.BatchDeleteTestCasesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_batch_delete_test_cases_flattened_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_delete_test_cases), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.batch_delete_test_cases(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_batch_delete_test_cases_flattened_error_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.batch_delete_test_cases(test_case.BatchDeleteTestCasesRequest(), parent='parent_value')

@pytest.mark.parametrize('request_type', [test_case.GetTestCaseRequest, dict])
def test_get_test_case(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_test_case), '__call__') as call:
        call.return_value = test_case.TestCase(name='name_value', tags=['tags_value'], display_name='display_name_value', notes='notes_value')
        response = client.get_test_case(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.GetTestCaseRequest()
    assert isinstance(response, test_case.TestCase)
    assert response.name == 'name_value'
    assert response.tags == ['tags_value']
    assert response.display_name == 'display_name_value'
    assert response.notes == 'notes_value'

def test_get_test_case_empty_call():
    if False:
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_test_case), '__call__') as call:
        client.get_test_case()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.GetTestCaseRequest()

@pytest.mark.asyncio
async def test_get_test_case_async(transport: str='grpc_asyncio', request_type=test_case.GetTestCaseRequest):
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_test_case), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(test_case.TestCase(name='name_value', tags=['tags_value'], display_name='display_name_value', notes='notes_value'))
        response = await client.get_test_case(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.GetTestCaseRequest()
    assert isinstance(response, test_case.TestCase)
    assert response.name == 'name_value'
    assert response.tags == ['tags_value']
    assert response.display_name == 'display_name_value'
    assert response.notes == 'notes_value'

@pytest.mark.asyncio
async def test_get_test_case_async_from_dict():
    await test_get_test_case_async(request_type=dict)

def test_get_test_case_field_headers():
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.GetTestCaseRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_test_case), '__call__') as call:
        call.return_value = test_case.TestCase()
        client.get_test_case(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_test_case_field_headers_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.GetTestCaseRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_test_case), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(test_case.TestCase())
        await client.get_test_case(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_test_case_flattened():
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_test_case), '__call__') as call:
        call.return_value = test_case.TestCase()
        client.get_test_case(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_test_case_flattened_error():
    if False:
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_test_case(test_case.GetTestCaseRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_test_case_flattened_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_test_case), '__call__') as call:
        call.return_value = test_case.TestCase()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(test_case.TestCase())
        response = await client.get_test_case(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_test_case_flattened_error_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_test_case(test_case.GetTestCaseRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcdc_test_case.CreateTestCaseRequest, dict])
def test_create_test_case(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_test_case), '__call__') as call:
        call.return_value = gcdc_test_case.TestCase(name='name_value', tags=['tags_value'], display_name='display_name_value', notes='notes_value')
        response = client.create_test_case(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcdc_test_case.CreateTestCaseRequest()
    assert isinstance(response, gcdc_test_case.TestCase)
    assert response.name == 'name_value'
    assert response.tags == ['tags_value']
    assert response.display_name == 'display_name_value'
    assert response.notes == 'notes_value'

def test_create_test_case_empty_call():
    if False:
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_test_case), '__call__') as call:
        client.create_test_case()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcdc_test_case.CreateTestCaseRequest()

@pytest.mark.asyncio
async def test_create_test_case_async(transport: str='grpc_asyncio', request_type=gcdc_test_case.CreateTestCaseRequest):
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_test_case), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcdc_test_case.TestCase(name='name_value', tags=['tags_value'], display_name='display_name_value', notes='notes_value'))
        response = await client.create_test_case(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcdc_test_case.CreateTestCaseRequest()
    assert isinstance(response, gcdc_test_case.TestCase)
    assert response.name == 'name_value'
    assert response.tags == ['tags_value']
    assert response.display_name == 'display_name_value'
    assert response.notes == 'notes_value'

@pytest.mark.asyncio
async def test_create_test_case_async_from_dict():
    await test_create_test_case_async(request_type=dict)

def test_create_test_case_field_headers():
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcdc_test_case.CreateTestCaseRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_test_case), '__call__') as call:
        call.return_value = gcdc_test_case.TestCase()
        client.create_test_case(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_test_case_field_headers_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcdc_test_case.CreateTestCaseRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_test_case), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcdc_test_case.TestCase())
        await client.create_test_case(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_test_case_flattened():
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_test_case), '__call__') as call:
        call.return_value = gcdc_test_case.TestCase()
        client.create_test_case(parent='parent_value', test_case=gcdc_test_case.TestCase(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].test_case
        mock_val = gcdc_test_case.TestCase(name='name_value')
        assert arg == mock_val

def test_create_test_case_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_test_case(gcdc_test_case.CreateTestCaseRequest(), parent='parent_value', test_case=gcdc_test_case.TestCase(name='name_value'))

@pytest.mark.asyncio
async def test_create_test_case_flattened_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_test_case), '__call__') as call:
        call.return_value = gcdc_test_case.TestCase()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcdc_test_case.TestCase())
        response = await client.create_test_case(parent='parent_value', test_case=gcdc_test_case.TestCase(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].test_case
        mock_val = gcdc_test_case.TestCase(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_test_case_flattened_error_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_test_case(gcdc_test_case.CreateTestCaseRequest(), parent='parent_value', test_case=gcdc_test_case.TestCase(name='name_value'))

@pytest.mark.parametrize('request_type', [gcdc_test_case.UpdateTestCaseRequest, dict])
def test_update_test_case(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_test_case), '__call__') as call:
        call.return_value = gcdc_test_case.TestCase(name='name_value', tags=['tags_value'], display_name='display_name_value', notes='notes_value')
        response = client.update_test_case(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcdc_test_case.UpdateTestCaseRequest()
    assert isinstance(response, gcdc_test_case.TestCase)
    assert response.name == 'name_value'
    assert response.tags == ['tags_value']
    assert response.display_name == 'display_name_value'
    assert response.notes == 'notes_value'

def test_update_test_case_empty_call():
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_test_case), '__call__') as call:
        client.update_test_case()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcdc_test_case.UpdateTestCaseRequest()

@pytest.mark.asyncio
async def test_update_test_case_async(transport: str='grpc_asyncio', request_type=gcdc_test_case.UpdateTestCaseRequest):
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_test_case), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcdc_test_case.TestCase(name='name_value', tags=['tags_value'], display_name='display_name_value', notes='notes_value'))
        response = await client.update_test_case(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcdc_test_case.UpdateTestCaseRequest()
    assert isinstance(response, gcdc_test_case.TestCase)
    assert response.name == 'name_value'
    assert response.tags == ['tags_value']
    assert response.display_name == 'display_name_value'
    assert response.notes == 'notes_value'

@pytest.mark.asyncio
async def test_update_test_case_async_from_dict():
    await test_update_test_case_async(request_type=dict)

def test_update_test_case_field_headers():
    if False:
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcdc_test_case.UpdateTestCaseRequest()
    request.test_case.name = 'name_value'
    with mock.patch.object(type(client.transport.update_test_case), '__call__') as call:
        call.return_value = gcdc_test_case.TestCase()
        client.update_test_case(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'test_case.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_test_case_field_headers_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcdc_test_case.UpdateTestCaseRequest()
    request.test_case.name = 'name_value'
    with mock.patch.object(type(client.transport.update_test_case), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcdc_test_case.TestCase())
        await client.update_test_case(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'test_case.name=name_value') in kw['metadata']

def test_update_test_case_flattened():
    if False:
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_test_case), '__call__') as call:
        call.return_value = gcdc_test_case.TestCase()
        client.update_test_case(test_case=gcdc_test_case.TestCase(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].test_case
        mock_val = gcdc_test_case.TestCase(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_test_case_flattened_error():
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_test_case(gcdc_test_case.UpdateTestCaseRequest(), test_case=gcdc_test_case.TestCase(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_test_case_flattened_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_test_case), '__call__') as call:
        call.return_value = gcdc_test_case.TestCase()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcdc_test_case.TestCase())
        response = await client.update_test_case(test_case=gcdc_test_case.TestCase(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].test_case
        mock_val = gcdc_test_case.TestCase(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_test_case_flattened_error_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_test_case(gcdc_test_case.UpdateTestCaseRequest(), test_case=gcdc_test_case.TestCase(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [test_case.RunTestCaseRequest, dict])
def test_run_test_case(request_type, transport: str='grpc'):
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.run_test_case), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.run_test_case(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.RunTestCaseRequest()
    assert isinstance(response, future.Future)

def test_run_test_case_empty_call():
    if False:
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.run_test_case), '__call__') as call:
        client.run_test_case()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.RunTestCaseRequest()

@pytest.mark.asyncio
async def test_run_test_case_async(transport: str='grpc_asyncio', request_type=test_case.RunTestCaseRequest):
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.run_test_case), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.run_test_case(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.RunTestCaseRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_run_test_case_async_from_dict():
    await test_run_test_case_async(request_type=dict)

def test_run_test_case_field_headers():
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.RunTestCaseRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.run_test_case), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.run_test_case(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_run_test_case_field_headers_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.RunTestCaseRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.run_test_case), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.run_test_case(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [test_case.BatchRunTestCasesRequest, dict])
def test_batch_run_test_cases(request_type, transport: str='grpc'):
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_run_test_cases), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.batch_run_test_cases(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.BatchRunTestCasesRequest()
    assert isinstance(response, future.Future)

def test_batch_run_test_cases_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_run_test_cases), '__call__') as call:
        client.batch_run_test_cases()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.BatchRunTestCasesRequest()

@pytest.mark.asyncio
async def test_batch_run_test_cases_async(transport: str='grpc_asyncio', request_type=test_case.BatchRunTestCasesRequest):
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_run_test_cases), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_run_test_cases(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.BatchRunTestCasesRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_batch_run_test_cases_async_from_dict():
    await test_batch_run_test_cases_async(request_type=dict)

def test_batch_run_test_cases_field_headers():
    if False:
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.BatchRunTestCasesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_run_test_cases), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_run_test_cases(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_run_test_cases_field_headers_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.BatchRunTestCasesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_run_test_cases), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.batch_run_test_cases(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [test_case.CalculateCoverageRequest, dict])
def test_calculate_coverage(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.calculate_coverage), '__call__') as call:
        call.return_value = test_case.CalculateCoverageResponse(agent='agent_value')
        response = client.calculate_coverage(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.CalculateCoverageRequest()
    assert isinstance(response, test_case.CalculateCoverageResponse)
    assert response.agent == 'agent_value'

def test_calculate_coverage_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.calculate_coverage), '__call__') as call:
        client.calculate_coverage()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.CalculateCoverageRequest()

@pytest.mark.asyncio
async def test_calculate_coverage_async(transport: str='grpc_asyncio', request_type=test_case.CalculateCoverageRequest):
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.calculate_coverage), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(test_case.CalculateCoverageResponse(agent='agent_value'))
        response = await client.calculate_coverage(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.CalculateCoverageRequest()
    assert isinstance(response, test_case.CalculateCoverageResponse)
    assert response.agent == 'agent_value'

@pytest.mark.asyncio
async def test_calculate_coverage_async_from_dict():
    await test_calculate_coverage_async(request_type=dict)

def test_calculate_coverage_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.CalculateCoverageRequest()
    request.agent = 'agent_value'
    with mock.patch.object(type(client.transport.calculate_coverage), '__call__') as call:
        call.return_value = test_case.CalculateCoverageResponse()
        client.calculate_coverage(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'agent=agent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_calculate_coverage_field_headers_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.CalculateCoverageRequest()
    request.agent = 'agent_value'
    with mock.patch.object(type(client.transport.calculate_coverage), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(test_case.CalculateCoverageResponse())
        await client.calculate_coverage(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'agent=agent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [test_case.ImportTestCasesRequest, dict])
def test_import_test_cases(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.import_test_cases), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.import_test_cases(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.ImportTestCasesRequest()
    assert isinstance(response, future.Future)

def test_import_test_cases_empty_call():
    if False:
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.import_test_cases), '__call__') as call:
        client.import_test_cases()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.ImportTestCasesRequest()

@pytest.mark.asyncio
async def test_import_test_cases_async(transport: str='grpc_asyncio', request_type=test_case.ImportTestCasesRequest):
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.import_test_cases), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.import_test_cases(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.ImportTestCasesRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_import_test_cases_async_from_dict():
    await test_import_test_cases_async(request_type=dict)

def test_import_test_cases_field_headers():
    if False:
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.ImportTestCasesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.import_test_cases), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.import_test_cases(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_import_test_cases_field_headers_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.ImportTestCasesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.import_test_cases), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.import_test_cases(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [test_case.ExportTestCasesRequest, dict])
def test_export_test_cases(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.export_test_cases), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.export_test_cases(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.ExportTestCasesRequest()
    assert isinstance(response, future.Future)

def test_export_test_cases_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.export_test_cases), '__call__') as call:
        client.export_test_cases()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.ExportTestCasesRequest()

@pytest.mark.asyncio
async def test_export_test_cases_async(transport: str='grpc_asyncio', request_type=test_case.ExportTestCasesRequest):
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.export_test_cases), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.export_test_cases(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.ExportTestCasesRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_export_test_cases_async_from_dict():
    await test_export_test_cases_async(request_type=dict)

def test_export_test_cases_field_headers():
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.ExportTestCasesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.export_test_cases), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.export_test_cases(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_export_test_cases_field_headers_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.ExportTestCasesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.export_test_cases), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.export_test_cases(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [test_case.ListTestCaseResultsRequest, dict])
def test_list_test_case_results(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_test_case_results), '__call__') as call:
        call.return_value = test_case.ListTestCaseResultsResponse(next_page_token='next_page_token_value')
        response = client.list_test_case_results(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.ListTestCaseResultsRequest()
    assert isinstance(response, pagers.ListTestCaseResultsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_test_case_results_empty_call():
    if False:
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_test_case_results), '__call__') as call:
        client.list_test_case_results()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.ListTestCaseResultsRequest()

@pytest.mark.asyncio
async def test_list_test_case_results_async(transport: str='grpc_asyncio', request_type=test_case.ListTestCaseResultsRequest):
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_test_case_results), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(test_case.ListTestCaseResultsResponse(next_page_token='next_page_token_value'))
        response = await client.list_test_case_results(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.ListTestCaseResultsRequest()
    assert isinstance(response, pagers.ListTestCaseResultsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_test_case_results_async_from_dict():
    await test_list_test_case_results_async(request_type=dict)

def test_list_test_case_results_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.ListTestCaseResultsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_test_case_results), '__call__') as call:
        call.return_value = test_case.ListTestCaseResultsResponse()
        client.list_test_case_results(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_test_case_results_field_headers_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.ListTestCaseResultsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_test_case_results), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(test_case.ListTestCaseResultsResponse())
        await client.list_test_case_results(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_test_case_results_flattened():
    if False:
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_test_case_results), '__call__') as call:
        call.return_value = test_case.ListTestCaseResultsResponse()
        client.list_test_case_results(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_test_case_results_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_test_case_results(test_case.ListTestCaseResultsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_test_case_results_flattened_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_test_case_results), '__call__') as call:
        call.return_value = test_case.ListTestCaseResultsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(test_case.ListTestCaseResultsResponse())
        response = await client.list_test_case_results(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_test_case_results_flattened_error_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_test_case_results(test_case.ListTestCaseResultsRequest(), parent='parent_value')

def test_list_test_case_results_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_test_case_results), '__call__') as call:
        call.side_effect = (test_case.ListTestCaseResultsResponse(test_case_results=[test_case.TestCaseResult(), test_case.TestCaseResult(), test_case.TestCaseResult()], next_page_token='abc'), test_case.ListTestCaseResultsResponse(test_case_results=[], next_page_token='def'), test_case.ListTestCaseResultsResponse(test_case_results=[test_case.TestCaseResult()], next_page_token='ghi'), test_case.ListTestCaseResultsResponse(test_case_results=[test_case.TestCaseResult(), test_case.TestCaseResult()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_test_case_results(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, test_case.TestCaseResult) for i in results))

def test_list_test_case_results_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_test_case_results), '__call__') as call:
        call.side_effect = (test_case.ListTestCaseResultsResponse(test_case_results=[test_case.TestCaseResult(), test_case.TestCaseResult(), test_case.TestCaseResult()], next_page_token='abc'), test_case.ListTestCaseResultsResponse(test_case_results=[], next_page_token='def'), test_case.ListTestCaseResultsResponse(test_case_results=[test_case.TestCaseResult()], next_page_token='ghi'), test_case.ListTestCaseResultsResponse(test_case_results=[test_case.TestCaseResult(), test_case.TestCaseResult()]), RuntimeError)
        pages = list(client.list_test_case_results(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_test_case_results_async_pager():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_test_case_results), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (test_case.ListTestCaseResultsResponse(test_case_results=[test_case.TestCaseResult(), test_case.TestCaseResult(), test_case.TestCaseResult()], next_page_token='abc'), test_case.ListTestCaseResultsResponse(test_case_results=[], next_page_token='def'), test_case.ListTestCaseResultsResponse(test_case_results=[test_case.TestCaseResult()], next_page_token='ghi'), test_case.ListTestCaseResultsResponse(test_case_results=[test_case.TestCaseResult(), test_case.TestCaseResult()]), RuntimeError)
        async_pager = await client.list_test_case_results(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, test_case.TestCaseResult) for i in responses))

@pytest.mark.asyncio
async def test_list_test_case_results_async_pages():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_test_case_results), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (test_case.ListTestCaseResultsResponse(test_case_results=[test_case.TestCaseResult(), test_case.TestCaseResult(), test_case.TestCaseResult()], next_page_token='abc'), test_case.ListTestCaseResultsResponse(test_case_results=[], next_page_token='def'), test_case.ListTestCaseResultsResponse(test_case_results=[test_case.TestCaseResult()], next_page_token='ghi'), test_case.ListTestCaseResultsResponse(test_case_results=[test_case.TestCaseResult(), test_case.TestCaseResult()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_test_case_results(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [test_case.GetTestCaseResultRequest, dict])
def test_get_test_case_result(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_test_case_result), '__call__') as call:
        call.return_value = test_case.TestCaseResult(name='name_value', environment='environment_value', test_result=test_case.TestResult.PASSED)
        response = client.get_test_case_result(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.GetTestCaseResultRequest()
    assert isinstance(response, test_case.TestCaseResult)
    assert response.name == 'name_value'
    assert response.environment == 'environment_value'
    assert response.test_result == test_case.TestResult.PASSED

def test_get_test_case_result_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_test_case_result), '__call__') as call:
        client.get_test_case_result()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.GetTestCaseResultRequest()

@pytest.mark.asyncio
async def test_get_test_case_result_async(transport: str='grpc_asyncio', request_type=test_case.GetTestCaseResultRequest):
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_test_case_result), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(test_case.TestCaseResult(name='name_value', environment='environment_value', test_result=test_case.TestResult.PASSED))
        response = await client.get_test_case_result(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == test_case.GetTestCaseResultRequest()
    assert isinstance(response, test_case.TestCaseResult)
    assert response.name == 'name_value'
    assert response.environment == 'environment_value'
    assert response.test_result == test_case.TestResult.PASSED

@pytest.mark.asyncio
async def test_get_test_case_result_async_from_dict():
    await test_get_test_case_result_async(request_type=dict)

def test_get_test_case_result_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.GetTestCaseResultRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_test_case_result), '__call__') as call:
        call.return_value = test_case.TestCaseResult()
        client.get_test_case_result(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_test_case_result_field_headers_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = test_case.GetTestCaseResultRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_test_case_result), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(test_case.TestCaseResult())
        await client.get_test_case_result(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_test_case_result_flattened():
    if False:
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_test_case_result), '__call__') as call:
        call.return_value = test_case.TestCaseResult()
        client.get_test_case_result(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_test_case_result_flattened_error():
    if False:
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_test_case_result(test_case.GetTestCaseResultRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_test_case_result_flattened_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_test_case_result), '__call__') as call:
        call.return_value = test_case.TestCaseResult()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(test_case.TestCaseResult())
        response = await client.get_test_case_result(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_test_case_result_flattened_error_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_test_case_result(test_case.GetTestCaseResultRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [test_case.ListTestCasesRequest, dict])
def test_list_test_cases_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/agents/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = test_case.ListTestCasesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = test_case.ListTestCasesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_test_cases(request)
    assert isinstance(response, pagers.ListTestCasesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_test_cases_rest_required_fields(request_type=test_case.ListTestCasesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.TestCasesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_test_cases._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_test_cases._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token', 'view'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = test_case.ListTestCasesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = test_case.ListTestCasesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_test_cases(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_test_cases_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_test_cases._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken', 'view')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_test_cases_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TestCasesRestInterceptor())
    client = TestCasesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TestCasesRestInterceptor, 'post_list_test_cases') as post, mock.patch.object(transports.TestCasesRestInterceptor, 'pre_list_test_cases') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = test_case.ListTestCasesRequest.pb(test_case.ListTestCasesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = test_case.ListTestCasesResponse.to_json(test_case.ListTestCasesResponse())
        request = test_case.ListTestCasesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = test_case.ListTestCasesResponse()
        client.list_test_cases(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_test_cases_rest_bad_request(transport: str='rest', request_type=test_case.ListTestCasesRequest):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/agents/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_test_cases(request)

def test_list_test_cases_rest_flattened():
    if False:
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = test_case.ListTestCasesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/agents/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = test_case.ListTestCasesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_test_cases(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{parent=projects/*/locations/*/agents/*}/testCases' % client.transport._host, args[1])

def test_list_test_cases_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_test_cases(test_case.ListTestCasesRequest(), parent='parent_value')

def test_list_test_cases_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (test_case.ListTestCasesResponse(test_cases=[test_case.TestCase(), test_case.TestCase(), test_case.TestCase()], next_page_token='abc'), test_case.ListTestCasesResponse(test_cases=[], next_page_token='def'), test_case.ListTestCasesResponse(test_cases=[test_case.TestCase()], next_page_token='ghi'), test_case.ListTestCasesResponse(test_cases=[test_case.TestCase(), test_case.TestCase()]))
        response = response + response
        response = tuple((test_case.ListTestCasesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/agents/sample3'}
        pager = client.list_test_cases(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, test_case.TestCase) for i in results))
        pages = list(client.list_test_cases(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [test_case.BatchDeleteTestCasesRequest, dict])
def test_batch_delete_test_cases_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/agents/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_delete_test_cases(request)
    assert response is None

def test_batch_delete_test_cases_rest_required_fields(request_type=test_case.BatchDeleteTestCasesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.TestCasesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['names'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_delete_test_cases._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['names'] = 'names_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_delete_test_cases._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'names' in jsonified_request
    assert jsonified_request['names'] == 'names_value'
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.batch_delete_test_cases(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_delete_test_cases_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_delete_test_cases._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'names'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_delete_test_cases_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TestCasesRestInterceptor())
    client = TestCasesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TestCasesRestInterceptor, 'pre_batch_delete_test_cases') as pre:
        pre.assert_not_called()
        pb_message = test_case.BatchDeleteTestCasesRequest.pb(test_case.BatchDeleteTestCasesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = test_case.BatchDeleteTestCasesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.batch_delete_test_cases(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_batch_delete_test_cases_rest_bad_request(transport: str='rest', request_type=test_case.BatchDeleteTestCasesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/agents/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_delete_test_cases(request)

def test_batch_delete_test_cases_rest_flattened():
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'parent': 'projects/sample1/locations/sample2/agents/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.batch_delete_test_cases(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{parent=projects/*/locations/*/agents/*}/testCases:batchDelete' % client.transport._host, args[1])

def test_batch_delete_test_cases_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.batch_delete_test_cases(test_case.BatchDeleteTestCasesRequest(), parent='parent_value')

def test_batch_delete_test_cases_rest_error():
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [test_case.GetTestCaseRequest, dict])
def test_get_test_case_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/agents/sample3/testCases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = test_case.TestCase(name='name_value', tags=['tags_value'], display_name='display_name_value', notes='notes_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = test_case.TestCase.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_test_case(request)
    assert isinstance(response, test_case.TestCase)
    assert response.name == 'name_value'
    assert response.tags == ['tags_value']
    assert response.display_name == 'display_name_value'
    assert response.notes == 'notes_value'

def test_get_test_case_rest_required_fields(request_type=test_case.GetTestCaseRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.TestCasesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_test_case._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_test_case._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = test_case.TestCase()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = test_case.TestCase.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_test_case(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_test_case_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_test_case._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_test_case_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TestCasesRestInterceptor())
    client = TestCasesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TestCasesRestInterceptor, 'post_get_test_case') as post, mock.patch.object(transports.TestCasesRestInterceptor, 'pre_get_test_case') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = test_case.GetTestCaseRequest.pb(test_case.GetTestCaseRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = test_case.TestCase.to_json(test_case.TestCase())
        request = test_case.GetTestCaseRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = test_case.TestCase()
        client.get_test_case(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_test_case_rest_bad_request(transport: str='rest', request_type=test_case.GetTestCaseRequest):
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/agents/sample3/testCases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_test_case(request)

def test_get_test_case_rest_flattened():
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = test_case.TestCase()
        sample_request = {'name': 'projects/sample1/locations/sample2/agents/sample3/testCases/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = test_case.TestCase.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_test_case(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{name=projects/*/locations/*/agents/*/testCases/*}' % client.transport._host, args[1])

def test_get_test_case_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_test_case(test_case.GetTestCaseRequest(), name='name_value')

def test_get_test_case_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcdc_test_case.CreateTestCaseRequest, dict])
def test_create_test_case_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/agents/sample3'}
    request_init['test_case'] = {'name': 'name_value', 'tags': ['tags_value1', 'tags_value2'], 'display_name': 'display_name_value', 'notes': 'notes_value', 'test_config': {'tracking_parameters': ['tracking_parameters_value1', 'tracking_parameters_value2'], 'flow': 'flow_value', 'page': 'page_value'}, 'test_case_conversation_turns': [{'user_input': {'input': {'text': {'text': 'text_value'}, 'intent': {'intent': 'intent_value'}, 'audio': {'config': {'audio_encoding': 1, 'sample_rate_hertz': 1817, 'enable_word_info': True, 'phrase_hints': ['phrase_hints_value1', 'phrase_hints_value2'], 'model': 'model_value', 'model_variant': 1, 'single_utterance': True}, 'audio': b'audio_blob'}, 'event': {'event': 'event_value'}, 'dtmf': {'digits': 'digits_value', 'finish_digit': 'finish_digit_value'}, 'language_code': 'language_code_value'}, 'injected_parameters': {'fields': {}}, 'is_webhook_enabled': True, 'enable_sentiment_analysis': True}, 'virtual_agent_output': {'session_parameters': {}, 'differences': [{'type_': 1, 'description': 'description_value'}], 'diagnostic_info': {}, 'triggered_intent': {'name': 'name_value', 'display_name': 'display_name_value', 'training_phrases': [{'id': 'id_value', 'parts': [{'text': 'text_value', 'parameter_id': 'parameter_id_value'}], 'repeat_count': 1289}], 'parameters': [{'id': 'id_value', 'entity_type': 'entity_type_value', 'is_list': True, 'redact': True}], 'priority': 898, 'is_fallback': True, 'labels': {}, 'description': 'description_value'}, 'current_page': {'name': 'name_value', 'display_name': 'display_name_value', 'entry_fulfillment': {'messages': [{'text': {'text': ['text_value1', 'text_value2'], 'allow_playback_interruption': True}, 'payload': {}, 'conversation_success': {'metadata': {}}, 'output_audio_text': {'text': 'text_value', 'ssml': 'ssml_value', 'allow_playback_interruption': True}, 'live_agent_handoff': {'metadata': {}}, 'end_interaction': {}, 'play_audio': {'audio_uri': 'audio_uri_value', 'allow_playback_interruption': True}, 'mixed_audio': {'segments': [{'audio': b'audio_blob', 'uri': 'uri_value', 'allow_playback_interruption': True}]}, 'telephony_transfer_call': {'phone_number': 'phone_number_value'}, 'knowledge_info_card': {}, 'channel': 'channel_value'}], 'webhook': 'webhook_value', 'return_partial_responses': True, 'tag': 'tag_value', 'set_parameter_actions': [{'parameter': 'parameter_value', 'value': {'null_value': 0, 'number_value': 0.1285, 'string_value': 'string_value_value', 'bool_value': True, 'struct_value': {}, 'list_value': {'values': {}}}}], 'conditional_cases': [{'cases': [{'condition': 'condition_value', 'case_content': [{'message': {}, 'additional_cases': {}}]}]}], 'advanced_settings': {'audio_export_gcs_destination': {'uri': 'uri_value'}, 'dtmf_settings': {'enabled': True, 'max_digits': 1065, 'finish_digit': 'finish_digit_value'}, 'logging_settings': {'enable_stackdriver_logging': True, 'enable_interaction_logging': True}}, 'enable_generative_fallback': True}, 'form': {'parameters': [{'display_name': 'display_name_value', 'required': True, 'entity_type': 'entity_type_value', 'is_list': True, 'fill_behavior': {'initial_prompt_fulfillment': {}, 'reprompt_event_handlers': [{'name': 'name_value', 'event': 'event_value', 'trigger_fulfillment': {}, 'target_page': 'target_page_value', 'target_flow': 'target_flow_value'}]}, 'default_value': {}, 'redact': True, 'advanced_settings': {}}]}, 'transition_route_groups': ['transition_route_groups_value1', 'transition_route_groups_value2'], 'transition_routes': [{'name': 'name_value', 'description': 'description_value', 'intent': 'intent_value', 'condition': 'condition_value', 'trigger_fulfillment': {}, 'target_page': 'target_page_value', 'target_flow': 'target_flow_value'}], 'event_handlers': {}, 'advanced_settings': {}, 'knowledge_connector_settings': {'enabled': True, 'trigger_fulfillment': {}, 'target_page': 'target_page_value', 'target_flow': 'target_flow_value', 'data_store_connections': [{'data_store_type': 1, 'data_store': 'data_store_value'}]}}, 'text_responses': {}, 'status': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}}}], 'creation_time': {'seconds': 751, 'nanos': 543}, 'last_test_result': {'name': 'name_value', 'environment': 'environment_value', 'conversation_turns': {}, 'test_result': 1, 'test_time': {}}}
    test_field = gcdc_test_case.CreateTestCaseRequest.meta.fields['test_case']

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
    for (field, value) in request_init['test_case'].items():
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
                for i in range(0, len(request_init['test_case'][field])):
                    del request_init['test_case'][field][i][subfield]
            else:
                del request_init['test_case'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcdc_test_case.TestCase(name='name_value', tags=['tags_value'], display_name='display_name_value', notes='notes_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcdc_test_case.TestCase.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_test_case(request)
    assert isinstance(response, gcdc_test_case.TestCase)
    assert response.name == 'name_value'
    assert response.tags == ['tags_value']
    assert response.display_name == 'display_name_value'
    assert response.notes == 'notes_value'

def test_create_test_case_rest_required_fields(request_type=gcdc_test_case.CreateTestCaseRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.TestCasesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_test_case._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_test_case._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcdc_test_case.TestCase()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcdc_test_case.TestCase.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_test_case(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_test_case_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_test_case._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'testCase'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_test_case_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TestCasesRestInterceptor())
    client = TestCasesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TestCasesRestInterceptor, 'post_create_test_case') as post, mock.patch.object(transports.TestCasesRestInterceptor, 'pre_create_test_case') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcdc_test_case.CreateTestCaseRequest.pb(gcdc_test_case.CreateTestCaseRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcdc_test_case.TestCase.to_json(gcdc_test_case.TestCase())
        request = gcdc_test_case.CreateTestCaseRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcdc_test_case.TestCase()
        client.create_test_case(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_test_case_rest_bad_request(transport: str='rest', request_type=gcdc_test_case.CreateTestCaseRequest):
    if False:
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/agents/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_test_case(request)

def test_create_test_case_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcdc_test_case.TestCase()
        sample_request = {'parent': 'projects/sample1/locations/sample2/agents/sample3'}
        mock_args = dict(parent='parent_value', test_case=gcdc_test_case.TestCase(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcdc_test_case.TestCase.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_test_case(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{parent=projects/*/locations/*/agents/*}/testCases' % client.transport._host, args[1])

def test_create_test_case_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_test_case(gcdc_test_case.CreateTestCaseRequest(), parent='parent_value', test_case=gcdc_test_case.TestCase(name='name_value'))

def test_create_test_case_rest_error():
    if False:
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcdc_test_case.UpdateTestCaseRequest, dict])
def test_update_test_case_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'test_case': {'name': 'projects/sample1/locations/sample2/agents/sample3/testCases/sample4'}}
    request_init['test_case'] = {'name': 'projects/sample1/locations/sample2/agents/sample3/testCases/sample4', 'tags': ['tags_value1', 'tags_value2'], 'display_name': 'display_name_value', 'notes': 'notes_value', 'test_config': {'tracking_parameters': ['tracking_parameters_value1', 'tracking_parameters_value2'], 'flow': 'flow_value', 'page': 'page_value'}, 'test_case_conversation_turns': [{'user_input': {'input': {'text': {'text': 'text_value'}, 'intent': {'intent': 'intent_value'}, 'audio': {'config': {'audio_encoding': 1, 'sample_rate_hertz': 1817, 'enable_word_info': True, 'phrase_hints': ['phrase_hints_value1', 'phrase_hints_value2'], 'model': 'model_value', 'model_variant': 1, 'single_utterance': True}, 'audio': b'audio_blob'}, 'event': {'event': 'event_value'}, 'dtmf': {'digits': 'digits_value', 'finish_digit': 'finish_digit_value'}, 'language_code': 'language_code_value'}, 'injected_parameters': {'fields': {}}, 'is_webhook_enabled': True, 'enable_sentiment_analysis': True}, 'virtual_agent_output': {'session_parameters': {}, 'differences': [{'type_': 1, 'description': 'description_value'}], 'diagnostic_info': {}, 'triggered_intent': {'name': 'name_value', 'display_name': 'display_name_value', 'training_phrases': [{'id': 'id_value', 'parts': [{'text': 'text_value', 'parameter_id': 'parameter_id_value'}], 'repeat_count': 1289}], 'parameters': [{'id': 'id_value', 'entity_type': 'entity_type_value', 'is_list': True, 'redact': True}], 'priority': 898, 'is_fallback': True, 'labels': {}, 'description': 'description_value'}, 'current_page': {'name': 'name_value', 'display_name': 'display_name_value', 'entry_fulfillment': {'messages': [{'text': {'text': ['text_value1', 'text_value2'], 'allow_playback_interruption': True}, 'payload': {}, 'conversation_success': {'metadata': {}}, 'output_audio_text': {'text': 'text_value', 'ssml': 'ssml_value', 'allow_playback_interruption': True}, 'live_agent_handoff': {'metadata': {}}, 'end_interaction': {}, 'play_audio': {'audio_uri': 'audio_uri_value', 'allow_playback_interruption': True}, 'mixed_audio': {'segments': [{'audio': b'audio_blob', 'uri': 'uri_value', 'allow_playback_interruption': True}]}, 'telephony_transfer_call': {'phone_number': 'phone_number_value'}, 'knowledge_info_card': {}, 'channel': 'channel_value'}], 'webhook': 'webhook_value', 'return_partial_responses': True, 'tag': 'tag_value', 'set_parameter_actions': [{'parameter': 'parameter_value', 'value': {'null_value': 0, 'number_value': 0.1285, 'string_value': 'string_value_value', 'bool_value': True, 'struct_value': {}, 'list_value': {'values': {}}}}], 'conditional_cases': [{'cases': [{'condition': 'condition_value', 'case_content': [{'message': {}, 'additional_cases': {}}]}]}], 'advanced_settings': {'audio_export_gcs_destination': {'uri': 'uri_value'}, 'dtmf_settings': {'enabled': True, 'max_digits': 1065, 'finish_digit': 'finish_digit_value'}, 'logging_settings': {'enable_stackdriver_logging': True, 'enable_interaction_logging': True}}, 'enable_generative_fallback': True}, 'form': {'parameters': [{'display_name': 'display_name_value', 'required': True, 'entity_type': 'entity_type_value', 'is_list': True, 'fill_behavior': {'initial_prompt_fulfillment': {}, 'reprompt_event_handlers': [{'name': 'name_value', 'event': 'event_value', 'trigger_fulfillment': {}, 'target_page': 'target_page_value', 'target_flow': 'target_flow_value'}]}, 'default_value': {}, 'redact': True, 'advanced_settings': {}}]}, 'transition_route_groups': ['transition_route_groups_value1', 'transition_route_groups_value2'], 'transition_routes': [{'name': 'name_value', 'description': 'description_value', 'intent': 'intent_value', 'condition': 'condition_value', 'trigger_fulfillment': {}, 'target_page': 'target_page_value', 'target_flow': 'target_flow_value'}], 'event_handlers': {}, 'advanced_settings': {}, 'knowledge_connector_settings': {'enabled': True, 'trigger_fulfillment': {}, 'target_page': 'target_page_value', 'target_flow': 'target_flow_value', 'data_store_connections': [{'data_store_type': 1, 'data_store': 'data_store_value'}]}}, 'text_responses': {}, 'status': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}}}], 'creation_time': {'seconds': 751, 'nanos': 543}, 'last_test_result': {'name': 'name_value', 'environment': 'environment_value', 'conversation_turns': {}, 'test_result': 1, 'test_time': {}}}
    test_field = gcdc_test_case.UpdateTestCaseRequest.meta.fields['test_case']

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
    for (field, value) in request_init['test_case'].items():
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
                for i in range(0, len(request_init['test_case'][field])):
                    del request_init['test_case'][field][i][subfield]
            else:
                del request_init['test_case'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcdc_test_case.TestCase(name='name_value', tags=['tags_value'], display_name='display_name_value', notes='notes_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcdc_test_case.TestCase.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_test_case(request)
    assert isinstance(response, gcdc_test_case.TestCase)
    assert response.name == 'name_value'
    assert response.tags == ['tags_value']
    assert response.display_name == 'display_name_value'
    assert response.notes == 'notes_value'

def test_update_test_case_rest_required_fields(request_type=gcdc_test_case.UpdateTestCaseRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.TestCasesRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_test_case._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_test_case._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcdc_test_case.TestCase()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcdc_test_case.TestCase.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_test_case(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_test_case_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_test_case._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('testCase', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_test_case_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TestCasesRestInterceptor())
    client = TestCasesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TestCasesRestInterceptor, 'post_update_test_case') as post, mock.patch.object(transports.TestCasesRestInterceptor, 'pre_update_test_case') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcdc_test_case.UpdateTestCaseRequest.pb(gcdc_test_case.UpdateTestCaseRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcdc_test_case.TestCase.to_json(gcdc_test_case.TestCase())
        request = gcdc_test_case.UpdateTestCaseRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcdc_test_case.TestCase()
        client.update_test_case(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_test_case_rest_bad_request(transport: str='rest', request_type=gcdc_test_case.UpdateTestCaseRequest):
    if False:
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'test_case': {'name': 'projects/sample1/locations/sample2/agents/sample3/testCases/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_test_case(request)

def test_update_test_case_rest_flattened():
    if False:
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcdc_test_case.TestCase()
        sample_request = {'test_case': {'name': 'projects/sample1/locations/sample2/agents/sample3/testCases/sample4'}}
        mock_args = dict(test_case=gcdc_test_case.TestCase(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcdc_test_case.TestCase.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_test_case(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{test_case.name=projects/*/locations/*/agents/*/testCases/*}' % client.transport._host, args[1])

def test_update_test_case_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_test_case(gcdc_test_case.UpdateTestCaseRequest(), test_case=gcdc_test_case.TestCase(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_test_case_rest_error():
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [test_case.RunTestCaseRequest, dict])
def test_run_test_case_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/agents/sample3/testCases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.run_test_case(request)
    assert response.operation.name == 'operations/spam'

def test_run_test_case_rest_required_fields(request_type=test_case.RunTestCaseRequest):
    if False:
        return 10
    transport_class = transports.TestCasesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).run_test_case._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).run_test_case._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.run_test_case(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_run_test_case_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.run_test_case._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_run_test_case_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TestCasesRestInterceptor())
    client = TestCasesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.TestCasesRestInterceptor, 'post_run_test_case') as post, mock.patch.object(transports.TestCasesRestInterceptor, 'pre_run_test_case') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = test_case.RunTestCaseRequest.pb(test_case.RunTestCaseRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = test_case.RunTestCaseRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.run_test_case(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_run_test_case_rest_bad_request(transport: str='rest', request_type=test_case.RunTestCaseRequest):
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/agents/sample3/testCases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.run_test_case(request)

def test_run_test_case_rest_error():
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [test_case.BatchRunTestCasesRequest, dict])
def test_batch_run_test_cases_rest(request_type):
    if False:
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/agents/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_run_test_cases(request)
    assert response.operation.name == 'operations/spam'

def test_batch_run_test_cases_rest_required_fields(request_type=test_case.BatchRunTestCasesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.TestCasesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['test_cases'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_run_test_cases._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['testCases'] = 'test_cases_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_run_test_cases._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'testCases' in jsonified_request
    assert jsonified_request['testCases'] == 'test_cases_value'
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.batch_run_test_cases(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_run_test_cases_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_run_test_cases._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'testCases'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_run_test_cases_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TestCasesRestInterceptor())
    client = TestCasesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.TestCasesRestInterceptor, 'post_batch_run_test_cases') as post, mock.patch.object(transports.TestCasesRestInterceptor, 'pre_batch_run_test_cases') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = test_case.BatchRunTestCasesRequest.pb(test_case.BatchRunTestCasesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = test_case.BatchRunTestCasesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.batch_run_test_cases(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_run_test_cases_rest_bad_request(transport: str='rest', request_type=test_case.BatchRunTestCasesRequest):
    if False:
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/agents/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_run_test_cases(request)

def test_batch_run_test_cases_rest_error():
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [test_case.CalculateCoverageRequest, dict])
def test_calculate_coverage_rest(request_type):
    if False:
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'agent': 'projects/sample1/locations/sample2/agents/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = test_case.CalculateCoverageResponse(agent='agent_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = test_case.CalculateCoverageResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.calculate_coverage(request)
    assert isinstance(response, test_case.CalculateCoverageResponse)
    assert response.agent == 'agent_value'

def test_calculate_coverage_rest_required_fields(request_type=test_case.CalculateCoverageRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.TestCasesRestTransport
    request_init = {}
    request_init['agent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).calculate_coverage._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['agent'] = 'agent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).calculate_coverage._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('type_',))
    jsonified_request.update(unset_fields)
    assert 'agent' in jsonified_request
    assert jsonified_request['agent'] == 'agent_value'
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = test_case.CalculateCoverageResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = test_case.CalculateCoverageResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.calculate_coverage(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_calculate_coverage_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.calculate_coverage._get_unset_required_fields({})
    assert set(unset_fields) == set(('type',)) & set(('agent', 'type'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_calculate_coverage_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TestCasesRestInterceptor())
    client = TestCasesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TestCasesRestInterceptor, 'post_calculate_coverage') as post, mock.patch.object(transports.TestCasesRestInterceptor, 'pre_calculate_coverage') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = test_case.CalculateCoverageRequest.pb(test_case.CalculateCoverageRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = test_case.CalculateCoverageResponse.to_json(test_case.CalculateCoverageResponse())
        request = test_case.CalculateCoverageRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = test_case.CalculateCoverageResponse()
        client.calculate_coverage(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_calculate_coverage_rest_bad_request(transport: str='rest', request_type=test_case.CalculateCoverageRequest):
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'agent': 'projects/sample1/locations/sample2/agents/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.calculate_coverage(request)

def test_calculate_coverage_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [test_case.ImportTestCasesRequest, dict])
def test_import_test_cases_rest(request_type):
    if False:
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/agents/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.import_test_cases(request)
    assert response.operation.name == 'operations/spam'

def test_import_test_cases_rest_required_fields(request_type=test_case.ImportTestCasesRequest):
    if False:
        return 10
    transport_class = transports.TestCasesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).import_test_cases._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).import_test_cases._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.import_test_cases(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_import_test_cases_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.import_test_cases._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_import_test_cases_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TestCasesRestInterceptor())
    client = TestCasesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.TestCasesRestInterceptor, 'post_import_test_cases') as post, mock.patch.object(transports.TestCasesRestInterceptor, 'pre_import_test_cases') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = test_case.ImportTestCasesRequest.pb(test_case.ImportTestCasesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = test_case.ImportTestCasesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.import_test_cases(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_import_test_cases_rest_bad_request(transport: str='rest', request_type=test_case.ImportTestCasesRequest):
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/agents/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.import_test_cases(request)

def test_import_test_cases_rest_error():
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [test_case.ExportTestCasesRequest, dict])
def test_export_test_cases_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/agents/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.export_test_cases(request)
    assert response.operation.name == 'operations/spam'

def test_export_test_cases_rest_required_fields(request_type=test_case.ExportTestCasesRequest):
    if False:
        return 10
    transport_class = transports.TestCasesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).export_test_cases._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).export_test_cases._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.export_test_cases(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_export_test_cases_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.export_test_cases._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_export_test_cases_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TestCasesRestInterceptor())
    client = TestCasesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.TestCasesRestInterceptor, 'post_export_test_cases') as post, mock.patch.object(transports.TestCasesRestInterceptor, 'pre_export_test_cases') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = test_case.ExportTestCasesRequest.pb(test_case.ExportTestCasesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = test_case.ExportTestCasesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.export_test_cases(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_export_test_cases_rest_bad_request(transport: str='rest', request_type=test_case.ExportTestCasesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/agents/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.export_test_cases(request)

def test_export_test_cases_rest_error():
    if False:
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [test_case.ListTestCaseResultsRequest, dict])
def test_list_test_case_results_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/agents/sample3/testCases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = test_case.ListTestCaseResultsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = test_case.ListTestCaseResultsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_test_case_results(request)
    assert isinstance(response, pagers.ListTestCaseResultsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_test_case_results_rest_required_fields(request_type=test_case.ListTestCaseResultsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.TestCasesRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_test_case_results._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_test_case_results._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = test_case.ListTestCaseResultsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = test_case.ListTestCaseResultsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_test_case_results(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_test_case_results_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_test_case_results._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_test_case_results_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TestCasesRestInterceptor())
    client = TestCasesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TestCasesRestInterceptor, 'post_list_test_case_results') as post, mock.patch.object(transports.TestCasesRestInterceptor, 'pre_list_test_case_results') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = test_case.ListTestCaseResultsRequest.pb(test_case.ListTestCaseResultsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = test_case.ListTestCaseResultsResponse.to_json(test_case.ListTestCaseResultsResponse())
        request = test_case.ListTestCaseResultsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = test_case.ListTestCaseResultsResponse()
        client.list_test_case_results(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_test_case_results_rest_bad_request(transport: str='rest', request_type=test_case.ListTestCaseResultsRequest):
    if False:
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/agents/sample3/testCases/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_test_case_results(request)

def test_list_test_case_results_rest_flattened():
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = test_case.ListTestCaseResultsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/agents/sample3/testCases/sample4'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = test_case.ListTestCaseResultsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_test_case_results(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{parent=projects/*/locations/*/agents/*/testCases/*}/results' % client.transport._host, args[1])

def test_list_test_case_results_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_test_case_results(test_case.ListTestCaseResultsRequest(), parent='parent_value')

def test_list_test_case_results_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (test_case.ListTestCaseResultsResponse(test_case_results=[test_case.TestCaseResult(), test_case.TestCaseResult(), test_case.TestCaseResult()], next_page_token='abc'), test_case.ListTestCaseResultsResponse(test_case_results=[], next_page_token='def'), test_case.ListTestCaseResultsResponse(test_case_results=[test_case.TestCaseResult()], next_page_token='ghi'), test_case.ListTestCaseResultsResponse(test_case_results=[test_case.TestCaseResult(), test_case.TestCaseResult()]))
        response = response + response
        response = tuple((test_case.ListTestCaseResultsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/agents/sample3/testCases/sample4'}
        pager = client.list_test_case_results(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, test_case.TestCaseResult) for i in results))
        pages = list(client.list_test_case_results(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [test_case.GetTestCaseResultRequest, dict])
def test_get_test_case_result_rest(request_type):
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/agents/sample3/testCases/sample4/results/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = test_case.TestCaseResult(name='name_value', environment='environment_value', test_result=test_case.TestResult.PASSED)
        response_value = Response()
        response_value.status_code = 200
        return_value = test_case.TestCaseResult.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_test_case_result(request)
    assert isinstance(response, test_case.TestCaseResult)
    assert response.name == 'name_value'
    assert response.environment == 'environment_value'
    assert response.test_result == test_case.TestResult.PASSED

def test_get_test_case_result_rest_required_fields(request_type=test_case.GetTestCaseResultRequest):
    if False:
        return 10
    transport_class = transports.TestCasesRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_test_case_result._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_test_case_result._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = test_case.TestCaseResult()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = test_case.TestCaseResult.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_test_case_result(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_test_case_result_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_test_case_result._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_test_case_result_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.TestCasesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TestCasesRestInterceptor())
    client = TestCasesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TestCasesRestInterceptor, 'post_get_test_case_result') as post, mock.patch.object(transports.TestCasesRestInterceptor, 'pre_get_test_case_result') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = test_case.GetTestCaseResultRequest.pb(test_case.GetTestCaseResultRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = test_case.TestCaseResult.to_json(test_case.TestCaseResult())
        request = test_case.GetTestCaseResultRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = test_case.TestCaseResult()
        client.get_test_case_result(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_test_case_result_rest_bad_request(transport: str='rest', request_type=test_case.GetTestCaseResultRequest):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/agents/sample3/testCases/sample4/results/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_test_case_result(request)

def test_get_test_case_result_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = test_case.TestCaseResult()
        sample_request = {'name': 'projects/sample1/locations/sample2/agents/sample3/testCases/sample4/results/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = test_case.TestCaseResult.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_test_case_result(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{name=projects/*/locations/*/agents/*/testCases/*/results/*}' % client.transport._host, args[1])

def test_get_test_case_result_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_test_case_result(test_case.GetTestCaseResultRequest(), name='name_value')

def test_get_test_case_result_rest_error():
    if False:
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.TestCasesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.TestCasesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TestCasesClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.TestCasesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = TestCasesClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = TestCasesClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.TestCasesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TestCasesClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TestCasesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = TestCasesClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.TestCasesGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.TestCasesGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.TestCasesGrpcTransport, transports.TestCasesGrpcAsyncIOTransport, transports.TestCasesRestTransport])
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
        while True:
            i = 10
    transport = TestCasesClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.TestCasesGrpcTransport)

def test_test_cases_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.TestCasesTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_test_cases_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.dialogflowcx_v3beta1.services.test_cases.transports.TestCasesTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.TestCasesTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_test_cases', 'batch_delete_test_cases', 'get_test_case', 'create_test_case', 'update_test_case', 'run_test_case', 'batch_run_test_cases', 'calculate_coverage', 'import_test_cases', 'export_test_cases', 'list_test_case_results', 'get_test_case_result', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'list_operations')
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

def test_test_cases_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.dialogflowcx_v3beta1.services.test_cases.transports.TestCasesTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.TestCasesTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id='octopus')

def test_test_cases_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.dialogflowcx_v3beta1.services.test_cases.transports.TestCasesTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.TestCasesTransport()
        adc.assert_called_once()

def test_test_cases_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        TestCasesClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.TestCasesGrpcTransport, transports.TestCasesGrpcAsyncIOTransport])
def test_test_cases_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.TestCasesGrpcTransport, transports.TestCasesGrpcAsyncIOTransport, transports.TestCasesRestTransport])
def test_test_cases_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.TestCasesGrpcTransport, grpc_helpers), (transports.TestCasesGrpcAsyncIOTransport, grpc_helpers_async)])
def test_test_cases_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('dialogflow.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), scopes=['1', '2'], default_host='dialogflow.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.TestCasesGrpcTransport, transports.TestCasesGrpcAsyncIOTransport])
def test_test_cases_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_test_cases_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.TestCasesRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_test_cases_rest_lro_client():
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_test_cases_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dialogflow.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_test_cases_host_with_port(transport_name):
    if False:
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dialogflow.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('dialogflow.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_test_cases_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = TestCasesClient(credentials=creds1, transport=transport_name)
    client2 = TestCasesClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_test_cases._session
    session2 = client2.transport.list_test_cases._session
    assert session1 != session2
    session1 = client1.transport.batch_delete_test_cases._session
    session2 = client2.transport.batch_delete_test_cases._session
    assert session1 != session2
    session1 = client1.transport.get_test_case._session
    session2 = client2.transport.get_test_case._session
    assert session1 != session2
    session1 = client1.transport.create_test_case._session
    session2 = client2.transport.create_test_case._session
    assert session1 != session2
    session1 = client1.transport.update_test_case._session
    session2 = client2.transport.update_test_case._session
    assert session1 != session2
    session1 = client1.transport.run_test_case._session
    session2 = client2.transport.run_test_case._session
    assert session1 != session2
    session1 = client1.transport.batch_run_test_cases._session
    session2 = client2.transport.batch_run_test_cases._session
    assert session1 != session2
    session1 = client1.transport.calculate_coverage._session
    session2 = client2.transport.calculate_coverage._session
    assert session1 != session2
    session1 = client1.transport.import_test_cases._session
    session2 = client2.transport.import_test_cases._session
    assert session1 != session2
    session1 = client1.transport.export_test_cases._session
    session2 = client2.transport.export_test_cases._session
    assert session1 != session2
    session1 = client1.transport.list_test_case_results._session
    session2 = client2.transport.list_test_case_results._session
    assert session1 != session2
    session1 = client1.transport.get_test_case_result._session
    session2 = client2.transport.get_test_case_result._session
    assert session1 != session2

def test_test_cases_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.TestCasesGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_test_cases_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.TestCasesGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.TestCasesGrpcTransport, transports.TestCasesGrpcAsyncIOTransport])
def test_test_cases_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.TestCasesGrpcTransport, transports.TestCasesGrpcAsyncIOTransport])
def test_test_cases_transport_channel_mtls_with_adc(transport_class):
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

def test_test_cases_grpc_lro_client():
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_test_cases_grpc_lro_async_client():
    if False:
        print('Hello World!')
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_agent_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    agent = 'whelk'
    expected = 'projects/{project}/locations/{location}/agents/{agent}'.format(project=project, location=location, agent=agent)
    actual = TestCasesClient.agent_path(project, location, agent)
    assert expected == actual

def test_parse_agent_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'octopus', 'location': 'oyster', 'agent': 'nudibranch'}
    path = TestCasesClient.agent_path(**expected)
    actual = TestCasesClient.parse_agent_path(path)
    assert expected == actual

def test_entity_type_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    location = 'mussel'
    agent = 'winkle'
    entity_type = 'nautilus'
    expected = 'projects/{project}/locations/{location}/agents/{agent}/entityTypes/{entity_type}'.format(project=project, location=location, agent=agent, entity_type=entity_type)
    actual = TestCasesClient.entity_type_path(project, location, agent, entity_type)
    assert expected == actual

def test_parse_entity_type_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'scallop', 'location': 'abalone', 'agent': 'squid', 'entity_type': 'clam'}
    path = TestCasesClient.entity_type_path(**expected)
    actual = TestCasesClient.parse_entity_type_path(path)
    assert expected == actual

def test_environment_path():
    if False:
        while True:
            i = 10
    project = 'whelk'
    location = 'octopus'
    agent = 'oyster'
    environment = 'nudibranch'
    expected = 'projects/{project}/locations/{location}/agents/{agent}/environments/{environment}'.format(project=project, location=location, agent=agent, environment=environment)
    actual = TestCasesClient.environment_path(project, location, agent, environment)
    assert expected == actual

def test_parse_environment_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'cuttlefish', 'location': 'mussel', 'agent': 'winkle', 'environment': 'nautilus'}
    path = TestCasesClient.environment_path(**expected)
    actual = TestCasesClient.parse_environment_path(path)
    assert expected == actual

def test_flow_path():
    if False:
        print('Hello World!')
    project = 'scallop'
    location = 'abalone'
    agent = 'squid'
    flow = 'clam'
    expected = 'projects/{project}/locations/{location}/agents/{agent}/flows/{flow}'.format(project=project, location=location, agent=agent, flow=flow)
    actual = TestCasesClient.flow_path(project, location, agent, flow)
    assert expected == actual

def test_parse_flow_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'whelk', 'location': 'octopus', 'agent': 'oyster', 'flow': 'nudibranch'}
    path = TestCasesClient.flow_path(**expected)
    actual = TestCasesClient.parse_flow_path(path)
    assert expected == actual

def test_intent_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    location = 'mussel'
    agent = 'winkle'
    intent = 'nautilus'
    expected = 'projects/{project}/locations/{location}/agents/{agent}/intents/{intent}'.format(project=project, location=location, agent=agent, intent=intent)
    actual = TestCasesClient.intent_path(project, location, agent, intent)
    assert expected == actual

def test_parse_intent_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'scallop', 'location': 'abalone', 'agent': 'squid', 'intent': 'clam'}
    path = TestCasesClient.intent_path(**expected)
    actual = TestCasesClient.parse_intent_path(path)
    assert expected == actual

def test_page_path():
    if False:
        print('Hello World!')
    project = 'whelk'
    location = 'octopus'
    agent = 'oyster'
    flow = 'nudibranch'
    page = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/agents/{agent}/flows/{flow}/pages/{page}'.format(project=project, location=location, agent=agent, flow=flow, page=page)
    actual = TestCasesClient.page_path(project, location, agent, flow, page)
    assert expected == actual

def test_parse_page_path():
    if False:
        return 10
    expected = {'project': 'mussel', 'location': 'winkle', 'agent': 'nautilus', 'flow': 'scallop', 'page': 'abalone'}
    path = TestCasesClient.page_path(**expected)
    actual = TestCasesClient.parse_page_path(path)
    assert expected == actual

def test_test_case_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    agent = 'whelk'
    test_case = 'octopus'
    expected = 'projects/{project}/locations/{location}/agents/{agent}/testCases/{test_case}'.format(project=project, location=location, agent=agent, test_case=test_case)
    actual = TestCasesClient.test_case_path(project, location, agent, test_case)
    assert expected == actual

def test_parse_test_case_path():
    if False:
        return 10
    expected = {'project': 'oyster', 'location': 'nudibranch', 'agent': 'cuttlefish', 'test_case': 'mussel'}
    path = TestCasesClient.test_case_path(**expected)
    actual = TestCasesClient.parse_test_case_path(path)
    assert expected == actual

def test_test_case_result_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    location = 'nautilus'
    agent = 'scallop'
    test_case = 'abalone'
    result = 'squid'
    expected = 'projects/{project}/locations/{location}/agents/{agent}/testCases/{test_case}/results/{result}'.format(project=project, location=location, agent=agent, test_case=test_case, result=result)
    actual = TestCasesClient.test_case_result_path(project, location, agent, test_case, result)
    assert expected == actual

def test_parse_test_case_result_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'clam', 'location': 'whelk', 'agent': 'octopus', 'test_case': 'oyster', 'result': 'nudibranch'}
    path = TestCasesClient.test_case_result_path(**expected)
    actual = TestCasesClient.parse_test_case_result_path(path)
    assert expected == actual

def test_transition_route_group_path():
    if False:
        return 10
    project = 'cuttlefish'
    location = 'mussel'
    agent = 'winkle'
    flow = 'nautilus'
    transition_route_group = 'scallop'
    expected = 'projects/{project}/locations/{location}/agents/{agent}/flows/{flow}/transitionRouteGroups/{transition_route_group}'.format(project=project, location=location, agent=agent, flow=flow, transition_route_group=transition_route_group)
    actual = TestCasesClient.transition_route_group_path(project, location, agent, flow, transition_route_group)
    assert expected == actual

def test_parse_transition_route_group_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'abalone', 'location': 'squid', 'agent': 'clam', 'flow': 'whelk', 'transition_route_group': 'octopus'}
    path = TestCasesClient.transition_route_group_path(**expected)
    actual = TestCasesClient.parse_transition_route_group_path(path)
    assert expected == actual

def test_webhook_path():
    if False:
        i = 10
        return i + 15
    project = 'oyster'
    location = 'nudibranch'
    agent = 'cuttlefish'
    webhook = 'mussel'
    expected = 'projects/{project}/locations/{location}/agents/{agent}/webhooks/{webhook}'.format(project=project, location=location, agent=agent, webhook=webhook)
    actual = TestCasesClient.webhook_path(project, location, agent, webhook)
    assert expected == actual

def test_parse_webhook_path():
    if False:
        print('Hello World!')
    expected = {'project': 'winkle', 'location': 'nautilus', 'agent': 'scallop', 'webhook': 'abalone'}
    path = TestCasesClient.webhook_path(**expected)
    actual = TestCasesClient.parse_webhook_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = TestCasesClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'clam'}
    path = TestCasesClient.common_billing_account_path(**expected)
    actual = TestCasesClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = TestCasesClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'octopus'}
    path = TestCasesClient.common_folder_path(**expected)
    actual = TestCasesClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = TestCasesClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'nudibranch'}
    path = TestCasesClient.common_organization_path(**expected)
    actual = TestCasesClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = TestCasesClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'mussel'}
    path = TestCasesClient.common_project_path(**expected)
    actual = TestCasesClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = TestCasesClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = TestCasesClient.common_location_path(**expected)
    actual = TestCasesClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.TestCasesTransport, '_prep_wrapped_messages') as prep:
        client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.TestCasesTransport, '_prep_wrapped_messages') as prep:
        transport_class = TestCasesClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_cancel_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.CancelOperationRequest):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/operations/sample2'}, request)
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
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/operations/sample2'}
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

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_list_operations_rest_bad_request(transport: str='rest', request_type=operations_pb2.ListOperationsRequest):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1'}, request)
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
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1'}
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

def test_cancel_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        return 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = TestCasesAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        for i in range(10):
            print('nop')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = TestCasesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(TestCasesClient, transports.TestCasesGrpcTransport), (TestCasesAsyncClient, transports.TestCasesGrpcAsyncIOTransport)])
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