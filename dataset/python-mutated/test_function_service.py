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
from google.cloud.functions_v2.services.function_service import FunctionServiceAsyncClient, FunctionServiceClient, pagers, transports
from google.cloud.functions_v2.types import functions

def client_cert_source_callback():
    if False:
        for i in range(10):
            print('nop')
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
    assert FunctionServiceClient._get_default_mtls_endpoint(None) is None
    assert FunctionServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert FunctionServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert FunctionServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert FunctionServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert FunctionServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(FunctionServiceClient, 'grpc'), (FunctionServiceAsyncClient, 'grpc_asyncio'), (FunctionServiceClient, 'rest')])
def test_function_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('cloudfunctions.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudfunctions.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.FunctionServiceGrpcTransport, 'grpc'), (transports.FunctionServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.FunctionServiceRestTransport, 'rest')])
def test_function_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(FunctionServiceClient, 'grpc'), (FunctionServiceAsyncClient, 'grpc_asyncio'), (FunctionServiceClient, 'rest')])
def test_function_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('cloudfunctions.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudfunctions.googleapis.com')

def test_function_service_client_get_transport_class():
    if False:
        for i in range(10):
            print('nop')
    transport = FunctionServiceClient.get_transport_class()
    available_transports = [transports.FunctionServiceGrpcTransport, transports.FunctionServiceRestTransport]
    assert transport in available_transports
    transport = FunctionServiceClient.get_transport_class('grpc')
    assert transport == transports.FunctionServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(FunctionServiceClient, transports.FunctionServiceGrpcTransport, 'grpc'), (FunctionServiceAsyncClient, transports.FunctionServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (FunctionServiceClient, transports.FunctionServiceRestTransport, 'rest')])
@mock.patch.object(FunctionServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(FunctionServiceClient))
@mock.patch.object(FunctionServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(FunctionServiceAsyncClient))
def test_function_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(FunctionServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(FunctionServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(FunctionServiceClient, transports.FunctionServiceGrpcTransport, 'grpc', 'true'), (FunctionServiceAsyncClient, transports.FunctionServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (FunctionServiceClient, transports.FunctionServiceGrpcTransport, 'grpc', 'false'), (FunctionServiceAsyncClient, transports.FunctionServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (FunctionServiceClient, transports.FunctionServiceRestTransport, 'rest', 'true'), (FunctionServiceClient, transports.FunctionServiceRestTransport, 'rest', 'false')])
@mock.patch.object(FunctionServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(FunctionServiceClient))
@mock.patch.object(FunctionServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(FunctionServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_function_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [FunctionServiceClient, FunctionServiceAsyncClient])
@mock.patch.object(FunctionServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(FunctionServiceClient))
@mock.patch.object(FunctionServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(FunctionServiceAsyncClient))
def test_function_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(FunctionServiceClient, transports.FunctionServiceGrpcTransport, 'grpc'), (FunctionServiceAsyncClient, transports.FunctionServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (FunctionServiceClient, transports.FunctionServiceRestTransport, 'rest')])
def test_function_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(FunctionServiceClient, transports.FunctionServiceGrpcTransport, 'grpc', grpc_helpers), (FunctionServiceAsyncClient, transports.FunctionServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (FunctionServiceClient, transports.FunctionServiceRestTransport, 'rest', None)])
def test_function_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_function_service_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.functions_v2.services.function_service.transports.FunctionServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = FunctionServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(FunctionServiceClient, transports.FunctionServiceGrpcTransport, 'grpc', grpc_helpers), (FunctionServiceAsyncClient, transports.FunctionServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_function_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
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

@pytest.mark.parametrize('request_type', [functions.GetFunctionRequest, dict])
def test_get_function(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_function), '__call__') as call:
        call.return_value = functions.Function(name='name_value', description='description_value', state=functions.Function.State.ACTIVE, environment=functions.Environment.GEN_1, url='url_value', kms_key_name='kms_key_name_value')
        response = client.get_function(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.GetFunctionRequest()
    assert isinstance(response, functions.Function)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == functions.Function.State.ACTIVE
    assert response.environment == functions.Environment.GEN_1
    assert response.url == 'url_value'
    assert response.kms_key_name == 'kms_key_name_value'

def test_get_function_empty_call():
    if False:
        i = 10
        return i + 15
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_function), '__call__') as call:
        client.get_function()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.GetFunctionRequest()

@pytest.mark.asyncio
async def test_get_function_async(transport: str='grpc_asyncio', request_type=functions.GetFunctionRequest):
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_function), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.Function(name='name_value', description='description_value', state=functions.Function.State.ACTIVE, environment=functions.Environment.GEN_1, url='url_value', kms_key_name='kms_key_name_value'))
        response = await client.get_function(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.GetFunctionRequest()
    assert isinstance(response, functions.Function)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == functions.Function.State.ACTIVE
    assert response.environment == functions.Environment.GEN_1
    assert response.url == 'url_value'
    assert response.kms_key_name == 'kms_key_name_value'

@pytest.mark.asyncio
async def test_get_function_async_from_dict():
    await test_get_function_async(request_type=dict)

def test_get_function_field_headers():
    if False:
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.GetFunctionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_function), '__call__') as call:
        call.return_value = functions.Function()
        client.get_function(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_function_field_headers_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.GetFunctionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_function), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.Function())
        await client.get_function(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_function_flattened():
    if False:
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_function), '__call__') as call:
        call.return_value = functions.Function()
        client.get_function(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_function_flattened_error():
    if False:
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_function(functions.GetFunctionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_function_flattened_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_function), '__call__') as call:
        call.return_value = functions.Function()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.Function())
        response = await client.get_function(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_function_flattened_error_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_function(functions.GetFunctionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [functions.ListFunctionsRequest, dict])
def test_list_functions(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_functions), '__call__') as call:
        client.list_functions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.ListFunctionsRequest()

@pytest.mark.asyncio
async def test_list_functions_async(transport: str='grpc_asyncio', request_type=functions.ListFunctionsRequest):
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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

def test_list_functions_flattened():
    if False:
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_functions), '__call__') as call:
        call.return_value = functions.ListFunctionsResponse()
        client.list_functions(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_functions_flattened_error():
    if False:
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_functions(functions.ListFunctionsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_functions_flattened_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_functions), '__call__') as call:
        call.return_value = functions.ListFunctionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.ListFunctionsResponse())
        response = await client.list_functions(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_functions_flattened_error_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_functions(functions.ListFunctionsRequest(), parent='parent_value')

def test_list_functions_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_functions), '__call__') as call:
        call.side_effect = (functions.ListFunctionsResponse(functions=[functions.Function(), functions.Function(), functions.Function()], next_page_token='abc'), functions.ListFunctionsResponse(functions=[], next_page_token='def'), functions.ListFunctionsResponse(functions=[functions.Function()], next_page_token='ghi'), functions.ListFunctionsResponse(functions=[functions.Function(), functions.Function()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_functions(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, functions.Function) for i in results))

def test_list_functions_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_functions), '__call__') as call:
        call.side_effect = (functions.ListFunctionsResponse(functions=[functions.Function(), functions.Function(), functions.Function()], next_page_token='abc'), functions.ListFunctionsResponse(functions=[], next_page_token='def'), functions.ListFunctionsResponse(functions=[functions.Function()], next_page_token='ghi'), functions.ListFunctionsResponse(functions=[functions.Function(), functions.Function()]), RuntimeError)
        pages = list(client.list_functions(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_functions_async_pager():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_functions), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (functions.ListFunctionsResponse(functions=[functions.Function(), functions.Function(), functions.Function()], next_page_token='abc'), functions.ListFunctionsResponse(functions=[], next_page_token='def'), functions.ListFunctionsResponse(functions=[functions.Function()], next_page_token='ghi'), functions.ListFunctionsResponse(functions=[functions.Function(), functions.Function()]), RuntimeError)
        async_pager = await client.list_functions(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, functions.Function) for i in responses))

@pytest.mark.asyncio
async def test_list_functions_async_pages():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_functions), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (functions.ListFunctionsResponse(functions=[functions.Function(), functions.Function(), functions.Function()], next_page_token='abc'), functions.ListFunctionsResponse(functions=[], next_page_token='def'), functions.ListFunctionsResponse(functions=[functions.Function()], next_page_token='ghi'), functions.ListFunctionsResponse(functions=[functions.Function(), functions.Function()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_functions(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [functions.CreateFunctionRequest, dict])
def test_create_function(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_function), '__call__') as call:
        client.create_function()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.CreateFunctionRequest()

@pytest.mark.asyncio
async def test_create_function_async(transport: str='grpc_asyncio', request_type=functions.CreateFunctionRequest):
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.CreateFunctionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_function), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_function(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_function_field_headers_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.CreateFunctionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_function), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_function(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_function_flattened():
    if False:
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_function), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_function(parent='parent_value', function=functions.Function(name='name_value'), function_id='function_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].function
        mock_val = functions.Function(name='name_value')
        assert arg == mock_val
        arg = args[0].function_id
        mock_val = 'function_id_value'
        assert arg == mock_val

def test_create_function_flattened_error():
    if False:
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_function(functions.CreateFunctionRequest(), parent='parent_value', function=functions.Function(name='name_value'), function_id='function_id_value')

@pytest.mark.asyncio
async def test_create_function_flattened_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_function), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_function(parent='parent_value', function=functions.Function(name='name_value'), function_id='function_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].function
        mock_val = functions.Function(name='name_value')
        assert arg == mock_val
        arg = args[0].function_id
        mock_val = 'function_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_function_flattened_error_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_function(functions.CreateFunctionRequest(), parent='parent_value', function=functions.Function(name='name_value'), function_id='function_id_value')

@pytest.mark.parametrize('request_type', [functions.UpdateFunctionRequest, dict])
def test_update_function(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_function), '__call__') as call:
        client.update_function()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.UpdateFunctionRequest()

@pytest.mark.asyncio
async def test_update_function_async(transport: str='grpc_asyncio', request_type=functions.UpdateFunctionRequest):
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_function), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_function(function=functions.Function(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].function
        mock_val = functions.Function(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_function_flattened_error():
    if False:
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_function(functions.UpdateFunctionRequest(), function=functions.Function(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_function_flattened_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_function), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_function(function=functions.Function(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].function
        mock_val = functions.Function(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_function_flattened_error_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_function(functions.UpdateFunctionRequest(), function=functions.Function(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [functions.DeleteFunctionRequest, dict])
def test_delete_function(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_function), '__call__') as call:
        client.delete_function()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.DeleteFunctionRequest()

@pytest.mark.asyncio
async def test_delete_function_async(transport: str='grpc_asyncio', request_type=functions.DeleteFunctionRequest):
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_function(functions.DeleteFunctionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_function_flattened_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_function(functions.DeleteFunctionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [functions.GenerateUploadUrlRequest, dict])
def test_generate_upload_url(request_type, transport: str='grpc'):
    if False:
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.generate_upload_url), '__call__') as call:
        client.generate_upload_url()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.GenerateUploadUrlRequest()

@pytest.mark.asyncio
async def test_generate_upload_url_async(transport: str='grpc_asyncio', request_type=functions.GenerateUploadUrlRequest):
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.generate_download_url), '__call__') as call:
        client.generate_download_url()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.GenerateDownloadUrlRequest()

@pytest.mark.asyncio
async def test_generate_download_url_async(transport: str='grpc_asyncio', request_type=functions.GenerateDownloadUrlRequest):
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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

@pytest.mark.parametrize('request_type', [functions.ListRuntimesRequest, dict])
def test_list_runtimes(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_runtimes), '__call__') as call:
        call.return_value = functions.ListRuntimesResponse()
        response = client.list_runtimes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.ListRuntimesRequest()
    assert isinstance(response, functions.ListRuntimesResponse)

def test_list_runtimes_empty_call():
    if False:
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_runtimes), '__call__') as call:
        client.list_runtimes()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.ListRuntimesRequest()

@pytest.mark.asyncio
async def test_list_runtimes_async(transport: str='grpc_asyncio', request_type=functions.ListRuntimesRequest):
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_runtimes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.ListRuntimesResponse())
        response = await client.list_runtimes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == functions.ListRuntimesRequest()
    assert isinstance(response, functions.ListRuntimesResponse)

@pytest.mark.asyncio
async def test_list_runtimes_async_from_dict():
    await test_list_runtimes_async(request_type=dict)

def test_list_runtimes_field_headers():
    if False:
        i = 10
        return i + 15
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.ListRuntimesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_runtimes), '__call__') as call:
        call.return_value = functions.ListRuntimesResponse()
        client.list_runtimes(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_runtimes_field_headers_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = functions.ListRuntimesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_runtimes), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.ListRuntimesResponse())
        await client.list_runtimes(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_runtimes_flattened():
    if False:
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_runtimes), '__call__') as call:
        call.return_value = functions.ListRuntimesResponse()
        client.list_runtimes(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_runtimes_flattened_error():
    if False:
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_runtimes(functions.ListRuntimesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_runtimes_flattened_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_runtimes), '__call__') as call:
        call.return_value = functions.ListRuntimesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(functions.ListRuntimesResponse())
        response = await client.list_runtimes(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_runtimes_flattened_error_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_runtimes(functions.ListRuntimesRequest(), parent='parent_value')

@pytest.mark.parametrize('request_type', [functions.GetFunctionRequest, dict])
def test_get_function_rest(request_type):
    if False:
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/functions/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = functions.Function(name='name_value', description='description_value', state=functions.Function.State.ACTIVE, environment=functions.Environment.GEN_1, url='url_value', kms_key_name='kms_key_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = functions.Function.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_function(request)
    assert isinstance(response, functions.Function)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.state == functions.Function.State.ACTIVE
    assert response.environment == functions.Environment.GEN_1
    assert response.url == 'url_value'
    assert response.kms_key_name == 'kms_key_name_value'

def test_get_function_rest_required_fields(request_type=functions.GetFunctionRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.FunctionServiceRestTransport
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
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = functions.Function()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = functions.Function.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_function(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_function_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.FunctionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_function._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_function_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.FunctionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FunctionServiceRestInterceptor())
    client = FunctionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.FunctionServiceRestInterceptor, 'post_get_function') as post, mock.patch.object(transports.FunctionServiceRestInterceptor, 'pre_get_function') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = functions.GetFunctionRequest.pb(functions.GetFunctionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = functions.Function.to_json(functions.Function())
        request = functions.GetFunctionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = functions.Function()
        client.get_function(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_function_rest_bad_request(transport: str='rest', request_type=functions.GetFunctionRequest):
    if False:
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = functions.Function()
        sample_request = {'name': 'projects/sample1/locations/sample2/functions/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = functions.Function.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_function(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/functions/*}' % client.transport._host, args[1])

def test_get_function_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_function(functions.GetFunctionRequest(), name='name_value')

def test_get_function_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [functions.ListFunctionsRequest, dict])
def test_list_functions_rest(request_type):
    if False:
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_list_functions_rest_required_fields(request_type=functions.ListFunctionsRequest):
    if False:
        return 10
    transport_class = transports.FunctionServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_functions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_functions._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = functions.ListFunctionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = functions.ListFunctionsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_functions(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_functions_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.FunctionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_functions._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_functions_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.FunctionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FunctionServiceRestInterceptor())
    client = FunctionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.FunctionServiceRestInterceptor, 'post_list_functions') as post, mock.patch.object(transports.FunctionServiceRestInterceptor, 'pre_list_functions') as pre:
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
        for i in range(10):
            print('nop')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_functions(request)

def test_list_functions_rest_flattened():
    if False:
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = functions.ListFunctionsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = functions.ListFunctionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_functions(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/functions' % client.transport._host, args[1])

def test_list_functions_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_functions(functions.ListFunctionsRequest(), parent='parent_value')

def test_list_functions_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (functions.ListFunctionsResponse(functions=[functions.Function(), functions.Function(), functions.Function()], next_page_token='abc'), functions.ListFunctionsResponse(functions=[], next_page_token='def'), functions.ListFunctionsResponse(functions=[functions.Function()], next_page_token='ghi'), functions.ListFunctionsResponse(functions=[functions.Function(), functions.Function()]))
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
        assert all((isinstance(i, functions.Function) for i in results))
        pages = list(client.list_functions(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [functions.CreateFunctionRequest, dict])
def test_create_function_rest(request_type):
    if False:
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['function'] = {'name': 'name_value', 'description': 'description_value', 'build_config': {'build': 'build_value', 'runtime': 'runtime_value', 'entry_point': 'entry_point_value', 'source': {'storage_source': {'bucket': 'bucket_value', 'object_': 'object__value', 'generation': 1068}, 'repo_source': {'branch_name': 'branch_name_value', 'tag_name': 'tag_name_value', 'commit_sha': 'commit_sha_value', 'project_id': 'project_id_value', 'repo_name': 'repo_name_value', 'dir_': 'dir__value', 'invert_regex': True}}, 'source_provenance': {'resolved_storage_source': {}, 'resolved_repo_source': {}}, 'worker_pool': 'worker_pool_value', 'environment_variables': {}, 'docker_registry': 1, 'docker_repository': 'docker_repository_value'}, 'service_config': {'service': 'service_value', 'timeout_seconds': 1621, 'available_memory': 'available_memory_value', 'available_cpu': 'available_cpu_value', 'environment_variables': {}, 'max_instance_count': 1922, 'min_instance_count': 1920, 'vpc_connector': 'vpc_connector_value', 'vpc_connector_egress_settings': 1, 'ingress_settings': 1, 'uri': 'uri_value', 'service_account_email': 'service_account_email_value', 'all_traffic_on_latest_revision': True, 'secret_environment_variables': [{'key': 'key_value', 'project_id': 'project_id_value', 'secret': 'secret_value', 'version': 'version_value'}], 'secret_volumes': [{'mount_path': 'mount_path_value', 'project_id': 'project_id_value', 'secret': 'secret_value', 'versions': [{'version': 'version_value', 'path': 'path_value'}]}], 'revision': 'revision_value', 'max_instance_request_concurrency': 3436, 'security_level': 1}, 'event_trigger': {'trigger': 'trigger_value', 'trigger_region': 'trigger_region_value', 'event_type': 'event_type_value', 'event_filters': [{'attribute': 'attribute_value', 'value': 'value_value', 'operator': 'operator_value'}], 'pubsub_topic': 'pubsub_topic_value', 'service_account_email': 'service_account_email_value', 'retry_policy': 1, 'channel': 'channel_value'}, 'state': 1, 'update_time': {'seconds': 751, 'nanos': 543}, 'labels': {}, 'state_messages': [{'severity': 1, 'type_': 'type__value', 'message': 'message_value'}], 'environment': 1, 'url': 'url_value', 'kms_key_name': 'kms_key_name_value'}
    test_field = functions.CreateFunctionRequest.meta.fields['function']

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
        while True:
            i = 10
    transport_class = transports.FunctionServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_function._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_function._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('function_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    transport = transports.FunctionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_function._get_unset_required_fields({})
    assert set(unset_fields) == set(('functionId',)) & set(('parent', 'function'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_function_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.FunctionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FunctionServiceRestInterceptor())
    client = FunctionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.FunctionServiceRestInterceptor, 'post_create_function') as post, mock.patch.object(transports.FunctionServiceRestInterceptor, 'pre_create_function') as pre:
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
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
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
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', function=functions.Function(name='name_value'), function_id='function_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_function(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/functions' % client.transport._host, args[1])

def test_create_function_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_function(functions.CreateFunctionRequest(), parent='parent_value', function=functions.Function(name='name_value'), function_id='function_id_value')

def test_create_function_rest_error():
    if False:
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [functions.UpdateFunctionRequest, dict])
def test_update_function_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'function': {'name': 'projects/sample1/locations/sample2/functions/sample3'}}
    request_init['function'] = {'name': 'projects/sample1/locations/sample2/functions/sample3', 'description': 'description_value', 'build_config': {'build': 'build_value', 'runtime': 'runtime_value', 'entry_point': 'entry_point_value', 'source': {'storage_source': {'bucket': 'bucket_value', 'object_': 'object__value', 'generation': 1068}, 'repo_source': {'branch_name': 'branch_name_value', 'tag_name': 'tag_name_value', 'commit_sha': 'commit_sha_value', 'project_id': 'project_id_value', 'repo_name': 'repo_name_value', 'dir_': 'dir__value', 'invert_regex': True}}, 'source_provenance': {'resolved_storage_source': {}, 'resolved_repo_source': {}}, 'worker_pool': 'worker_pool_value', 'environment_variables': {}, 'docker_registry': 1, 'docker_repository': 'docker_repository_value'}, 'service_config': {'service': 'service_value', 'timeout_seconds': 1621, 'available_memory': 'available_memory_value', 'available_cpu': 'available_cpu_value', 'environment_variables': {}, 'max_instance_count': 1922, 'min_instance_count': 1920, 'vpc_connector': 'vpc_connector_value', 'vpc_connector_egress_settings': 1, 'ingress_settings': 1, 'uri': 'uri_value', 'service_account_email': 'service_account_email_value', 'all_traffic_on_latest_revision': True, 'secret_environment_variables': [{'key': 'key_value', 'project_id': 'project_id_value', 'secret': 'secret_value', 'version': 'version_value'}], 'secret_volumes': [{'mount_path': 'mount_path_value', 'project_id': 'project_id_value', 'secret': 'secret_value', 'versions': [{'version': 'version_value', 'path': 'path_value'}]}], 'revision': 'revision_value', 'max_instance_request_concurrency': 3436, 'security_level': 1}, 'event_trigger': {'trigger': 'trigger_value', 'trigger_region': 'trigger_region_value', 'event_type': 'event_type_value', 'event_filters': [{'attribute': 'attribute_value', 'value': 'value_value', 'operator': 'operator_value'}], 'pubsub_topic': 'pubsub_topic_value', 'service_account_email': 'service_account_email_value', 'retry_policy': 1, 'channel': 'channel_value'}, 'state': 1, 'update_time': {'seconds': 751, 'nanos': 543}, 'labels': {}, 'state_messages': [{'severity': 1, 'type_': 'type__value', 'message': 'message_value'}], 'environment': 1, 'url': 'url_value', 'kms_key_name': 'kms_key_name_value'}
    test_field = functions.UpdateFunctionRequest.meta.fields['function']

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
        return 10
    transport_class = transports.FunctionServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_function._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_function._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    transport = transports.FunctionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_function._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('function',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_function_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.FunctionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FunctionServiceRestInterceptor())
    client = FunctionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.FunctionServiceRestInterceptor, 'post_update_function') as post, mock.patch.object(transports.FunctionServiceRestInterceptor, 'pre_update_function') as pre:
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
        for i in range(10):
            print('nop')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'function': {'name': 'projects/sample1/locations/sample2/functions/sample3'}}
        mock_args = dict(function=functions.Function(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_function(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{function.name=projects/*/locations/*/functions/*}' % client.transport._host, args[1])

def test_update_function_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_function(functions.UpdateFunctionRequest(), function=functions.Function(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_function_rest_error():
    if False:
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [functions.DeleteFunctionRequest, dict])
def test_delete_function_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    transport_class = transports.FunctionServiceRestTransport
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
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    transport = transports.FunctionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_function._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_function_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.FunctionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FunctionServiceRestInterceptor())
    client = FunctionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.FunctionServiceRestInterceptor, 'post_delete_function') as post, mock.patch.object(transports.FunctionServiceRestInterceptor, 'pre_delete_function') as pre:
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
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/functions/*}' % client.transport._host, args[1])

def test_delete_function_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_function(functions.DeleteFunctionRequest(), name='name_value')

def test_delete_function_rest_error():
    if False:
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [functions.GenerateUploadUrlRequest, dict])
def test_generate_upload_url_rest(request_type):
    if False:
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_generate_upload_url_rest_required_fields(request_type=functions.GenerateUploadUrlRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.FunctionServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_upload_url._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_upload_url._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = functions.GenerateUploadUrlResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = functions.GenerateUploadUrlResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.generate_upload_url(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_generate_upload_url_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.FunctionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.generate_upload_url._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_generate_upload_url_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.FunctionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FunctionServiceRestInterceptor())
    client = FunctionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.FunctionServiceRestInterceptor, 'post_generate_upload_url') as post, mock.patch.object(transports.FunctionServiceRestInterceptor, 'pre_generate_upload_url') as pre:
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
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [functions.GenerateDownloadUrlRequest, dict])
def test_generate_download_url_rest(request_type):
    if False:
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_generate_download_url_rest_required_fields(request_type=functions.GenerateDownloadUrlRequest):
    if False:
        return 10
    transport_class = transports.FunctionServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_download_url._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_download_url._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = functions.GenerateDownloadUrlResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = functions.GenerateDownloadUrlResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.generate_download_url(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_generate_download_url_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.FunctionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.generate_download_url._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_generate_download_url_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.FunctionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FunctionServiceRestInterceptor())
    client = FunctionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.FunctionServiceRestInterceptor, 'post_generate_download_url') as post, mock.patch.object(transports.FunctionServiceRestInterceptor, 'pre_generate_download_url') as pre:
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
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [functions.ListRuntimesRequest, dict])
def test_list_runtimes_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = functions.ListRuntimesResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = functions.ListRuntimesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_runtimes(request)
    assert isinstance(response, functions.ListRuntimesResponse)

def test_list_runtimes_rest_required_fields(request_type=functions.ListRuntimesRequest):
    if False:
        return 10
    transport_class = transports.FunctionServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_runtimes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_runtimes._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = functions.ListRuntimesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = functions.ListRuntimesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_runtimes(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_runtimes_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.FunctionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_runtimes._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter',)) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_runtimes_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.FunctionServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.FunctionServiceRestInterceptor())
    client = FunctionServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.FunctionServiceRestInterceptor, 'post_list_runtimes') as post, mock.patch.object(transports.FunctionServiceRestInterceptor, 'pre_list_runtimes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = functions.ListRuntimesRequest.pb(functions.ListRuntimesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = functions.ListRuntimesResponse.to_json(functions.ListRuntimesResponse())
        request = functions.ListRuntimesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = functions.ListRuntimesResponse()
        client.list_runtimes(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_runtimes_rest_bad_request(transport: str='rest', request_type=functions.ListRuntimesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_runtimes(request)

def test_list_runtimes_rest_flattened():
    if False:
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = functions.ListRuntimesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = functions.ListRuntimesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_runtimes(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/runtimes' % client.transport._host, args[1])

def test_list_runtimes_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_runtimes(functions.ListRuntimesRequest(), parent='parent_value')

def test_list_runtimes_rest_error():
    if False:
        i = 10
        return i + 15
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.FunctionServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.FunctionServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = FunctionServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.FunctionServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = FunctionServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = FunctionServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.FunctionServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = FunctionServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.FunctionServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = FunctionServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.FunctionServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.FunctionServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.FunctionServiceGrpcTransport, transports.FunctionServiceGrpcAsyncIOTransport, transports.FunctionServiceRestTransport])
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
    transport = FunctionServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.FunctionServiceGrpcTransport)

def test_function_service_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.FunctionServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_function_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.functions_v2.services.function_service.transports.FunctionServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.FunctionServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('get_function', 'list_functions', 'create_function', 'update_function', 'delete_function', 'generate_upload_url', 'generate_download_url', 'list_runtimes', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'list_locations', 'get_operation', 'list_operations')
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

def test_function_service_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.functions_v2.services.function_service.transports.FunctionServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.FunctionServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_function_service_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.functions_v2.services.function_service.transports.FunctionServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.FunctionServiceTransport()
        adc.assert_called_once()

def test_function_service_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        FunctionServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.FunctionServiceGrpcTransport, transports.FunctionServiceGrpcAsyncIOTransport])
def test_function_service_transport_auth_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.FunctionServiceGrpcTransport, transports.FunctionServiceGrpcAsyncIOTransport, transports.FunctionServiceRestTransport])
def test_function_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.FunctionServiceGrpcTransport, grpc_helpers), (transports.FunctionServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_function_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('cloudfunctions.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='cloudfunctions.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.FunctionServiceGrpcTransport, transports.FunctionServiceGrpcAsyncIOTransport])
def test_function_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_function_service_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.FunctionServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_function_service_rest_lro_client():
    if False:
        for i in range(10):
            print('nop')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_function_service_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudfunctions.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('cloudfunctions.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudfunctions.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_function_service_host_with_port(transport_name):
    if False:
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudfunctions.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('cloudfunctions.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudfunctions.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_function_service_client_transport_session_collision(transport_name):
    if False:
        while True:
            i = 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = FunctionServiceClient(credentials=creds1, transport=transport_name)
    client2 = FunctionServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.get_function._session
    session2 = client2.transport.get_function._session
    assert session1 != session2
    session1 = client1.transport.list_functions._session
    session2 = client2.transport.list_functions._session
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
    session1 = client1.transport.generate_upload_url._session
    session2 = client2.transport.generate_upload_url._session
    assert session1 != session2
    session1 = client1.transport.generate_download_url._session
    session2 = client2.transport.generate_download_url._session
    assert session1 != session2
    session1 = client1.transport.list_runtimes._session
    session2 = client2.transport.list_runtimes._session
    assert session1 != session2

def test_function_service_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.FunctionServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_function_service_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.FunctionServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.FunctionServiceGrpcTransport, transports.FunctionServiceGrpcAsyncIOTransport])
def test_function_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.FunctionServiceGrpcTransport, transports.FunctionServiceGrpcAsyncIOTransport])
def test_function_service_transport_channel_mtls_with_adc(transport_class):
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

def test_function_service_grpc_lro_client():
    if False:
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_function_service_grpc_lro_async_client():
    if False:
        print('Hello World!')
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_build_path():
    if False:
        return 10
    project = 'squid'
    location = 'clam'
    build = 'whelk'
    expected = 'projects/{project}/locations/{location}/builds/{build}'.format(project=project, location=location, build=build)
    actual = FunctionServiceClient.build_path(project, location, build)
    assert expected == actual

def test_parse_build_path():
    if False:
        return 10
    expected = {'project': 'octopus', 'location': 'oyster', 'build': 'nudibranch'}
    path = FunctionServiceClient.build_path(**expected)
    actual = FunctionServiceClient.parse_build_path(path)
    assert expected == actual

def test_channel_path():
    if False:
        return 10
    project = 'cuttlefish'
    location = 'mussel'
    channel = 'winkle'
    expected = 'projects/{project}/locations/{location}/channels/{channel}'.format(project=project, location=location, channel=channel)
    actual = FunctionServiceClient.channel_path(project, location, channel)
    assert expected == actual

def test_parse_channel_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'nautilus', 'location': 'scallop', 'channel': 'abalone'}
    path = FunctionServiceClient.channel_path(**expected)
    actual = FunctionServiceClient.parse_channel_path(path)
    assert expected == actual

def test_connector_path():
    if False:
        return 10
    project = 'squid'
    location = 'clam'
    connector = 'whelk'
    expected = 'projects/{project}/locations/{location}/connectors/{connector}'.format(project=project, location=location, connector=connector)
    actual = FunctionServiceClient.connector_path(project, location, connector)
    assert expected == actual

def test_parse_connector_path():
    if False:
        print('Hello World!')
    expected = {'project': 'octopus', 'location': 'oyster', 'connector': 'nudibranch'}
    path = FunctionServiceClient.connector_path(**expected)
    actual = FunctionServiceClient.parse_connector_path(path)
    assert expected == actual

def test_crypto_key_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    location = 'mussel'
    key_ring = 'winkle'
    crypto_key = 'nautilus'
    expected = 'projects/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}'.format(project=project, location=location, key_ring=key_ring, crypto_key=crypto_key)
    actual = FunctionServiceClient.crypto_key_path(project, location, key_ring, crypto_key)
    assert expected == actual

def test_parse_crypto_key_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'scallop', 'location': 'abalone', 'key_ring': 'squid', 'crypto_key': 'clam'}
    path = FunctionServiceClient.crypto_key_path(**expected)
    actual = FunctionServiceClient.parse_crypto_key_path(path)
    assert expected == actual

def test_function_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    location = 'octopus'
    function = 'oyster'
    expected = 'projects/{project}/locations/{location}/functions/{function}'.format(project=project, location=location, function=function)
    actual = FunctionServiceClient.function_path(project, location, function)
    assert expected == actual

def test_parse_function_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'function': 'mussel'}
    path = FunctionServiceClient.function_path(**expected)
    actual = FunctionServiceClient.parse_function_path(path)
    assert expected == actual

def test_repository_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    repository = 'scallop'
    expected = 'projects/{project}/locations/{location}/repositories/{repository}'.format(project=project, location=location, repository=repository)
    actual = FunctionServiceClient.repository_path(project, location, repository)
    assert expected == actual

def test_parse_repository_path():
    if False:
        return 10
    expected = {'project': 'abalone', 'location': 'squid', 'repository': 'clam'}
    path = FunctionServiceClient.repository_path(**expected)
    actual = FunctionServiceClient.parse_repository_path(path)
    assert expected == actual

def test_service_path():
    if False:
        while True:
            i = 10
    project = 'whelk'
    location = 'octopus'
    service = 'oyster'
    expected = 'projects/{project}/locations/{location}/services/{service}'.format(project=project, location=location, service=service)
    actual = FunctionServiceClient.service_path(project, location, service)
    assert expected == actual

def test_parse_service_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'service': 'mussel'}
    path = FunctionServiceClient.service_path(**expected)
    actual = FunctionServiceClient.parse_service_path(path)
    assert expected == actual

def test_topic_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    topic = 'nautilus'
    expected = 'projects/{project}/topics/{topic}'.format(project=project, topic=topic)
    actual = FunctionServiceClient.topic_path(project, topic)
    assert expected == actual

def test_parse_topic_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'scallop', 'topic': 'abalone'}
    path = FunctionServiceClient.topic_path(**expected)
    actual = FunctionServiceClient.parse_topic_path(path)
    assert expected == actual

def test_trigger_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    trigger = 'whelk'
    expected = 'projects/{project}/locations/{location}/triggers/{trigger}'.format(project=project, location=location, trigger=trigger)
    actual = FunctionServiceClient.trigger_path(project, location, trigger)
    assert expected == actual

def test_parse_trigger_path():
    if False:
        print('Hello World!')
    expected = {'project': 'octopus', 'location': 'oyster', 'trigger': 'nudibranch'}
    path = FunctionServiceClient.trigger_path(**expected)
    actual = FunctionServiceClient.parse_trigger_path(path)
    assert expected == actual

def test_worker_pool_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    location = 'mussel'
    worker_pool = 'winkle'
    expected = 'projects/{project}/locations/{location}/workerPools/{worker_pool}'.format(project=project, location=location, worker_pool=worker_pool)
    actual = FunctionServiceClient.worker_pool_path(project, location, worker_pool)
    assert expected == actual

def test_parse_worker_pool_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'nautilus', 'location': 'scallop', 'worker_pool': 'abalone'}
    path = FunctionServiceClient.worker_pool_path(**expected)
    actual = FunctionServiceClient.parse_worker_pool_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = FunctionServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'clam'}
    path = FunctionServiceClient.common_billing_account_path(**expected)
    actual = FunctionServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = FunctionServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'octopus'}
    path = FunctionServiceClient.common_folder_path(**expected)
    actual = FunctionServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = FunctionServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'nudibranch'}
    path = FunctionServiceClient.common_organization_path(**expected)
    actual = FunctionServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = FunctionServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'mussel'}
    path = FunctionServiceClient.common_project_path(**expected)
    actual = FunctionServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = FunctionServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = FunctionServiceClient.common_location_path(**expected)
    actual = FunctionServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.FunctionServiceTransport, '_prep_wrapped_messages') as prep:
        client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.FunctionServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = FunctionServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_list_locations_rest_bad_request(transport: str='rest', request_type=locations_pb2.ListLocationsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/functions/sample3'}, request)
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
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/functions/sample3'}
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
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/functions/sample3'}, request)
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
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/functions/sample3'}
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
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/functions/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/functions/sample3'}
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

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_get_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = FunctionServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = FunctionServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(FunctionServiceClient, transports.FunctionServiceGrpcTransport), (FunctionServiceAsyncClient, transports.FunctionServiceGrpcAsyncIOTransport)])
def test_api_key_credentials(client_class, transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth._default, 'get_api_key_credentials', create=True) as get_api_key_credentials:
        mock_cred = mock.Mock()
        get_api_key_credentials.return_value = mock_cred
        options = client_options.ClientOptions()
        options.api_key = 'api_key'
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options)
            patched.assert_called_once_with(credentials=mock_cred, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)