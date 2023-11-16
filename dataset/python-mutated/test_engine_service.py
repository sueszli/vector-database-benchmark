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
from google.cloud.discoveryengine_v1alpha.services.engine_service import EngineServiceAsyncClient, EngineServiceClient, pagers, transports
from google.cloud.discoveryengine_v1alpha.types import common
from google.cloud.discoveryengine_v1alpha.types import engine
from google.cloud.discoveryengine_v1alpha.types import engine as gcd_engine
from google.cloud.discoveryengine_v1alpha.types import engine_service

def client_cert_source_callback():
    if False:
        while True:
            i = 10
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
    assert EngineServiceClient._get_default_mtls_endpoint(None) is None
    assert EngineServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert EngineServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert EngineServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert EngineServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert EngineServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(EngineServiceClient, 'grpc'), (EngineServiceAsyncClient, 'grpc_asyncio'), (EngineServiceClient, 'rest')])
def test_engine_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('discoveryengine.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://discoveryengine.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.EngineServiceGrpcTransport, 'grpc'), (transports.EngineServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.EngineServiceRestTransport, 'rest')])
def test_engine_service_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(EngineServiceClient, 'grpc'), (EngineServiceAsyncClient, 'grpc_asyncio'), (EngineServiceClient, 'rest')])
def test_engine_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('discoveryengine.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://discoveryengine.googleapis.com')

def test_engine_service_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = EngineServiceClient.get_transport_class()
    available_transports = [transports.EngineServiceGrpcTransport, transports.EngineServiceRestTransport]
    assert transport in available_transports
    transport = EngineServiceClient.get_transport_class('grpc')
    assert transport == transports.EngineServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(EngineServiceClient, transports.EngineServiceGrpcTransport, 'grpc'), (EngineServiceAsyncClient, transports.EngineServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (EngineServiceClient, transports.EngineServiceRestTransport, 'rest')])
@mock.patch.object(EngineServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EngineServiceClient))
@mock.patch.object(EngineServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EngineServiceAsyncClient))
def test_engine_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(EngineServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(EngineServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(EngineServiceClient, transports.EngineServiceGrpcTransport, 'grpc', 'true'), (EngineServiceAsyncClient, transports.EngineServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (EngineServiceClient, transports.EngineServiceGrpcTransport, 'grpc', 'false'), (EngineServiceAsyncClient, transports.EngineServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (EngineServiceClient, transports.EngineServiceRestTransport, 'rest', 'true'), (EngineServiceClient, transports.EngineServiceRestTransport, 'rest', 'false')])
@mock.patch.object(EngineServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EngineServiceClient))
@mock.patch.object(EngineServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EngineServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_engine_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [EngineServiceClient, EngineServiceAsyncClient])
@mock.patch.object(EngineServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EngineServiceClient))
@mock.patch.object(EngineServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EngineServiceAsyncClient))
def test_engine_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(EngineServiceClient, transports.EngineServiceGrpcTransport, 'grpc'), (EngineServiceAsyncClient, transports.EngineServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (EngineServiceClient, transports.EngineServiceRestTransport, 'rest')])
def test_engine_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(EngineServiceClient, transports.EngineServiceGrpcTransport, 'grpc', grpc_helpers), (EngineServiceAsyncClient, transports.EngineServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (EngineServiceClient, transports.EngineServiceRestTransport, 'rest', None)])
def test_engine_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_engine_service_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.discoveryengine_v1alpha.services.engine_service.transports.EngineServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = EngineServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(EngineServiceClient, transports.EngineServiceGrpcTransport, 'grpc', grpc_helpers), (EngineServiceAsyncClient, transports.EngineServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_engine_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('discoveryengine.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='discoveryengine.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [engine_service.CreateEngineRequest, dict])
def test_create_engine(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_engine), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_engine(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.CreateEngineRequest()
    assert isinstance(response, future.Future)

def test_create_engine_empty_call():
    if False:
        i = 10
        return i + 15
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_engine), '__call__') as call:
        client.create_engine()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.CreateEngineRequest()

@pytest.mark.asyncio
async def test_create_engine_async(transport: str='grpc_asyncio', request_type=engine_service.CreateEngineRequest):
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_engine), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_engine(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.CreateEngineRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_engine_async_from_dict():
    await test_create_engine_async(request_type=dict)

def test_create_engine_field_headers():
    if False:
        i = 10
        return i + 15
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = engine_service.CreateEngineRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_engine), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_engine(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_engine_field_headers_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = engine_service.CreateEngineRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_engine), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_engine(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_engine_flattened():
    if False:
        i = 10
        return i + 15
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_engine), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_engine(parent='parent_value', engine=gcd_engine.Engine(similar_documents_config=None), engine_id='engine_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].engine
        mock_val = gcd_engine.Engine(similar_documents_config=None)
        assert arg == mock_val
        arg = args[0].engine_id
        mock_val = 'engine_id_value'
        assert arg == mock_val

def test_create_engine_flattened_error():
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_engine(engine_service.CreateEngineRequest(), parent='parent_value', engine=gcd_engine.Engine(similar_documents_config=None), engine_id='engine_id_value')

@pytest.mark.asyncio
async def test_create_engine_flattened_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_engine), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_engine(parent='parent_value', engine=gcd_engine.Engine(similar_documents_config=None), engine_id='engine_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].engine
        mock_val = gcd_engine.Engine(similar_documents_config=None)
        assert arg == mock_val
        arg = args[0].engine_id
        mock_val = 'engine_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_engine_flattened_error_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_engine(engine_service.CreateEngineRequest(), parent='parent_value', engine=gcd_engine.Engine(similar_documents_config=None), engine_id='engine_id_value')

@pytest.mark.parametrize('request_type', [engine_service.DeleteEngineRequest, dict])
def test_delete_engine(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_engine), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_engine(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.DeleteEngineRequest()
    assert isinstance(response, future.Future)

def test_delete_engine_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_engine), '__call__') as call:
        client.delete_engine()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.DeleteEngineRequest()

@pytest.mark.asyncio
async def test_delete_engine_async(transport: str='grpc_asyncio', request_type=engine_service.DeleteEngineRequest):
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_engine), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_engine(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.DeleteEngineRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_engine_async_from_dict():
    await test_delete_engine_async(request_type=dict)

def test_delete_engine_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = engine_service.DeleteEngineRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_engine), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_engine(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_engine_field_headers_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = engine_service.DeleteEngineRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_engine), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_engine(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_engine_flattened():
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_engine), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_engine(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_engine_flattened_error():
    if False:
        i = 10
        return i + 15
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_engine(engine_service.DeleteEngineRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_engine_flattened_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_engine), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_engine(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_engine_flattened_error_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_engine(engine_service.DeleteEngineRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [engine_service.UpdateEngineRequest, dict])
def test_update_engine(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_engine), '__call__') as call:
        call.return_value = gcd_engine.Engine(name='name_value', display_name='display_name_value', data_store_ids=['data_store_ids_value'], solution_type=common.SolutionType.SOLUTION_TYPE_RECOMMENDATION, industry_vertical=common.IndustryVertical.GENERIC)
        response = client.update_engine(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.UpdateEngineRequest()
    assert isinstance(response, gcd_engine.Engine)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_store_ids == ['data_store_ids_value']
    assert response.solution_type == common.SolutionType.SOLUTION_TYPE_RECOMMENDATION
    assert response.industry_vertical == common.IndustryVertical.GENERIC

def test_update_engine_empty_call():
    if False:
        i = 10
        return i + 15
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_engine), '__call__') as call:
        client.update_engine()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.UpdateEngineRequest()

@pytest.mark.asyncio
async def test_update_engine_async(transport: str='grpc_asyncio', request_type=engine_service.UpdateEngineRequest):
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_engine), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_engine.Engine(name='name_value', display_name='display_name_value', data_store_ids=['data_store_ids_value'], solution_type=common.SolutionType.SOLUTION_TYPE_RECOMMENDATION, industry_vertical=common.IndustryVertical.GENERIC))
        response = await client.update_engine(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.UpdateEngineRequest()
    assert isinstance(response, gcd_engine.Engine)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_store_ids == ['data_store_ids_value']
    assert response.solution_type == common.SolutionType.SOLUTION_TYPE_RECOMMENDATION
    assert response.industry_vertical == common.IndustryVertical.GENERIC

@pytest.mark.asyncio
async def test_update_engine_async_from_dict():
    await test_update_engine_async(request_type=dict)

def test_update_engine_field_headers():
    if False:
        print('Hello World!')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = engine_service.UpdateEngineRequest()
    request.engine.name = 'name_value'
    with mock.patch.object(type(client.transport.update_engine), '__call__') as call:
        call.return_value = gcd_engine.Engine()
        client.update_engine(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'engine.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_engine_field_headers_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = engine_service.UpdateEngineRequest()
    request.engine.name = 'name_value'
    with mock.patch.object(type(client.transport.update_engine), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_engine.Engine())
        await client.update_engine(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'engine.name=name_value') in kw['metadata']

def test_update_engine_flattened():
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_engine), '__call__') as call:
        call.return_value = gcd_engine.Engine()
        client.update_engine(engine=gcd_engine.Engine(similar_documents_config=None), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].engine
        mock_val = gcd_engine.Engine(similar_documents_config=None)
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_engine_flattened_error():
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_engine(engine_service.UpdateEngineRequest(), engine=gcd_engine.Engine(similar_documents_config=None), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_engine_flattened_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_engine), '__call__') as call:
        call.return_value = gcd_engine.Engine()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_engine.Engine())
        response = await client.update_engine(engine=gcd_engine.Engine(similar_documents_config=None), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].engine
        mock_val = gcd_engine.Engine(similar_documents_config=None)
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_engine_flattened_error_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_engine(engine_service.UpdateEngineRequest(), engine=gcd_engine.Engine(similar_documents_config=None), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [engine_service.GetEngineRequest, dict])
def test_get_engine(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_engine), '__call__') as call:
        call.return_value = engine.Engine(name='name_value', display_name='display_name_value', data_store_ids=['data_store_ids_value'], solution_type=common.SolutionType.SOLUTION_TYPE_RECOMMENDATION, industry_vertical=common.IndustryVertical.GENERIC)
        response = client.get_engine(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.GetEngineRequest()
    assert isinstance(response, engine.Engine)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_store_ids == ['data_store_ids_value']
    assert response.solution_type == common.SolutionType.SOLUTION_TYPE_RECOMMENDATION
    assert response.industry_vertical == common.IndustryVertical.GENERIC

def test_get_engine_empty_call():
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_engine), '__call__') as call:
        client.get_engine()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.GetEngineRequest()

@pytest.mark.asyncio
async def test_get_engine_async(transport: str='grpc_asyncio', request_type=engine_service.GetEngineRequest):
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_engine), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(engine.Engine(name='name_value', display_name='display_name_value', data_store_ids=['data_store_ids_value'], solution_type=common.SolutionType.SOLUTION_TYPE_RECOMMENDATION, industry_vertical=common.IndustryVertical.GENERIC))
        response = await client.get_engine(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.GetEngineRequest()
    assert isinstance(response, engine.Engine)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_store_ids == ['data_store_ids_value']
    assert response.solution_type == common.SolutionType.SOLUTION_TYPE_RECOMMENDATION
    assert response.industry_vertical == common.IndustryVertical.GENERIC

@pytest.mark.asyncio
async def test_get_engine_async_from_dict():
    await test_get_engine_async(request_type=dict)

def test_get_engine_field_headers():
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = engine_service.GetEngineRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_engine), '__call__') as call:
        call.return_value = engine.Engine()
        client.get_engine(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_engine_field_headers_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = engine_service.GetEngineRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_engine), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(engine.Engine())
        await client.get_engine(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_engine_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_engine), '__call__') as call:
        call.return_value = engine.Engine()
        client.get_engine(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_engine_flattened_error():
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_engine(engine_service.GetEngineRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_engine_flattened_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_engine), '__call__') as call:
        call.return_value = engine.Engine()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(engine.Engine())
        response = await client.get_engine(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_engine_flattened_error_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_engine(engine_service.GetEngineRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [engine_service.ListEnginesRequest, dict])
def test_list_engines(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_engines), '__call__') as call:
        call.return_value = engine_service.ListEnginesResponse(next_page_token='next_page_token_value')
        response = client.list_engines(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.ListEnginesRequest()
    assert isinstance(response, pagers.ListEnginesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_engines_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_engines), '__call__') as call:
        client.list_engines()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.ListEnginesRequest()

@pytest.mark.asyncio
async def test_list_engines_async(transport: str='grpc_asyncio', request_type=engine_service.ListEnginesRequest):
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_engines), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(engine_service.ListEnginesResponse(next_page_token='next_page_token_value'))
        response = await client.list_engines(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.ListEnginesRequest()
    assert isinstance(response, pagers.ListEnginesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_engines_async_from_dict():
    await test_list_engines_async(request_type=dict)

def test_list_engines_field_headers():
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = engine_service.ListEnginesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_engines), '__call__') as call:
        call.return_value = engine_service.ListEnginesResponse()
        client.list_engines(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_engines_field_headers_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = engine_service.ListEnginesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_engines), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(engine_service.ListEnginesResponse())
        await client.list_engines(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_engines_flattened():
    if False:
        print('Hello World!')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_engines), '__call__') as call:
        call.return_value = engine_service.ListEnginesResponse()
        client.list_engines(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_engines_flattened_error():
    if False:
        i = 10
        return i + 15
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_engines(engine_service.ListEnginesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_engines_flattened_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_engines), '__call__') as call:
        call.return_value = engine_service.ListEnginesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(engine_service.ListEnginesResponse())
        response = await client.list_engines(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_engines_flattened_error_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_engines(engine_service.ListEnginesRequest(), parent='parent_value')

def test_list_engines_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_engines), '__call__') as call:
        call.side_effect = (engine_service.ListEnginesResponse(engines=[engine.Engine(), engine.Engine(), engine.Engine()], next_page_token='abc'), engine_service.ListEnginesResponse(engines=[], next_page_token='def'), engine_service.ListEnginesResponse(engines=[engine.Engine()], next_page_token='ghi'), engine_service.ListEnginesResponse(engines=[engine.Engine(), engine.Engine()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_engines(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, engine.Engine) for i in results))

def test_list_engines_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_engines), '__call__') as call:
        call.side_effect = (engine_service.ListEnginesResponse(engines=[engine.Engine(), engine.Engine(), engine.Engine()], next_page_token='abc'), engine_service.ListEnginesResponse(engines=[], next_page_token='def'), engine_service.ListEnginesResponse(engines=[engine.Engine()], next_page_token='ghi'), engine_service.ListEnginesResponse(engines=[engine.Engine(), engine.Engine()]), RuntimeError)
        pages = list(client.list_engines(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_engines_async_pager():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_engines), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (engine_service.ListEnginesResponse(engines=[engine.Engine(), engine.Engine(), engine.Engine()], next_page_token='abc'), engine_service.ListEnginesResponse(engines=[], next_page_token='def'), engine_service.ListEnginesResponse(engines=[engine.Engine()], next_page_token='ghi'), engine_service.ListEnginesResponse(engines=[engine.Engine(), engine.Engine()]), RuntimeError)
        async_pager = await client.list_engines(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, engine.Engine) for i in responses))

@pytest.mark.asyncio
async def test_list_engines_async_pages():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_engines), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (engine_service.ListEnginesResponse(engines=[engine.Engine(), engine.Engine(), engine.Engine()], next_page_token='abc'), engine_service.ListEnginesResponse(engines=[], next_page_token='def'), engine_service.ListEnginesResponse(engines=[engine.Engine()], next_page_token='ghi'), engine_service.ListEnginesResponse(engines=[engine.Engine(), engine.Engine()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_engines(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [engine_service.PauseEngineRequest, dict])
def test_pause_engine(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.pause_engine), '__call__') as call:
        call.return_value = engine.Engine(name='name_value', display_name='display_name_value', data_store_ids=['data_store_ids_value'], solution_type=common.SolutionType.SOLUTION_TYPE_RECOMMENDATION, industry_vertical=common.IndustryVertical.GENERIC)
        response = client.pause_engine(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.PauseEngineRequest()
    assert isinstance(response, engine.Engine)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_store_ids == ['data_store_ids_value']
    assert response.solution_type == common.SolutionType.SOLUTION_TYPE_RECOMMENDATION
    assert response.industry_vertical == common.IndustryVertical.GENERIC

def test_pause_engine_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.pause_engine), '__call__') as call:
        client.pause_engine()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.PauseEngineRequest()

@pytest.mark.asyncio
async def test_pause_engine_async(transport: str='grpc_asyncio', request_type=engine_service.PauseEngineRequest):
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.pause_engine), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(engine.Engine(name='name_value', display_name='display_name_value', data_store_ids=['data_store_ids_value'], solution_type=common.SolutionType.SOLUTION_TYPE_RECOMMENDATION, industry_vertical=common.IndustryVertical.GENERIC))
        response = await client.pause_engine(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.PauseEngineRequest()
    assert isinstance(response, engine.Engine)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_store_ids == ['data_store_ids_value']
    assert response.solution_type == common.SolutionType.SOLUTION_TYPE_RECOMMENDATION
    assert response.industry_vertical == common.IndustryVertical.GENERIC

@pytest.mark.asyncio
async def test_pause_engine_async_from_dict():
    await test_pause_engine_async(request_type=dict)

def test_pause_engine_field_headers():
    if False:
        print('Hello World!')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = engine_service.PauseEngineRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.pause_engine), '__call__') as call:
        call.return_value = engine.Engine()
        client.pause_engine(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_pause_engine_field_headers_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = engine_service.PauseEngineRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.pause_engine), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(engine.Engine())
        await client.pause_engine(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_pause_engine_flattened():
    if False:
        print('Hello World!')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.pause_engine), '__call__') as call:
        call.return_value = engine.Engine()
        client.pause_engine(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_pause_engine_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.pause_engine(engine_service.PauseEngineRequest(), name='name_value')

@pytest.mark.asyncio
async def test_pause_engine_flattened_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.pause_engine), '__call__') as call:
        call.return_value = engine.Engine()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(engine.Engine())
        response = await client.pause_engine(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_pause_engine_flattened_error_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.pause_engine(engine_service.PauseEngineRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [engine_service.ResumeEngineRequest, dict])
def test_resume_engine(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resume_engine), '__call__') as call:
        call.return_value = engine.Engine(name='name_value', display_name='display_name_value', data_store_ids=['data_store_ids_value'], solution_type=common.SolutionType.SOLUTION_TYPE_RECOMMENDATION, industry_vertical=common.IndustryVertical.GENERIC)
        response = client.resume_engine(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.ResumeEngineRequest()
    assert isinstance(response, engine.Engine)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_store_ids == ['data_store_ids_value']
    assert response.solution_type == common.SolutionType.SOLUTION_TYPE_RECOMMENDATION
    assert response.industry_vertical == common.IndustryVertical.GENERIC

def test_resume_engine_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.resume_engine), '__call__') as call:
        client.resume_engine()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.ResumeEngineRequest()

@pytest.mark.asyncio
async def test_resume_engine_async(transport: str='grpc_asyncio', request_type=engine_service.ResumeEngineRequest):
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.resume_engine), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(engine.Engine(name='name_value', display_name='display_name_value', data_store_ids=['data_store_ids_value'], solution_type=common.SolutionType.SOLUTION_TYPE_RECOMMENDATION, industry_vertical=common.IndustryVertical.GENERIC))
        response = await client.resume_engine(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.ResumeEngineRequest()
    assert isinstance(response, engine.Engine)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_store_ids == ['data_store_ids_value']
    assert response.solution_type == common.SolutionType.SOLUTION_TYPE_RECOMMENDATION
    assert response.industry_vertical == common.IndustryVertical.GENERIC

@pytest.mark.asyncio
async def test_resume_engine_async_from_dict():
    await test_resume_engine_async(request_type=dict)

def test_resume_engine_field_headers():
    if False:
        print('Hello World!')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = engine_service.ResumeEngineRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.resume_engine), '__call__') as call:
        call.return_value = engine.Engine()
        client.resume_engine(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_resume_engine_field_headers_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = engine_service.ResumeEngineRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.resume_engine), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(engine.Engine())
        await client.resume_engine(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_resume_engine_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.resume_engine), '__call__') as call:
        call.return_value = engine.Engine()
        client.resume_engine(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_resume_engine_flattened_error():
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.resume_engine(engine_service.ResumeEngineRequest(), name='name_value')

@pytest.mark.asyncio
async def test_resume_engine_flattened_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.resume_engine), '__call__') as call:
        call.return_value = engine.Engine()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(engine.Engine())
        response = await client.resume_engine(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_resume_engine_flattened_error_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.resume_engine(engine_service.ResumeEngineRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [engine_service.TuneEngineRequest, dict])
def test_tune_engine(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.tune_engine), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.tune_engine(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.TuneEngineRequest()
    assert isinstance(response, future.Future)

def test_tune_engine_empty_call():
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.tune_engine), '__call__') as call:
        client.tune_engine()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.TuneEngineRequest()

@pytest.mark.asyncio
async def test_tune_engine_async(transport: str='grpc_asyncio', request_type=engine_service.TuneEngineRequest):
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.tune_engine), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.tune_engine(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == engine_service.TuneEngineRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_tune_engine_async_from_dict():
    await test_tune_engine_async(request_type=dict)

def test_tune_engine_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = engine_service.TuneEngineRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.tune_engine), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.tune_engine(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_tune_engine_field_headers_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = engine_service.TuneEngineRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.tune_engine), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.tune_engine(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_tune_engine_flattened():
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.tune_engine), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.tune_engine(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_tune_engine_flattened_error():
    if False:
        print('Hello World!')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.tune_engine(engine_service.TuneEngineRequest(), name='name_value')

@pytest.mark.asyncio
async def test_tune_engine_flattened_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.tune_engine), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.tune_engine(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_tune_engine_flattened_error_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.tune_engine(engine_service.TuneEngineRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [engine_service.CreateEngineRequest, dict])
def test_create_engine_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/collections/sample3'}
    request_init['engine'] = {'similar_documents_config': {}, 'chat_engine_config': {'agent_creation_config': {'business': 'business_value', 'default_language_code': 'default_language_code_value', 'time_zone': 'time_zone_value'}, 'dialogflow_agent_to_link': 'dialogflow_agent_to_link_value'}, 'search_engine_config': {'search_tier': 1, 'search_add_ons': [1]}, 'media_recommendation_engine_config': {'type_': 'type__value', 'optimization_objective': 'optimization_objective_value', 'optimization_objective_config': {'target_field': 'target_field_value', 'target_field_value_float': 0.2523}, 'training_state': 1}, 'recommendation_metadata': {'serving_state': 1, 'data_state': 1, 'last_tune_time': {'seconds': 751, 'nanos': 543}, 'tuning_operation': 'tuning_operation_value'}, 'chat_engine_metadata': {'dialogflow_agent': 'dialogflow_agent_value'}, 'name': 'name_value', 'display_name': 'display_name_value', 'create_time': {}, 'update_time': {}, 'data_store_ids': ['data_store_ids_value1', 'data_store_ids_value2'], 'solution_type': 1, 'industry_vertical': 1, 'common_config': {'company_name': 'company_name_value'}}
    test_field = engine_service.CreateEngineRequest.meta.fields['engine']

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
    for (field, value) in request_init['engine'].items():
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
                for i in range(0, len(request_init['engine'][field])):
                    del request_init['engine'][field][i][subfield]
            else:
                del request_init['engine'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_engine(request)
    assert response.operation.name == 'operations/spam'

def test_create_engine_rest_required_fields(request_type=engine_service.CreateEngineRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EngineServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['engine_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'engineId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_engine._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'engineId' in jsonified_request
    assert jsonified_request['engineId'] == request_init['engine_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['engineId'] = 'engine_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_engine._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('engine_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'engineId' in jsonified_request
    assert jsonified_request['engineId'] == 'engine_id_value'
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_engine(request)
            expected_params = [('engineId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_engine_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EngineServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_engine._get_unset_required_fields({})
    assert set(unset_fields) == set(('engineId',)) & set(('parent', 'engine', 'engineId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_engine_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EngineServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EngineServiceRestInterceptor())
    client = EngineServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EngineServiceRestInterceptor, 'post_create_engine') as post, mock.patch.object(transports.EngineServiceRestInterceptor, 'pre_create_engine') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = engine_service.CreateEngineRequest.pb(engine_service.CreateEngineRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = engine_service.CreateEngineRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_engine(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_engine_rest_bad_request(transport: str='rest', request_type=engine_service.CreateEngineRequest):
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/collections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_engine(request)

def test_create_engine_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/collections/sample3'}
        mock_args = dict(parent='parent_value', engine=gcd_engine.Engine(similar_documents_config=None), engine_id='engine_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_engine(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{parent=projects/*/locations/*/collections/*}/engines' % client.transport._host, args[1])

def test_create_engine_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_engine(engine_service.CreateEngineRequest(), parent='parent_value', engine=gcd_engine.Engine(similar_documents_config=None), engine_id='engine_id_value')

def test_create_engine_rest_error():
    if False:
        i = 10
        return i + 15
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [engine_service.DeleteEngineRequest, dict])
def test_delete_engine_rest(request_type):
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_engine(request)
    assert response.operation.name == 'operations/spam'

def test_delete_engine_rest_required_fields(request_type=engine_service.DeleteEngineRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.EngineServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_engine._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_engine._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_engine(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_engine_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EngineServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_engine._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_engine_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.EngineServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EngineServiceRestInterceptor())
    client = EngineServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EngineServiceRestInterceptor, 'post_delete_engine') as post, mock.patch.object(transports.EngineServiceRestInterceptor, 'pre_delete_engine') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = engine_service.DeleteEngineRequest.pb(engine_service.DeleteEngineRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = engine_service.DeleteEngineRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_engine(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_engine_rest_bad_request(transport: str='rest', request_type=engine_service.DeleteEngineRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_engine(request)

def test_delete_engine_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_engine(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=projects/*/locations/*/collections/*/engines/*}' % client.transport._host, args[1])

def test_delete_engine_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_engine(engine_service.DeleteEngineRequest(), name='name_value')

def test_delete_engine_rest_error():
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [engine_service.UpdateEngineRequest, dict])
def test_update_engine_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'engine': {'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4'}}
    request_init['engine'] = {'similar_documents_config': {}, 'chat_engine_config': {'agent_creation_config': {'business': 'business_value', 'default_language_code': 'default_language_code_value', 'time_zone': 'time_zone_value'}, 'dialogflow_agent_to_link': 'dialogflow_agent_to_link_value'}, 'search_engine_config': {'search_tier': 1, 'search_add_ons': [1]}, 'media_recommendation_engine_config': {'type_': 'type__value', 'optimization_objective': 'optimization_objective_value', 'optimization_objective_config': {'target_field': 'target_field_value', 'target_field_value_float': 0.2523}, 'training_state': 1}, 'recommendation_metadata': {'serving_state': 1, 'data_state': 1, 'last_tune_time': {'seconds': 751, 'nanos': 543}, 'tuning_operation': 'tuning_operation_value'}, 'chat_engine_metadata': {'dialogflow_agent': 'dialogflow_agent_value'}, 'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4', 'display_name': 'display_name_value', 'create_time': {}, 'update_time': {}, 'data_store_ids': ['data_store_ids_value1', 'data_store_ids_value2'], 'solution_type': 1, 'industry_vertical': 1, 'common_config': {'company_name': 'company_name_value'}}
    test_field = engine_service.UpdateEngineRequest.meta.fields['engine']

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
    for (field, value) in request_init['engine'].items():
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
                for i in range(0, len(request_init['engine'][field])):
                    del request_init['engine'][field][i][subfield]
            else:
                del request_init['engine'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcd_engine.Engine(name='name_value', display_name='display_name_value', data_store_ids=['data_store_ids_value'], solution_type=common.SolutionType.SOLUTION_TYPE_RECOMMENDATION, industry_vertical=common.IndustryVertical.GENERIC)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcd_engine.Engine.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_engine(request)
    assert isinstance(response, gcd_engine.Engine)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_store_ids == ['data_store_ids_value']
    assert response.solution_type == common.SolutionType.SOLUTION_TYPE_RECOMMENDATION
    assert response.industry_vertical == common.IndustryVertical.GENERIC

def test_update_engine_rest_required_fields(request_type=engine_service.UpdateEngineRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.EngineServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_engine._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_engine._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcd_engine.Engine()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcd_engine.Engine.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_engine(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_engine_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EngineServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_engine._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('engine',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_engine_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EngineServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EngineServiceRestInterceptor())
    client = EngineServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EngineServiceRestInterceptor, 'post_update_engine') as post, mock.patch.object(transports.EngineServiceRestInterceptor, 'pre_update_engine') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = engine_service.UpdateEngineRequest.pb(engine_service.UpdateEngineRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcd_engine.Engine.to_json(gcd_engine.Engine())
        request = engine_service.UpdateEngineRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcd_engine.Engine()
        client.update_engine(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_engine_rest_bad_request(transport: str='rest', request_type=engine_service.UpdateEngineRequest):
    if False:
        print('Hello World!')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'engine': {'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_engine(request)

def test_update_engine_rest_flattened():
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcd_engine.Engine()
        sample_request = {'engine': {'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4'}}
        mock_args = dict(engine=gcd_engine.Engine(similar_documents_config=None), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcd_engine.Engine.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_engine(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{engine.name=projects/*/locations/*/collections/*/engines/*}' % client.transport._host, args[1])

def test_update_engine_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_engine(engine_service.UpdateEngineRequest(), engine=gcd_engine.Engine(similar_documents_config=None), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_engine_rest_error():
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [engine_service.GetEngineRequest, dict])
def test_get_engine_rest(request_type):
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = engine.Engine(name='name_value', display_name='display_name_value', data_store_ids=['data_store_ids_value'], solution_type=common.SolutionType.SOLUTION_TYPE_RECOMMENDATION, industry_vertical=common.IndustryVertical.GENERIC)
        response_value = Response()
        response_value.status_code = 200
        return_value = engine.Engine.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_engine(request)
    assert isinstance(response, engine.Engine)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_store_ids == ['data_store_ids_value']
    assert response.solution_type == common.SolutionType.SOLUTION_TYPE_RECOMMENDATION
    assert response.industry_vertical == common.IndustryVertical.GENERIC

def test_get_engine_rest_required_fields(request_type=engine_service.GetEngineRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EngineServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_engine._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_engine._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = engine.Engine()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = engine.Engine.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_engine(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_engine_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.EngineServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_engine._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_engine_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EngineServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EngineServiceRestInterceptor())
    client = EngineServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EngineServiceRestInterceptor, 'post_get_engine') as post, mock.patch.object(transports.EngineServiceRestInterceptor, 'pre_get_engine') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = engine_service.GetEngineRequest.pb(engine_service.GetEngineRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = engine.Engine.to_json(engine.Engine())
        request = engine_service.GetEngineRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = engine.Engine()
        client.get_engine(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_engine_rest_bad_request(transport: str='rest', request_type=engine_service.GetEngineRequest):
    if False:
        print('Hello World!')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_engine(request)

def test_get_engine_rest_flattened():
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = engine.Engine()
        sample_request = {'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = engine.Engine.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_engine(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=projects/*/locations/*/collections/*/engines/*}' % client.transport._host, args[1])

def test_get_engine_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_engine(engine_service.GetEngineRequest(), name='name_value')

def test_get_engine_rest_error():
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [engine_service.ListEnginesRequest, dict])
def test_list_engines_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/collections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = engine_service.ListEnginesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = engine_service.ListEnginesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_engines(request)
    assert isinstance(response, pagers.ListEnginesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_engines_rest_required_fields(request_type=engine_service.ListEnginesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.EngineServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_engines._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_engines._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = engine_service.ListEnginesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = engine_service.ListEnginesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_engines(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_engines_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EngineServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_engines._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_engines_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.EngineServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EngineServiceRestInterceptor())
    client = EngineServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EngineServiceRestInterceptor, 'post_list_engines') as post, mock.patch.object(transports.EngineServiceRestInterceptor, 'pre_list_engines') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = engine_service.ListEnginesRequest.pb(engine_service.ListEnginesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = engine_service.ListEnginesResponse.to_json(engine_service.ListEnginesResponse())
        request = engine_service.ListEnginesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = engine_service.ListEnginesResponse()
        client.list_engines(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_engines_rest_bad_request(transport: str='rest', request_type=engine_service.ListEnginesRequest):
    if False:
        print('Hello World!')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/collections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_engines(request)

def test_list_engines_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = engine_service.ListEnginesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/collections/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = engine_service.ListEnginesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_engines(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{parent=projects/*/locations/*/collections/*}/engines' % client.transport._host, args[1])

def test_list_engines_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_engines(engine_service.ListEnginesRequest(), parent='parent_value')

def test_list_engines_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (engine_service.ListEnginesResponse(engines=[engine.Engine(), engine.Engine(), engine.Engine()], next_page_token='abc'), engine_service.ListEnginesResponse(engines=[], next_page_token='def'), engine_service.ListEnginesResponse(engines=[engine.Engine()], next_page_token='ghi'), engine_service.ListEnginesResponse(engines=[engine.Engine(), engine.Engine()]))
        response = response + response
        response = tuple((engine_service.ListEnginesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/collections/sample3'}
        pager = client.list_engines(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, engine.Engine) for i in results))
        pages = list(client.list_engines(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [engine_service.PauseEngineRequest, dict])
def test_pause_engine_rest(request_type):
    if False:
        print('Hello World!')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = engine.Engine(name='name_value', display_name='display_name_value', data_store_ids=['data_store_ids_value'], solution_type=common.SolutionType.SOLUTION_TYPE_RECOMMENDATION, industry_vertical=common.IndustryVertical.GENERIC)
        response_value = Response()
        response_value.status_code = 200
        return_value = engine.Engine.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.pause_engine(request)
    assert isinstance(response, engine.Engine)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_store_ids == ['data_store_ids_value']
    assert response.solution_type == common.SolutionType.SOLUTION_TYPE_RECOMMENDATION
    assert response.industry_vertical == common.IndustryVertical.GENERIC

def test_pause_engine_rest_required_fields(request_type=engine_service.PauseEngineRequest):
    if False:
        return 10
    transport_class = transports.EngineServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).pause_engine._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).pause_engine._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = engine.Engine()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = engine.Engine.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.pause_engine(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_pause_engine_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.EngineServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.pause_engine._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_pause_engine_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.EngineServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EngineServiceRestInterceptor())
    client = EngineServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EngineServiceRestInterceptor, 'post_pause_engine') as post, mock.patch.object(transports.EngineServiceRestInterceptor, 'pre_pause_engine') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = engine_service.PauseEngineRequest.pb(engine_service.PauseEngineRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = engine.Engine.to_json(engine.Engine())
        request = engine_service.PauseEngineRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = engine.Engine()
        client.pause_engine(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_pause_engine_rest_bad_request(transport: str='rest', request_type=engine_service.PauseEngineRequest):
    if False:
        print('Hello World!')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.pause_engine(request)

def test_pause_engine_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = engine.Engine()
        sample_request = {'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = engine.Engine.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.pause_engine(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=projects/*/locations/*/collections/*/engines/*}:pause' % client.transport._host, args[1])

def test_pause_engine_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.pause_engine(engine_service.PauseEngineRequest(), name='name_value')

def test_pause_engine_rest_error():
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [engine_service.ResumeEngineRequest, dict])
def test_resume_engine_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = engine.Engine(name='name_value', display_name='display_name_value', data_store_ids=['data_store_ids_value'], solution_type=common.SolutionType.SOLUTION_TYPE_RECOMMENDATION, industry_vertical=common.IndustryVertical.GENERIC)
        response_value = Response()
        response_value.status_code = 200
        return_value = engine.Engine.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.resume_engine(request)
    assert isinstance(response, engine.Engine)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.data_store_ids == ['data_store_ids_value']
    assert response.solution_type == common.SolutionType.SOLUTION_TYPE_RECOMMENDATION
    assert response.industry_vertical == common.IndustryVertical.GENERIC

def test_resume_engine_rest_required_fields(request_type=engine_service.ResumeEngineRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EngineServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resume_engine._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resume_engine._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = engine.Engine()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = engine.Engine.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.resume_engine(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_resume_engine_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.EngineServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.resume_engine._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_resume_engine_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EngineServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EngineServiceRestInterceptor())
    client = EngineServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EngineServiceRestInterceptor, 'post_resume_engine') as post, mock.patch.object(transports.EngineServiceRestInterceptor, 'pre_resume_engine') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = engine_service.ResumeEngineRequest.pb(engine_service.ResumeEngineRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = engine.Engine.to_json(engine.Engine())
        request = engine_service.ResumeEngineRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = engine.Engine()
        client.resume_engine(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_resume_engine_rest_bad_request(transport: str='rest', request_type=engine_service.ResumeEngineRequest):
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.resume_engine(request)

def test_resume_engine_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = engine.Engine()
        sample_request = {'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = engine.Engine.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.resume_engine(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=projects/*/locations/*/collections/*/engines/*}:resume' % client.transport._host, args[1])

def test_resume_engine_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.resume_engine(engine_service.ResumeEngineRequest(), name='name_value')

def test_resume_engine_rest_error():
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [engine_service.TuneEngineRequest, dict])
def test_tune_engine_rest(request_type):
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.tune_engine(request)
    assert response.operation.name == 'operations/spam'

def test_tune_engine_rest_required_fields(request_type=engine_service.TuneEngineRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EngineServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).tune_engine._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).tune_engine._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.tune_engine(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_tune_engine_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.EngineServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.tune_engine._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_tune_engine_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EngineServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EngineServiceRestInterceptor())
    client = EngineServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EngineServiceRestInterceptor, 'post_tune_engine') as post, mock.patch.object(transports.EngineServiceRestInterceptor, 'pre_tune_engine') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = engine_service.TuneEngineRequest.pb(engine_service.TuneEngineRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = engine_service.TuneEngineRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.tune_engine(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_tune_engine_rest_bad_request(transport: str='rest', request_type=engine_service.TuneEngineRequest):
    if False:
        i = 10
        return i + 15
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.tune_engine(request)

def test_tune_engine_rest_flattened():
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/collections/sample3/engines/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.tune_engine(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=projects/*/locations/*/collections/*/engines/*}:tune' % client.transport._host, args[1])

def test_tune_engine_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.tune_engine(engine_service.TuneEngineRequest(), name='name_value')

def test_tune_engine_rest_error():
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.EngineServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.EngineServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EngineServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.EngineServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = EngineServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = EngineServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.EngineServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EngineServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EngineServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = EngineServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EngineServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.EngineServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.EngineServiceGrpcTransport, transports.EngineServiceGrpcAsyncIOTransport, transports.EngineServiceRestTransport])
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
    transport = EngineServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.EngineServiceGrpcTransport)

def test_engine_service_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.EngineServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_engine_service_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.discoveryengine_v1alpha.services.engine_service.transports.EngineServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.EngineServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_engine', 'delete_engine', 'update_engine', 'get_engine', 'list_engines', 'pause_engine', 'resume_engine', 'tune_engine', 'get_operation', 'list_operations')
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

def test_engine_service_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.discoveryengine_v1alpha.services.engine_service.transports.EngineServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.EngineServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_engine_service_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.discoveryengine_v1alpha.services.engine_service.transports.EngineServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.EngineServiceTransport()
        adc.assert_called_once()

def test_engine_service_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        EngineServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.EngineServiceGrpcTransport, transports.EngineServiceGrpcAsyncIOTransport])
def test_engine_service_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.EngineServiceGrpcTransport, transports.EngineServiceGrpcAsyncIOTransport, transports.EngineServiceRestTransport])
def test_engine_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.EngineServiceGrpcTransport, grpc_helpers), (transports.EngineServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_engine_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('discoveryengine.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='discoveryengine.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.EngineServiceGrpcTransport, transports.EngineServiceGrpcAsyncIOTransport])
def test_engine_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_engine_service_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.EngineServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_engine_service_rest_lro_client():
    if False:
        i = 10
        return i + 15
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_engine_service_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='discoveryengine.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('discoveryengine.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://discoveryengine.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_engine_service_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='discoveryengine.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('discoveryengine.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://discoveryengine.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_engine_service_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = EngineServiceClient(credentials=creds1, transport=transport_name)
    client2 = EngineServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_engine._session
    session2 = client2.transport.create_engine._session
    assert session1 != session2
    session1 = client1.transport.delete_engine._session
    session2 = client2.transport.delete_engine._session
    assert session1 != session2
    session1 = client1.transport.update_engine._session
    session2 = client2.transport.update_engine._session
    assert session1 != session2
    session1 = client1.transport.get_engine._session
    session2 = client2.transport.get_engine._session
    assert session1 != session2
    session1 = client1.transport.list_engines._session
    session2 = client2.transport.list_engines._session
    assert session1 != session2
    session1 = client1.transport.pause_engine._session
    session2 = client2.transport.pause_engine._session
    assert session1 != session2
    session1 = client1.transport.resume_engine._session
    session2 = client2.transport.resume_engine._session
    assert session1 != session2
    session1 = client1.transport.tune_engine._session
    session2 = client2.transport.tune_engine._session
    assert session1 != session2

def test_engine_service_grpc_transport_channel():
    if False:
        print('Hello World!')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.EngineServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_engine_service_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.EngineServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.EngineServiceGrpcTransport, transports.EngineServiceGrpcAsyncIOTransport])
def test_engine_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.EngineServiceGrpcTransport, transports.EngineServiceGrpcAsyncIOTransport])
def test_engine_service_transport_channel_mtls_with_adc(transport_class):
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

def test_engine_service_grpc_lro_client():
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_engine_service_grpc_lro_async_client():
    if False:
        while True:
            i = 10
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_collection_path():
    if False:
        return 10
    project = 'squid'
    location = 'clam'
    collection = 'whelk'
    expected = 'projects/{project}/locations/{location}/collections/{collection}'.format(project=project, location=location, collection=collection)
    actual = EngineServiceClient.collection_path(project, location, collection)
    assert expected == actual

def test_parse_collection_path():
    if False:
        return 10
    expected = {'project': 'octopus', 'location': 'oyster', 'collection': 'nudibranch'}
    path = EngineServiceClient.collection_path(**expected)
    actual = EngineServiceClient.parse_collection_path(path)
    assert expected == actual

def test_engine_path():
    if False:
        return 10
    project = 'cuttlefish'
    location = 'mussel'
    collection = 'winkle'
    engine = 'nautilus'
    expected = 'projects/{project}/locations/{location}/collections/{collection}/engines/{engine}'.format(project=project, location=location, collection=collection, engine=engine)
    actual = EngineServiceClient.engine_path(project, location, collection, engine)
    assert expected == actual

def test_parse_engine_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'scallop', 'location': 'abalone', 'collection': 'squid', 'engine': 'clam'}
    path = EngineServiceClient.engine_path(**expected)
    actual = EngineServiceClient.parse_engine_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = EngineServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'octopus'}
    path = EngineServiceClient.common_billing_account_path(**expected)
    actual = EngineServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = EngineServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'nudibranch'}
    path = EngineServiceClient.common_folder_path(**expected)
    actual = EngineServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = EngineServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'mussel'}
    path = EngineServiceClient.common_organization_path(**expected)
    actual = EngineServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = EngineServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nautilus'}
    path = EngineServiceClient.common_project_path(**expected)
    actual = EngineServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = EngineServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'squid', 'location': 'clam'}
    path = EngineServiceClient.common_location_path(**expected)
    actual = EngineServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.EngineServiceTransport, '_prep_wrapped_messages') as prep:
        client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.EngineServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = EngineServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        return 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2/collections/sample3/dataStores/sample4/branches/sample5/operations/sample6'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.GetOperationRequest, dict])
def test_get_operation_rest(request_type):
    if False:
        while True:
            i = 10
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/collections/sample3/dataStores/sample4/branches/sample5/operations/sample6'}
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
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2/collections/sample3/dataStores/sample4/branches/sample5'}, request)
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
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/collections/sample3/dataStores/sample4/branches/sample5'}
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
        for i in range(10):
            print('nop')
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = EngineServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        while True:
            i = 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = EngineServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(EngineServiceClient, transports.EngineServiceGrpcTransport), (EngineServiceAsyncClient, transports.EngineServiceGrpcAsyncIOTransport)])
def test_api_key_credentials(client_class, transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth._default, 'get_api_key_credentials', create=True) as get_api_key_credentials:
        mock_cred = mock.Mock()
        get_api_key_credentials.return_value = mock_cred
        options = client_options.ClientOptions()
        options.api_key = 'api_key'
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options)
            patched.assert_called_once_with(credentials=mock_cred, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)