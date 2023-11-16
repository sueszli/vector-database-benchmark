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
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.dialogflow_v2.services.conversation_models import ConversationModelsAsyncClient, ConversationModelsClient, pagers, transports
from google.cloud.dialogflow_v2.types import conversation_model as gcd_conversation_model
from google.cloud.dialogflow_v2.types import conversation_model

def client_cert_source_callback():
    if False:
        print('Hello World!')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        i = 10
        return i + 15
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
    assert ConversationModelsClient._get_default_mtls_endpoint(None) is None
    assert ConversationModelsClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert ConversationModelsClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert ConversationModelsClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert ConversationModelsClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert ConversationModelsClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(ConversationModelsClient, 'grpc'), (ConversationModelsAsyncClient, 'grpc_asyncio'), (ConversationModelsClient, 'rest')])
def test_conversation_models_client_from_service_account_info(client_class, transport_name):
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

@pytest.mark.parametrize('transport_class,transport_name', [(transports.ConversationModelsGrpcTransport, 'grpc'), (transports.ConversationModelsGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.ConversationModelsRestTransport, 'rest')])
def test_conversation_models_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(ConversationModelsClient, 'grpc'), (ConversationModelsAsyncClient, 'grpc_asyncio'), (ConversationModelsClient, 'rest')])
def test_conversation_models_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

def test_conversation_models_client_get_transport_class():
    if False:
        for i in range(10):
            print('nop')
    transport = ConversationModelsClient.get_transport_class()
    available_transports = [transports.ConversationModelsGrpcTransport, transports.ConversationModelsRestTransport]
    assert transport in available_transports
    transport = ConversationModelsClient.get_transport_class('grpc')
    assert transport == transports.ConversationModelsGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ConversationModelsClient, transports.ConversationModelsGrpcTransport, 'grpc'), (ConversationModelsAsyncClient, transports.ConversationModelsGrpcAsyncIOTransport, 'grpc_asyncio'), (ConversationModelsClient, transports.ConversationModelsRestTransport, 'rest')])
@mock.patch.object(ConversationModelsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ConversationModelsClient))
@mock.patch.object(ConversationModelsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ConversationModelsAsyncClient))
def test_conversation_models_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(ConversationModelsClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(ConversationModelsClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(ConversationModelsClient, transports.ConversationModelsGrpcTransport, 'grpc', 'true'), (ConversationModelsAsyncClient, transports.ConversationModelsGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (ConversationModelsClient, transports.ConversationModelsGrpcTransport, 'grpc', 'false'), (ConversationModelsAsyncClient, transports.ConversationModelsGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (ConversationModelsClient, transports.ConversationModelsRestTransport, 'rest', 'true'), (ConversationModelsClient, transports.ConversationModelsRestTransport, 'rest', 'false')])
@mock.patch.object(ConversationModelsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ConversationModelsClient))
@mock.patch.object(ConversationModelsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ConversationModelsAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_conversation_models_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [ConversationModelsClient, ConversationModelsAsyncClient])
@mock.patch.object(ConversationModelsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ConversationModelsClient))
@mock.patch.object(ConversationModelsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ConversationModelsAsyncClient))
def test_conversation_models_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ConversationModelsClient, transports.ConversationModelsGrpcTransport, 'grpc'), (ConversationModelsAsyncClient, transports.ConversationModelsGrpcAsyncIOTransport, 'grpc_asyncio'), (ConversationModelsClient, transports.ConversationModelsRestTransport, 'rest')])
def test_conversation_models_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ConversationModelsClient, transports.ConversationModelsGrpcTransport, 'grpc', grpc_helpers), (ConversationModelsAsyncClient, transports.ConversationModelsGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (ConversationModelsClient, transports.ConversationModelsRestTransport, 'rest', None)])
def test_conversation_models_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_conversation_models_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.dialogflow_v2.services.conversation_models.transports.ConversationModelsGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = ConversationModelsClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ConversationModelsClient, transports.ConversationModelsGrpcTransport, 'grpc', grpc_helpers), (ConversationModelsAsyncClient, transports.ConversationModelsGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_conversation_models_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
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

@pytest.mark.parametrize('request_type', [gcd_conversation_model.CreateConversationModelRequest, dict])
def test_create_conversation_model(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_conversation_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_conversation_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_conversation_model.CreateConversationModelRequest()
    assert isinstance(response, future.Future)

def test_create_conversation_model_empty_call():
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_conversation_model), '__call__') as call:
        client.create_conversation_model()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_conversation_model.CreateConversationModelRequest()

@pytest.mark.asyncio
async def test_create_conversation_model_async(transport: str='grpc_asyncio', request_type=gcd_conversation_model.CreateConversationModelRequest):
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_conversation_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_conversation_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_conversation_model.CreateConversationModelRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_conversation_model_async_from_dict():
    await test_create_conversation_model_async(request_type=dict)

def test_create_conversation_model_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_conversation_model.CreateConversationModelRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_conversation_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_conversation_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_conversation_model_field_headers_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_conversation_model.CreateConversationModelRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_conversation_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_conversation_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_conversation_model_flattened():
    if False:
        i = 10
        return i + 15
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_conversation_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_conversation_model(parent='parent_value', conversation_model=gcd_conversation_model.ConversationModel(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].conversation_model
        mock_val = gcd_conversation_model.ConversationModel(name='name_value')
        assert arg == mock_val

def test_create_conversation_model_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_conversation_model(gcd_conversation_model.CreateConversationModelRequest(), parent='parent_value', conversation_model=gcd_conversation_model.ConversationModel(name='name_value'))

@pytest.mark.asyncio
async def test_create_conversation_model_flattened_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_conversation_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_conversation_model(parent='parent_value', conversation_model=gcd_conversation_model.ConversationModel(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].conversation_model
        mock_val = gcd_conversation_model.ConversationModel(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_conversation_model_flattened_error_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_conversation_model(gcd_conversation_model.CreateConversationModelRequest(), parent='parent_value', conversation_model=gcd_conversation_model.ConversationModel(name='name_value'))

@pytest.mark.parametrize('request_type', [conversation_model.GetConversationModelRequest, dict])
def test_get_conversation_model(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_conversation_model), '__call__') as call:
        call.return_value = conversation_model.ConversationModel(name='name_value', display_name='display_name_value', state=conversation_model.ConversationModel.State.CREATING, language_code='language_code_value')
        response = client.get_conversation_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.GetConversationModelRequest()
    assert isinstance(response, conversation_model.ConversationModel)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == conversation_model.ConversationModel.State.CREATING
    assert response.language_code == 'language_code_value'

def test_get_conversation_model_empty_call():
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_conversation_model), '__call__') as call:
        client.get_conversation_model()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.GetConversationModelRequest()

@pytest.mark.asyncio
async def test_get_conversation_model_async(transport: str='grpc_asyncio', request_type=conversation_model.GetConversationModelRequest):
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_conversation_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(conversation_model.ConversationModel(name='name_value', display_name='display_name_value', state=conversation_model.ConversationModel.State.CREATING, language_code='language_code_value'))
        response = await client.get_conversation_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.GetConversationModelRequest()
    assert isinstance(response, conversation_model.ConversationModel)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == conversation_model.ConversationModel.State.CREATING
    assert response.language_code == 'language_code_value'

@pytest.mark.asyncio
async def test_get_conversation_model_async_from_dict():
    await test_get_conversation_model_async(request_type=dict)

def test_get_conversation_model_field_headers():
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_model.GetConversationModelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_conversation_model), '__call__') as call:
        call.return_value = conversation_model.ConversationModel()
        client.get_conversation_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_conversation_model_field_headers_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_model.GetConversationModelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_conversation_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(conversation_model.ConversationModel())
        await client.get_conversation_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_conversation_model_flattened():
    if False:
        print('Hello World!')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_conversation_model), '__call__') as call:
        call.return_value = conversation_model.ConversationModel()
        client.get_conversation_model(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_conversation_model_flattened_error():
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_conversation_model(conversation_model.GetConversationModelRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_conversation_model_flattened_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_conversation_model), '__call__') as call:
        call.return_value = conversation_model.ConversationModel()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(conversation_model.ConversationModel())
        response = await client.get_conversation_model(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_conversation_model_flattened_error_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_conversation_model(conversation_model.GetConversationModelRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [conversation_model.ListConversationModelsRequest, dict])
def test_list_conversation_models(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_conversation_models), '__call__') as call:
        call.return_value = conversation_model.ListConversationModelsResponse(next_page_token='next_page_token_value')
        response = client.list_conversation_models(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.ListConversationModelsRequest()
    assert isinstance(response, pagers.ListConversationModelsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_conversation_models_empty_call():
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_conversation_models), '__call__') as call:
        client.list_conversation_models()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.ListConversationModelsRequest()

@pytest.mark.asyncio
async def test_list_conversation_models_async(transport: str='grpc_asyncio', request_type=conversation_model.ListConversationModelsRequest):
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_conversation_models), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(conversation_model.ListConversationModelsResponse(next_page_token='next_page_token_value'))
        response = await client.list_conversation_models(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.ListConversationModelsRequest()
    assert isinstance(response, pagers.ListConversationModelsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_conversation_models_async_from_dict():
    await test_list_conversation_models_async(request_type=dict)

def test_list_conversation_models_field_headers():
    if False:
        i = 10
        return i + 15
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_model.ListConversationModelsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_conversation_models), '__call__') as call:
        call.return_value = conversation_model.ListConversationModelsResponse()
        client.list_conversation_models(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_conversation_models_field_headers_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_model.ListConversationModelsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_conversation_models), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(conversation_model.ListConversationModelsResponse())
        await client.list_conversation_models(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_conversation_models_flattened():
    if False:
        i = 10
        return i + 15
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_conversation_models), '__call__') as call:
        call.return_value = conversation_model.ListConversationModelsResponse()
        client.list_conversation_models(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_conversation_models_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_conversation_models(conversation_model.ListConversationModelsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_conversation_models_flattened_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_conversation_models), '__call__') as call:
        call.return_value = conversation_model.ListConversationModelsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(conversation_model.ListConversationModelsResponse())
        response = await client.list_conversation_models(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_conversation_models_flattened_error_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_conversation_models(conversation_model.ListConversationModelsRequest(), parent='parent_value')

def test_list_conversation_models_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_conversation_models), '__call__') as call:
        call.side_effect = (conversation_model.ListConversationModelsResponse(conversation_models=[conversation_model.ConversationModel(), conversation_model.ConversationModel(), conversation_model.ConversationModel()], next_page_token='abc'), conversation_model.ListConversationModelsResponse(conversation_models=[], next_page_token='def'), conversation_model.ListConversationModelsResponse(conversation_models=[conversation_model.ConversationModel()], next_page_token='ghi'), conversation_model.ListConversationModelsResponse(conversation_models=[conversation_model.ConversationModel(), conversation_model.ConversationModel()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_conversation_models(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, conversation_model.ConversationModel) for i in results))

def test_list_conversation_models_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_conversation_models), '__call__') as call:
        call.side_effect = (conversation_model.ListConversationModelsResponse(conversation_models=[conversation_model.ConversationModel(), conversation_model.ConversationModel(), conversation_model.ConversationModel()], next_page_token='abc'), conversation_model.ListConversationModelsResponse(conversation_models=[], next_page_token='def'), conversation_model.ListConversationModelsResponse(conversation_models=[conversation_model.ConversationModel()], next_page_token='ghi'), conversation_model.ListConversationModelsResponse(conversation_models=[conversation_model.ConversationModel(), conversation_model.ConversationModel()]), RuntimeError)
        pages = list(client.list_conversation_models(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_conversation_models_async_pager():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_conversation_models), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (conversation_model.ListConversationModelsResponse(conversation_models=[conversation_model.ConversationModel(), conversation_model.ConversationModel(), conversation_model.ConversationModel()], next_page_token='abc'), conversation_model.ListConversationModelsResponse(conversation_models=[], next_page_token='def'), conversation_model.ListConversationModelsResponse(conversation_models=[conversation_model.ConversationModel()], next_page_token='ghi'), conversation_model.ListConversationModelsResponse(conversation_models=[conversation_model.ConversationModel(), conversation_model.ConversationModel()]), RuntimeError)
        async_pager = await client.list_conversation_models(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, conversation_model.ConversationModel) for i in responses))

@pytest.mark.asyncio
async def test_list_conversation_models_async_pages():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_conversation_models), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (conversation_model.ListConversationModelsResponse(conversation_models=[conversation_model.ConversationModel(), conversation_model.ConversationModel(), conversation_model.ConversationModel()], next_page_token='abc'), conversation_model.ListConversationModelsResponse(conversation_models=[], next_page_token='def'), conversation_model.ListConversationModelsResponse(conversation_models=[conversation_model.ConversationModel()], next_page_token='ghi'), conversation_model.ListConversationModelsResponse(conversation_models=[conversation_model.ConversationModel(), conversation_model.ConversationModel()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_conversation_models(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [conversation_model.DeleteConversationModelRequest, dict])
def test_delete_conversation_model(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_conversation_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_conversation_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.DeleteConversationModelRequest()
    assert isinstance(response, future.Future)

def test_delete_conversation_model_empty_call():
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_conversation_model), '__call__') as call:
        client.delete_conversation_model()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.DeleteConversationModelRequest()

@pytest.mark.asyncio
async def test_delete_conversation_model_async(transport: str='grpc_asyncio', request_type=conversation_model.DeleteConversationModelRequest):
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_conversation_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_conversation_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.DeleteConversationModelRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_conversation_model_async_from_dict():
    await test_delete_conversation_model_async(request_type=dict)

def test_delete_conversation_model_field_headers():
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_model.DeleteConversationModelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_conversation_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_conversation_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_conversation_model_field_headers_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_model.DeleteConversationModelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_conversation_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_conversation_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_conversation_model_flattened():
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_conversation_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_conversation_model(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_conversation_model_flattened_error():
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_conversation_model(conversation_model.DeleteConversationModelRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_conversation_model_flattened_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_conversation_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_conversation_model(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_conversation_model_flattened_error_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_conversation_model(conversation_model.DeleteConversationModelRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [conversation_model.DeployConversationModelRequest, dict])
def test_deploy_conversation_model(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.deploy_conversation_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.deploy_conversation_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.DeployConversationModelRequest()
    assert isinstance(response, future.Future)

def test_deploy_conversation_model_empty_call():
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.deploy_conversation_model), '__call__') as call:
        client.deploy_conversation_model()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.DeployConversationModelRequest()

@pytest.mark.asyncio
async def test_deploy_conversation_model_async(transport: str='grpc_asyncio', request_type=conversation_model.DeployConversationModelRequest):
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.deploy_conversation_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.deploy_conversation_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.DeployConversationModelRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_deploy_conversation_model_async_from_dict():
    await test_deploy_conversation_model_async(request_type=dict)

def test_deploy_conversation_model_field_headers():
    if False:
        i = 10
        return i + 15
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_model.DeployConversationModelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.deploy_conversation_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.deploy_conversation_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_deploy_conversation_model_field_headers_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_model.DeployConversationModelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.deploy_conversation_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.deploy_conversation_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [conversation_model.UndeployConversationModelRequest, dict])
def test_undeploy_conversation_model(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.undeploy_conversation_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.undeploy_conversation_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.UndeployConversationModelRequest()
    assert isinstance(response, future.Future)

def test_undeploy_conversation_model_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.undeploy_conversation_model), '__call__') as call:
        client.undeploy_conversation_model()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.UndeployConversationModelRequest()

@pytest.mark.asyncio
async def test_undeploy_conversation_model_async(transport: str='grpc_asyncio', request_type=conversation_model.UndeployConversationModelRequest):
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.undeploy_conversation_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.undeploy_conversation_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.UndeployConversationModelRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_undeploy_conversation_model_async_from_dict():
    await test_undeploy_conversation_model_async(request_type=dict)

def test_undeploy_conversation_model_field_headers():
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_model.UndeployConversationModelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.undeploy_conversation_model), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.undeploy_conversation_model(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_undeploy_conversation_model_field_headers_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_model.UndeployConversationModelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.undeploy_conversation_model), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.undeploy_conversation_model(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [conversation_model.GetConversationModelEvaluationRequest, dict])
def test_get_conversation_model_evaluation(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_conversation_model_evaluation), '__call__') as call:
        call.return_value = conversation_model.ConversationModelEvaluation(name='name_value', display_name='display_name_value', raw_human_eval_template_csv='raw_human_eval_template_csv_value')
        response = client.get_conversation_model_evaluation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.GetConversationModelEvaluationRequest()
    assert isinstance(response, conversation_model.ConversationModelEvaluation)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.raw_human_eval_template_csv == 'raw_human_eval_template_csv_value'

def test_get_conversation_model_evaluation_empty_call():
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_conversation_model_evaluation), '__call__') as call:
        client.get_conversation_model_evaluation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.GetConversationModelEvaluationRequest()

@pytest.mark.asyncio
async def test_get_conversation_model_evaluation_async(transport: str='grpc_asyncio', request_type=conversation_model.GetConversationModelEvaluationRequest):
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_conversation_model_evaluation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(conversation_model.ConversationModelEvaluation(name='name_value', display_name='display_name_value', raw_human_eval_template_csv='raw_human_eval_template_csv_value'))
        response = await client.get_conversation_model_evaluation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.GetConversationModelEvaluationRequest()
    assert isinstance(response, conversation_model.ConversationModelEvaluation)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.raw_human_eval_template_csv == 'raw_human_eval_template_csv_value'

@pytest.mark.asyncio
async def test_get_conversation_model_evaluation_async_from_dict():
    await test_get_conversation_model_evaluation_async(request_type=dict)

def test_get_conversation_model_evaluation_field_headers():
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_model.GetConversationModelEvaluationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_conversation_model_evaluation), '__call__') as call:
        call.return_value = conversation_model.ConversationModelEvaluation()
        client.get_conversation_model_evaluation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_conversation_model_evaluation_field_headers_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_model.GetConversationModelEvaluationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_conversation_model_evaluation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(conversation_model.ConversationModelEvaluation())
        await client.get_conversation_model_evaluation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_conversation_model_evaluation_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_conversation_model_evaluation), '__call__') as call:
        call.return_value = conversation_model.ConversationModelEvaluation()
        client.get_conversation_model_evaluation(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_conversation_model_evaluation_flattened_error():
    if False:
        print('Hello World!')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_conversation_model_evaluation(conversation_model.GetConversationModelEvaluationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_conversation_model_evaluation_flattened_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_conversation_model_evaluation), '__call__') as call:
        call.return_value = conversation_model.ConversationModelEvaluation()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(conversation_model.ConversationModelEvaluation())
        response = await client.get_conversation_model_evaluation(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_conversation_model_evaluation_flattened_error_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_conversation_model_evaluation(conversation_model.GetConversationModelEvaluationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [conversation_model.ListConversationModelEvaluationsRequest, dict])
def test_list_conversation_model_evaluations(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_conversation_model_evaluations), '__call__') as call:
        call.return_value = conversation_model.ListConversationModelEvaluationsResponse(next_page_token='next_page_token_value')
        response = client.list_conversation_model_evaluations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.ListConversationModelEvaluationsRequest()
    assert isinstance(response, pagers.ListConversationModelEvaluationsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_conversation_model_evaluations_empty_call():
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_conversation_model_evaluations), '__call__') as call:
        client.list_conversation_model_evaluations()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.ListConversationModelEvaluationsRequest()

@pytest.mark.asyncio
async def test_list_conversation_model_evaluations_async(transport: str='grpc_asyncio', request_type=conversation_model.ListConversationModelEvaluationsRequest):
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_conversation_model_evaluations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(conversation_model.ListConversationModelEvaluationsResponse(next_page_token='next_page_token_value'))
        response = await client.list_conversation_model_evaluations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.ListConversationModelEvaluationsRequest()
    assert isinstance(response, pagers.ListConversationModelEvaluationsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_conversation_model_evaluations_async_from_dict():
    await test_list_conversation_model_evaluations_async(request_type=dict)

def test_list_conversation_model_evaluations_field_headers():
    if False:
        print('Hello World!')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_model.ListConversationModelEvaluationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_conversation_model_evaluations), '__call__') as call:
        call.return_value = conversation_model.ListConversationModelEvaluationsResponse()
        client.list_conversation_model_evaluations(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_conversation_model_evaluations_field_headers_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_model.ListConversationModelEvaluationsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_conversation_model_evaluations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(conversation_model.ListConversationModelEvaluationsResponse())
        await client.list_conversation_model_evaluations(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_conversation_model_evaluations_flattened():
    if False:
        print('Hello World!')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_conversation_model_evaluations), '__call__') as call:
        call.return_value = conversation_model.ListConversationModelEvaluationsResponse()
        client.list_conversation_model_evaluations(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_conversation_model_evaluations_flattened_error():
    if False:
        print('Hello World!')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_conversation_model_evaluations(conversation_model.ListConversationModelEvaluationsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_conversation_model_evaluations_flattened_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_conversation_model_evaluations), '__call__') as call:
        call.return_value = conversation_model.ListConversationModelEvaluationsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(conversation_model.ListConversationModelEvaluationsResponse())
        response = await client.list_conversation_model_evaluations(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_conversation_model_evaluations_flattened_error_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_conversation_model_evaluations(conversation_model.ListConversationModelEvaluationsRequest(), parent='parent_value')

def test_list_conversation_model_evaluations_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_conversation_model_evaluations), '__call__') as call:
        call.side_effect = (conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[conversation_model.ConversationModelEvaluation(), conversation_model.ConversationModelEvaluation(), conversation_model.ConversationModelEvaluation()], next_page_token='abc'), conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[], next_page_token='def'), conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[conversation_model.ConversationModelEvaluation()], next_page_token='ghi'), conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[conversation_model.ConversationModelEvaluation(), conversation_model.ConversationModelEvaluation()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_conversation_model_evaluations(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, conversation_model.ConversationModelEvaluation) for i in results))

def test_list_conversation_model_evaluations_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_conversation_model_evaluations), '__call__') as call:
        call.side_effect = (conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[conversation_model.ConversationModelEvaluation(), conversation_model.ConversationModelEvaluation(), conversation_model.ConversationModelEvaluation()], next_page_token='abc'), conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[], next_page_token='def'), conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[conversation_model.ConversationModelEvaluation()], next_page_token='ghi'), conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[conversation_model.ConversationModelEvaluation(), conversation_model.ConversationModelEvaluation()]), RuntimeError)
        pages = list(client.list_conversation_model_evaluations(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_conversation_model_evaluations_async_pager():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_conversation_model_evaluations), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[conversation_model.ConversationModelEvaluation(), conversation_model.ConversationModelEvaluation(), conversation_model.ConversationModelEvaluation()], next_page_token='abc'), conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[], next_page_token='def'), conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[conversation_model.ConversationModelEvaluation()], next_page_token='ghi'), conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[conversation_model.ConversationModelEvaluation(), conversation_model.ConversationModelEvaluation()]), RuntimeError)
        async_pager = await client.list_conversation_model_evaluations(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, conversation_model.ConversationModelEvaluation) for i in responses))

@pytest.mark.asyncio
async def test_list_conversation_model_evaluations_async_pages():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_conversation_model_evaluations), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[conversation_model.ConversationModelEvaluation(), conversation_model.ConversationModelEvaluation(), conversation_model.ConversationModelEvaluation()], next_page_token='abc'), conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[], next_page_token='def'), conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[conversation_model.ConversationModelEvaluation()], next_page_token='ghi'), conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[conversation_model.ConversationModelEvaluation(), conversation_model.ConversationModelEvaluation()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_conversation_model_evaluations(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [conversation_model.CreateConversationModelEvaluationRequest, dict])
def test_create_conversation_model_evaluation(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_conversation_model_evaluation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_conversation_model_evaluation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.CreateConversationModelEvaluationRequest()
    assert isinstance(response, future.Future)

def test_create_conversation_model_evaluation_empty_call():
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_conversation_model_evaluation), '__call__') as call:
        client.create_conversation_model_evaluation()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.CreateConversationModelEvaluationRequest()

@pytest.mark.asyncio
async def test_create_conversation_model_evaluation_async(transport: str='grpc_asyncio', request_type=conversation_model.CreateConversationModelEvaluationRequest):
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_conversation_model_evaluation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_conversation_model_evaluation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == conversation_model.CreateConversationModelEvaluationRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_conversation_model_evaluation_async_from_dict():
    await test_create_conversation_model_evaluation_async(request_type=dict)

def test_create_conversation_model_evaluation_field_headers():
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_model.CreateConversationModelEvaluationRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_conversation_model_evaluation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_conversation_model_evaluation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_conversation_model_evaluation_field_headers_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = conversation_model.CreateConversationModelEvaluationRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_conversation_model_evaluation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_conversation_model_evaluation(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_conversation_model_evaluation_flattened():
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_conversation_model_evaluation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_conversation_model_evaluation(parent='parent_value', conversation_model_evaluation=conversation_model.ConversationModelEvaluation(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].conversation_model_evaluation
        mock_val = conversation_model.ConversationModelEvaluation(name='name_value')
        assert arg == mock_val

def test_create_conversation_model_evaluation_flattened_error():
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_conversation_model_evaluation(conversation_model.CreateConversationModelEvaluationRequest(), parent='parent_value', conversation_model_evaluation=conversation_model.ConversationModelEvaluation(name='name_value'))

@pytest.mark.asyncio
async def test_create_conversation_model_evaluation_flattened_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_conversation_model_evaluation), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_conversation_model_evaluation(parent='parent_value', conversation_model_evaluation=conversation_model.ConversationModelEvaluation(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].conversation_model_evaluation
        mock_val = conversation_model.ConversationModelEvaluation(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_conversation_model_evaluation_flattened_error_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_conversation_model_evaluation(conversation_model.CreateConversationModelEvaluationRequest(), parent='parent_value', conversation_model_evaluation=conversation_model.ConversationModelEvaluation(name='name_value'))

@pytest.mark.parametrize('request_type', [gcd_conversation_model.CreateConversationModelRequest, dict])
def test_create_conversation_model_rest(request_type):
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request_init['conversation_model'] = {'name': 'name_value', 'display_name': 'display_name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'datasets': [{'dataset': 'dataset_value'}], 'state': 1, 'language_code': 'language_code_value', 'article_suggestion_model_metadata': {'training_model_type': 2}, 'smart_reply_model_metadata': {'training_model_type': 2}}
    test_field = gcd_conversation_model.CreateConversationModelRequest.meta.fields['conversation_model']

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
    for (field, value) in request_init['conversation_model'].items():
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
                for i in range(0, len(request_init['conversation_model'][field])):
                    del request_init['conversation_model'][field][i][subfield]
            else:
                del request_init['conversation_model'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_conversation_model(request)
    assert response.operation.name == 'operations/spam'

def test_create_conversation_model_rest_required_fields(request_type=gcd_conversation_model.CreateConversationModelRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ConversationModelsRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_conversation_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_conversation_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_conversation_model(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_conversation_model_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ConversationModelsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_conversation_model._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('conversationModel',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_conversation_model_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ConversationModelsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConversationModelsRestInterceptor())
    client = ConversationModelsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ConversationModelsRestInterceptor, 'post_create_conversation_model') as post, mock.patch.object(transports.ConversationModelsRestInterceptor, 'pre_create_conversation_model') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcd_conversation_model.CreateConversationModelRequest.pb(gcd_conversation_model.CreateConversationModelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = gcd_conversation_model.CreateConversationModelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_conversation_model(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_conversation_model_rest_bad_request(transport: str='rest', request_type=gcd_conversation_model.CreateConversationModelRequest):
    if False:
        i = 10
        return i + 15
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_conversation_model(request)

def test_create_conversation_model_rest_flattened():
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value', conversation_model=gcd_conversation_model.ConversationModel(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_conversation_model(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*}/conversationModels' % client.transport._host, args[1])

def test_create_conversation_model_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_conversation_model(gcd_conversation_model.CreateConversationModelRequest(), parent='parent_value', conversation_model=gcd_conversation_model.ConversationModel(name='name_value'))

def test_create_conversation_model_rest_error():
    if False:
        print('Hello World!')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [conversation_model.GetConversationModelRequest, dict])
def test_get_conversation_model_rest(request_type):
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/conversationModels/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = conversation_model.ConversationModel(name='name_value', display_name='display_name_value', state=conversation_model.ConversationModel.State.CREATING, language_code='language_code_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = conversation_model.ConversationModel.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_conversation_model(request)
    assert isinstance(response, conversation_model.ConversationModel)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.state == conversation_model.ConversationModel.State.CREATING
    assert response.language_code == 'language_code_value'

def test_get_conversation_model_rest_required_fields(request_type=conversation_model.GetConversationModelRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ConversationModelsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_conversation_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_conversation_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = conversation_model.ConversationModel()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = conversation_model.ConversationModel.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_conversation_model(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_conversation_model_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ConversationModelsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_conversation_model._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_conversation_model_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ConversationModelsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConversationModelsRestInterceptor())
    client = ConversationModelsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ConversationModelsRestInterceptor, 'post_get_conversation_model') as post, mock.patch.object(transports.ConversationModelsRestInterceptor, 'pre_get_conversation_model') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = conversation_model.GetConversationModelRequest.pb(conversation_model.GetConversationModelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = conversation_model.ConversationModel.to_json(conversation_model.ConversationModel())
        request = conversation_model.GetConversationModelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = conversation_model.ConversationModel()
        client.get_conversation_model(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_conversation_model_rest_bad_request(transport: str='rest', request_type=conversation_model.GetConversationModelRequest):
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/conversationModels/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_conversation_model(request)

def test_get_conversation_model_rest_flattened():
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = conversation_model.ConversationModel()
        sample_request = {'name': 'projects/sample1/conversationModels/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = conversation_model.ConversationModel.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_conversation_model(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/conversationModels/*}' % client.transport._host, args[1])

def test_get_conversation_model_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_conversation_model(conversation_model.GetConversationModelRequest(), name='name_value')

def test_get_conversation_model_rest_error():
    if False:
        print('Hello World!')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [conversation_model.ListConversationModelsRequest, dict])
def test_list_conversation_models_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = conversation_model.ListConversationModelsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = conversation_model.ListConversationModelsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_conversation_models(request)
    assert isinstance(response, pagers.ListConversationModelsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_conversation_models_rest_required_fields(request_type=conversation_model.ListConversationModelsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ConversationModelsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_conversation_models._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_conversation_models._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = conversation_model.ListConversationModelsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = conversation_model.ListConversationModelsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_conversation_models(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_conversation_models_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ConversationModelsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_conversation_models._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_conversation_models_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ConversationModelsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConversationModelsRestInterceptor())
    client = ConversationModelsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ConversationModelsRestInterceptor, 'post_list_conversation_models') as post, mock.patch.object(transports.ConversationModelsRestInterceptor, 'pre_list_conversation_models') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = conversation_model.ListConversationModelsRequest.pb(conversation_model.ListConversationModelsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = conversation_model.ListConversationModelsResponse.to_json(conversation_model.ListConversationModelsResponse())
        request = conversation_model.ListConversationModelsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = conversation_model.ListConversationModelsResponse()
        client.list_conversation_models(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_conversation_models_rest_bad_request(transport: str='rest', request_type=conversation_model.ListConversationModelsRequest):
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_conversation_models(request)

def test_list_conversation_models_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = conversation_model.ListConversationModelsResponse()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = conversation_model.ListConversationModelsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_conversation_models(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*}/conversationModels' % client.transport._host, args[1])

def test_list_conversation_models_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_conversation_models(conversation_model.ListConversationModelsRequest(), parent='parent_value')

def test_list_conversation_models_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (conversation_model.ListConversationModelsResponse(conversation_models=[conversation_model.ConversationModel(), conversation_model.ConversationModel(), conversation_model.ConversationModel()], next_page_token='abc'), conversation_model.ListConversationModelsResponse(conversation_models=[], next_page_token='def'), conversation_model.ListConversationModelsResponse(conversation_models=[conversation_model.ConversationModel()], next_page_token='ghi'), conversation_model.ListConversationModelsResponse(conversation_models=[conversation_model.ConversationModel(), conversation_model.ConversationModel()]))
        response = response + response
        response = tuple((conversation_model.ListConversationModelsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1'}
        pager = client.list_conversation_models(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, conversation_model.ConversationModel) for i in results))
        pages = list(client.list_conversation_models(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [conversation_model.DeleteConversationModelRequest, dict])
def test_delete_conversation_model_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/conversationModels/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_conversation_model(request)
    assert response.operation.name == 'operations/spam'

def test_delete_conversation_model_rest_required_fields(request_type=conversation_model.DeleteConversationModelRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ConversationModelsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_conversation_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_conversation_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_conversation_model(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_conversation_model_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ConversationModelsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_conversation_model._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_conversation_model_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ConversationModelsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConversationModelsRestInterceptor())
    client = ConversationModelsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ConversationModelsRestInterceptor, 'post_delete_conversation_model') as post, mock.patch.object(transports.ConversationModelsRestInterceptor, 'pre_delete_conversation_model') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = conversation_model.DeleteConversationModelRequest.pb(conversation_model.DeleteConversationModelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = conversation_model.DeleteConversationModelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_conversation_model(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_conversation_model_rest_bad_request(transport: str='rest', request_type=conversation_model.DeleteConversationModelRequest):
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/conversationModels/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_conversation_model(request)

def test_delete_conversation_model_rest_flattened():
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/conversationModels/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_conversation_model(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/conversationModels/*}' % client.transport._host, args[1])

def test_delete_conversation_model_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_conversation_model(conversation_model.DeleteConversationModelRequest(), name='name_value')

def test_delete_conversation_model_rest_error():
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [conversation_model.DeployConversationModelRequest, dict])
def test_deploy_conversation_model_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/conversationModels/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.deploy_conversation_model(request)
    assert response.operation.name == 'operations/spam'

def test_deploy_conversation_model_rest_required_fields(request_type=conversation_model.DeployConversationModelRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ConversationModelsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).deploy_conversation_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).deploy_conversation_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.deploy_conversation_model(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_deploy_conversation_model_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ConversationModelsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.deploy_conversation_model._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_deploy_conversation_model_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ConversationModelsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConversationModelsRestInterceptor())
    client = ConversationModelsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ConversationModelsRestInterceptor, 'post_deploy_conversation_model') as post, mock.patch.object(transports.ConversationModelsRestInterceptor, 'pre_deploy_conversation_model') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = conversation_model.DeployConversationModelRequest.pb(conversation_model.DeployConversationModelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = conversation_model.DeployConversationModelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.deploy_conversation_model(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_deploy_conversation_model_rest_bad_request(transport: str='rest', request_type=conversation_model.DeployConversationModelRequest):
    if False:
        print('Hello World!')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/conversationModels/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.deploy_conversation_model(request)

def test_deploy_conversation_model_rest_error():
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [conversation_model.UndeployConversationModelRequest, dict])
def test_undeploy_conversation_model_rest(request_type):
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/conversationModels/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.undeploy_conversation_model(request)
    assert response.operation.name == 'operations/spam'

def test_undeploy_conversation_model_rest_required_fields(request_type=conversation_model.UndeployConversationModelRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ConversationModelsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).undeploy_conversation_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).undeploy_conversation_model._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.undeploy_conversation_model(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_undeploy_conversation_model_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ConversationModelsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.undeploy_conversation_model._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_undeploy_conversation_model_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ConversationModelsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConversationModelsRestInterceptor())
    client = ConversationModelsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ConversationModelsRestInterceptor, 'post_undeploy_conversation_model') as post, mock.patch.object(transports.ConversationModelsRestInterceptor, 'pre_undeploy_conversation_model') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = conversation_model.UndeployConversationModelRequest.pb(conversation_model.UndeployConversationModelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = conversation_model.UndeployConversationModelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.undeploy_conversation_model(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_undeploy_conversation_model_rest_bad_request(transport: str='rest', request_type=conversation_model.UndeployConversationModelRequest):
    if False:
        print('Hello World!')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/conversationModels/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.undeploy_conversation_model(request)

def test_undeploy_conversation_model_rest_error():
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [conversation_model.GetConversationModelEvaluationRequest, dict])
def test_get_conversation_model_evaluation_rest(request_type):
    if False:
        print('Hello World!')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/conversationModels/sample2/evaluations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = conversation_model.ConversationModelEvaluation(name='name_value', display_name='display_name_value', raw_human_eval_template_csv='raw_human_eval_template_csv_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = conversation_model.ConversationModelEvaluation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_conversation_model_evaluation(request)
    assert isinstance(response, conversation_model.ConversationModelEvaluation)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.raw_human_eval_template_csv == 'raw_human_eval_template_csv_value'

def test_get_conversation_model_evaluation_rest_required_fields(request_type=conversation_model.GetConversationModelEvaluationRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ConversationModelsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_conversation_model_evaluation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_conversation_model_evaluation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = conversation_model.ConversationModelEvaluation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = conversation_model.ConversationModelEvaluation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_conversation_model_evaluation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_conversation_model_evaluation_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ConversationModelsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_conversation_model_evaluation._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_conversation_model_evaluation_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ConversationModelsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConversationModelsRestInterceptor())
    client = ConversationModelsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ConversationModelsRestInterceptor, 'post_get_conversation_model_evaluation') as post, mock.patch.object(transports.ConversationModelsRestInterceptor, 'pre_get_conversation_model_evaluation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = conversation_model.GetConversationModelEvaluationRequest.pb(conversation_model.GetConversationModelEvaluationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = conversation_model.ConversationModelEvaluation.to_json(conversation_model.ConversationModelEvaluation())
        request = conversation_model.GetConversationModelEvaluationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = conversation_model.ConversationModelEvaluation()
        client.get_conversation_model_evaluation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_conversation_model_evaluation_rest_bad_request(transport: str='rest', request_type=conversation_model.GetConversationModelEvaluationRequest):
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/conversationModels/sample2/evaluations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_conversation_model_evaluation(request)

def test_get_conversation_model_evaluation_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = conversation_model.ConversationModelEvaluation()
        sample_request = {'name': 'projects/sample1/conversationModels/sample2/evaluations/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = conversation_model.ConversationModelEvaluation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_conversation_model_evaluation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/conversationModels/*/evaluations/*}' % client.transport._host, args[1])

def test_get_conversation_model_evaluation_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_conversation_model_evaluation(conversation_model.GetConversationModelEvaluationRequest(), name='name_value')

def test_get_conversation_model_evaluation_rest_error():
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [conversation_model.ListConversationModelEvaluationsRequest, dict])
def test_list_conversation_model_evaluations_rest(request_type):
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/conversationModels/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = conversation_model.ListConversationModelEvaluationsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = conversation_model.ListConversationModelEvaluationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_conversation_model_evaluations(request)
    assert isinstance(response, pagers.ListConversationModelEvaluationsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_conversation_model_evaluations_rest_required_fields(request_type=conversation_model.ListConversationModelEvaluationsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ConversationModelsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_conversation_model_evaluations._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_conversation_model_evaluations._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = conversation_model.ListConversationModelEvaluationsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = conversation_model.ListConversationModelEvaluationsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_conversation_model_evaluations(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_conversation_model_evaluations_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ConversationModelsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_conversation_model_evaluations._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_conversation_model_evaluations_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ConversationModelsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConversationModelsRestInterceptor())
    client = ConversationModelsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ConversationModelsRestInterceptor, 'post_list_conversation_model_evaluations') as post, mock.patch.object(transports.ConversationModelsRestInterceptor, 'pre_list_conversation_model_evaluations') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = conversation_model.ListConversationModelEvaluationsRequest.pb(conversation_model.ListConversationModelEvaluationsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = conversation_model.ListConversationModelEvaluationsResponse.to_json(conversation_model.ListConversationModelEvaluationsResponse())
        request = conversation_model.ListConversationModelEvaluationsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = conversation_model.ListConversationModelEvaluationsResponse()
        client.list_conversation_model_evaluations(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_conversation_model_evaluations_rest_bad_request(transport: str='rest', request_type=conversation_model.ListConversationModelEvaluationsRequest):
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/conversationModels/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_conversation_model_evaluations(request)

def test_list_conversation_model_evaluations_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = conversation_model.ListConversationModelEvaluationsResponse()
        sample_request = {'parent': 'projects/sample1/conversationModels/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = conversation_model.ListConversationModelEvaluationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_conversation_model_evaluations(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/conversationModels/*}/evaluations' % client.transport._host, args[1])

def test_list_conversation_model_evaluations_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_conversation_model_evaluations(conversation_model.ListConversationModelEvaluationsRequest(), parent='parent_value')

def test_list_conversation_model_evaluations_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[conversation_model.ConversationModelEvaluation(), conversation_model.ConversationModelEvaluation(), conversation_model.ConversationModelEvaluation()], next_page_token='abc'), conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[], next_page_token='def'), conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[conversation_model.ConversationModelEvaluation()], next_page_token='ghi'), conversation_model.ListConversationModelEvaluationsResponse(conversation_model_evaluations=[conversation_model.ConversationModelEvaluation(), conversation_model.ConversationModelEvaluation()]))
        response = response + response
        response = tuple((conversation_model.ListConversationModelEvaluationsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/conversationModels/sample2'}
        pager = client.list_conversation_model_evaluations(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, conversation_model.ConversationModelEvaluation) for i in results))
        pages = list(client.list_conversation_model_evaluations(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [conversation_model.CreateConversationModelEvaluationRequest, dict])
def test_create_conversation_model_evaluation_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/conversationModels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_conversation_model_evaluation(request)
    assert response.operation.name == 'operations/spam'

def test_create_conversation_model_evaluation_rest_required_fields(request_type=conversation_model.CreateConversationModelEvaluationRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ConversationModelsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_conversation_model_evaluation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_conversation_model_evaluation._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_conversation_model_evaluation(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_conversation_model_evaluation_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ConversationModelsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_conversation_model_evaluation._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'conversationModelEvaluation'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_conversation_model_evaluation_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ConversationModelsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ConversationModelsRestInterceptor())
    client = ConversationModelsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ConversationModelsRestInterceptor, 'post_create_conversation_model_evaluation') as post, mock.patch.object(transports.ConversationModelsRestInterceptor, 'pre_create_conversation_model_evaluation') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = conversation_model.CreateConversationModelEvaluationRequest.pb(conversation_model.CreateConversationModelEvaluationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = conversation_model.CreateConversationModelEvaluationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_conversation_model_evaluation(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_conversation_model_evaluation_rest_bad_request(transport: str='rest', request_type=conversation_model.CreateConversationModelEvaluationRequest):
    if False:
        i = 10
        return i + 15
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/conversationModels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_conversation_model_evaluation(request)

def test_create_conversation_model_evaluation_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2/conversationModels/sample3'}
        mock_args = dict(parent='parent_value', conversation_model_evaluation=conversation_model.ConversationModelEvaluation(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_conversation_model_evaluation(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*/conversationModels/*}/evaluations' % client.transport._host, args[1])

def test_create_conversation_model_evaluation_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_conversation_model_evaluation(conversation_model.CreateConversationModelEvaluationRequest(), parent='parent_value', conversation_model_evaluation=conversation_model.ConversationModelEvaluation(name='name_value'))

def test_create_conversation_model_evaluation_rest_error():
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.ConversationModelsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.ConversationModelsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ConversationModelsClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.ConversationModelsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ConversationModelsClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ConversationModelsClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.ConversationModelsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ConversationModelsClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ConversationModelsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = ConversationModelsClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.ConversationModelsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.ConversationModelsGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.ConversationModelsGrpcTransport, transports.ConversationModelsGrpcAsyncIOTransport, transports.ConversationModelsRestTransport])
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
    transport = ConversationModelsClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.ConversationModelsGrpcTransport)

def test_conversation_models_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.ConversationModelsTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_conversation_models_base_transport():
    if False:
        return 10
    with mock.patch('google.cloud.dialogflow_v2.services.conversation_models.transports.ConversationModelsTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.ConversationModelsTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_conversation_model', 'get_conversation_model', 'list_conversation_models', 'delete_conversation_model', 'deploy_conversation_model', 'undeploy_conversation_model', 'get_conversation_model_evaluation', 'list_conversation_model_evaluations', 'create_conversation_model_evaluation', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'list_operations')
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

def test_conversation_models_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.dialogflow_v2.services.conversation_models.transports.ConversationModelsTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ConversationModelsTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id='octopus')

def test_conversation_models_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.dialogflow_v2.services.conversation_models.transports.ConversationModelsTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ConversationModelsTransport()
        adc.assert_called_once()

def test_conversation_models_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        ConversationModelsClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.ConversationModelsGrpcTransport, transports.ConversationModelsGrpcAsyncIOTransport])
def test_conversation_models_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.ConversationModelsGrpcTransport, transports.ConversationModelsGrpcAsyncIOTransport, transports.ConversationModelsRestTransport])
def test_conversation_models_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.ConversationModelsGrpcTransport, grpc_helpers), (transports.ConversationModelsGrpcAsyncIOTransport, grpc_helpers_async)])
def test_conversation_models_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('dialogflow.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), scopes=['1', '2'], default_host='dialogflow.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.ConversationModelsGrpcTransport, transports.ConversationModelsGrpcAsyncIOTransport])
def test_conversation_models_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_conversation_models_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.ConversationModelsRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_conversation_models_rest_lro_client():
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_conversation_models_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dialogflow.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_conversation_models_host_with_port(transport_name):
    if False:
        print('Hello World!')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dialogflow.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('dialogflow.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_conversation_models_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = ConversationModelsClient(credentials=creds1, transport=transport_name)
    client2 = ConversationModelsClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_conversation_model._session
    session2 = client2.transport.create_conversation_model._session
    assert session1 != session2
    session1 = client1.transport.get_conversation_model._session
    session2 = client2.transport.get_conversation_model._session
    assert session1 != session2
    session1 = client1.transport.list_conversation_models._session
    session2 = client2.transport.list_conversation_models._session
    assert session1 != session2
    session1 = client1.transport.delete_conversation_model._session
    session2 = client2.transport.delete_conversation_model._session
    assert session1 != session2
    session1 = client1.transport.deploy_conversation_model._session
    session2 = client2.transport.deploy_conversation_model._session
    assert session1 != session2
    session1 = client1.transport.undeploy_conversation_model._session
    session2 = client2.transport.undeploy_conversation_model._session
    assert session1 != session2
    session1 = client1.transport.get_conversation_model_evaluation._session
    session2 = client2.transport.get_conversation_model_evaluation._session
    assert session1 != session2
    session1 = client1.transport.list_conversation_model_evaluations._session
    session2 = client2.transport.list_conversation_model_evaluations._session
    assert session1 != session2
    session1 = client1.transport.create_conversation_model_evaluation._session
    session2 = client2.transport.create_conversation_model_evaluation._session
    assert session1 != session2

def test_conversation_models_grpc_transport_channel():
    if False:
        print('Hello World!')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ConversationModelsGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_conversation_models_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ConversationModelsGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.ConversationModelsGrpcTransport, transports.ConversationModelsGrpcAsyncIOTransport])
def test_conversation_models_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.ConversationModelsGrpcTransport, transports.ConversationModelsGrpcAsyncIOTransport])
def test_conversation_models_transport_channel_mtls_with_adc(transport_class):
    if False:
        print('Hello World!')
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

def test_conversation_models_grpc_lro_client():
    if False:
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_conversation_models_grpc_lro_async_client():
    if False:
        i = 10
        return i + 15
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_conversation_dataset_path():
    if False:
        return 10
    project = 'squid'
    location = 'clam'
    conversation_dataset = 'whelk'
    expected = 'projects/{project}/locations/{location}/conversationDatasets/{conversation_dataset}'.format(project=project, location=location, conversation_dataset=conversation_dataset)
    actual = ConversationModelsClient.conversation_dataset_path(project, location, conversation_dataset)
    assert expected == actual

def test_parse_conversation_dataset_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'octopus', 'location': 'oyster', 'conversation_dataset': 'nudibranch'}
    path = ConversationModelsClient.conversation_dataset_path(**expected)
    actual = ConversationModelsClient.parse_conversation_dataset_path(path)
    assert expected == actual

def test_conversation_model_path():
    if False:
        return 10
    project = 'cuttlefish'
    location = 'mussel'
    conversation_model = 'winkle'
    expected = 'projects/{project}/locations/{location}/conversationModels/{conversation_model}'.format(project=project, location=location, conversation_model=conversation_model)
    actual = ConversationModelsClient.conversation_model_path(project, location, conversation_model)
    assert expected == actual

def test_parse_conversation_model_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'nautilus', 'location': 'scallop', 'conversation_model': 'abalone'}
    path = ConversationModelsClient.conversation_model_path(**expected)
    actual = ConversationModelsClient.parse_conversation_model_path(path)
    assert expected == actual

def test_conversation_model_evaluation_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    conversation_model = 'clam'
    evaluation = 'whelk'
    expected = 'projects/{project}/conversationModels/{conversation_model}/evaluations/{evaluation}'.format(project=project, conversation_model=conversation_model, evaluation=evaluation)
    actual = ConversationModelsClient.conversation_model_evaluation_path(project, conversation_model, evaluation)
    assert expected == actual

def test_parse_conversation_model_evaluation_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus', 'conversation_model': 'oyster', 'evaluation': 'nudibranch'}
    path = ConversationModelsClient.conversation_model_evaluation_path(**expected)
    actual = ConversationModelsClient.parse_conversation_model_evaluation_path(path)
    assert expected == actual

def test_document_path():
    if False:
        return 10
    project = 'cuttlefish'
    knowledge_base = 'mussel'
    document = 'winkle'
    expected = 'projects/{project}/knowledgeBases/{knowledge_base}/documents/{document}'.format(project=project, knowledge_base=knowledge_base, document=document)
    actual = ConversationModelsClient.document_path(project, knowledge_base, document)
    assert expected == actual

def test_parse_document_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'nautilus', 'knowledge_base': 'scallop', 'document': 'abalone'}
    path = ConversationModelsClient.document_path(**expected)
    actual = ConversationModelsClient.parse_document_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = ConversationModelsClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'clam'}
    path = ConversationModelsClient.common_billing_account_path(**expected)
    actual = ConversationModelsClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = ConversationModelsClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'octopus'}
    path = ConversationModelsClient.common_folder_path(**expected)
    actual = ConversationModelsClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = ConversationModelsClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'nudibranch'}
    path = ConversationModelsClient.common_organization_path(**expected)
    actual = ConversationModelsClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = ConversationModelsClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'mussel'}
    path = ConversationModelsClient.common_project_path(**expected)
    actual = ConversationModelsClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = ConversationModelsClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = ConversationModelsClient.common_location_path(**expected)
    actual = ConversationModelsClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.ConversationModelsTransport, '_prep_wrapped_messages') as prep:
        client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.ConversationModelsTransport, '_prep_wrapped_messages') as prep:
        transport_class = ConversationModelsClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        return 10
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = ConversationModelsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        i = 10
        return i + 15
    transports = ['rest', 'grpc']
    for transport in transports:
        client = ConversationModelsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(ConversationModelsClient, transports.ConversationModelsGrpcTransport), (ConversationModelsAsyncClient, transports.ConversationModelsGrpcAsyncIOTransport)])
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