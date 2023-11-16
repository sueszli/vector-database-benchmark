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
from google.protobuf import json_format
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.ai.generativelanguage_v1beta2.services.text_service import TextServiceAsyncClient, TextServiceClient, transports
from google.ai.generativelanguage_v1beta2.types import safety, text_service

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        i = 10
        return i + 15
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        return 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert TextServiceClient._get_default_mtls_endpoint(None) is None
    assert TextServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert TextServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert TextServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert TextServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert TextServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(TextServiceClient, 'grpc'), (TextServiceAsyncClient, 'grpc_asyncio'), (TextServiceClient, 'rest')])
def test_text_service_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('generativelanguage.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://generativelanguage.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.TextServiceGrpcTransport, 'grpc'), (transports.TextServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.TextServiceRestTransport, 'rest')])
def test_text_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(TextServiceClient, 'grpc'), (TextServiceAsyncClient, 'grpc_asyncio'), (TextServiceClient, 'rest')])
def test_text_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('generativelanguage.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://generativelanguage.googleapis.com')

def test_text_service_client_get_transport_class():
    if False:
        for i in range(10):
            print('nop')
    transport = TextServiceClient.get_transport_class()
    available_transports = [transports.TextServiceGrpcTransport, transports.TextServiceRestTransport]
    assert transport in available_transports
    transport = TextServiceClient.get_transport_class('grpc')
    assert transport == transports.TextServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(TextServiceClient, transports.TextServiceGrpcTransport, 'grpc'), (TextServiceAsyncClient, transports.TextServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (TextServiceClient, transports.TextServiceRestTransport, 'rest')])
@mock.patch.object(TextServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TextServiceClient))
@mock.patch.object(TextServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TextServiceAsyncClient))
def test_text_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(TextServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(TextServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(TextServiceClient, transports.TextServiceGrpcTransport, 'grpc', 'true'), (TextServiceAsyncClient, transports.TextServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (TextServiceClient, transports.TextServiceGrpcTransport, 'grpc', 'false'), (TextServiceAsyncClient, transports.TextServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (TextServiceClient, transports.TextServiceRestTransport, 'rest', 'true'), (TextServiceClient, transports.TextServiceRestTransport, 'rest', 'false')])
@mock.patch.object(TextServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TextServiceClient))
@mock.patch.object(TextServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TextServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_text_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [TextServiceClient, TextServiceAsyncClient])
@mock.patch.object(TextServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TextServiceClient))
@mock.patch.object(TextServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TextServiceAsyncClient))
def test_text_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(TextServiceClient, transports.TextServiceGrpcTransport, 'grpc'), (TextServiceAsyncClient, transports.TextServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (TextServiceClient, transports.TextServiceRestTransport, 'rest')])
def test_text_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(TextServiceClient, transports.TextServiceGrpcTransport, 'grpc', grpc_helpers), (TextServiceAsyncClient, transports.TextServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (TextServiceClient, transports.TextServiceRestTransport, 'rest', None)])
def test_text_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_text_service_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.ai.generativelanguage_v1beta2.services.text_service.transports.TextServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = TextServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(TextServiceClient, transports.TextServiceGrpcTransport, 'grpc', grpc_helpers), (TextServiceAsyncClient, transports.TextServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_text_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('generativelanguage.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=(), scopes=None, default_host='generativelanguage.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [text_service.GenerateTextRequest, dict])
def test_generate_text(request_type, transport: str='grpc'):
    if False:
        return 10
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_text), '__call__') as call:
        call.return_value = text_service.GenerateTextResponse()
        response = client.generate_text(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == text_service.GenerateTextRequest()
    assert isinstance(response, text_service.GenerateTextResponse)

def test_generate_text_empty_call():
    if False:
        print('Hello World!')
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.generate_text), '__call__') as call:
        client.generate_text()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == text_service.GenerateTextRequest()

@pytest.mark.asyncio
async def test_generate_text_async(transport: str='grpc_asyncio', request_type=text_service.GenerateTextRequest):
    client = TextServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_text), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(text_service.GenerateTextResponse())
        response = await client.generate_text(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == text_service.GenerateTextRequest()
    assert isinstance(response, text_service.GenerateTextResponse)

@pytest.mark.asyncio
async def test_generate_text_async_from_dict():
    await test_generate_text_async(request_type=dict)

def test_generate_text_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = text_service.GenerateTextRequest()
    request.model = 'model_value'
    with mock.patch.object(type(client.transport.generate_text), '__call__') as call:
        call.return_value = text_service.GenerateTextResponse()
        client.generate_text(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'model=model_value') in kw['metadata']

@pytest.mark.asyncio
async def test_generate_text_field_headers_async():
    client = TextServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = text_service.GenerateTextRequest()
    request.model = 'model_value'
    with mock.patch.object(type(client.transport.generate_text), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(text_service.GenerateTextResponse())
        await client.generate_text(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'model=model_value') in kw['metadata']

def test_generate_text_flattened():
    if False:
        return 10
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.generate_text), '__call__') as call:
        call.return_value = text_service.GenerateTextResponse()
        client.generate_text(model='model_value', prompt=text_service.TextPrompt(text='text_value'), temperature=0.1198, candidate_count=1573, max_output_tokens=1865, top_p=0.546, top_k=541)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].model
        mock_val = 'model_value'
        assert arg == mock_val
        arg = args[0].prompt
        mock_val = text_service.TextPrompt(text='text_value')
        assert arg == mock_val
        assert math.isclose(args[0].temperature, 0.1198, rel_tol=1e-06)
        arg = args[0].candidate_count
        mock_val = 1573
        assert arg == mock_val
        arg = args[0].max_output_tokens
        mock_val = 1865
        assert arg == mock_val
        assert math.isclose(args[0].top_p, 0.546, rel_tol=1e-06)
        arg = args[0].top_k
        mock_val = 541
        assert arg == mock_val

def test_generate_text_flattened_error():
    if False:
        i = 10
        return i + 15
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.generate_text(text_service.GenerateTextRequest(), model='model_value', prompt=text_service.TextPrompt(text='text_value'), temperature=0.1198, candidate_count=1573, max_output_tokens=1865, top_p=0.546, top_k=541)

@pytest.mark.asyncio
async def test_generate_text_flattened_async():
    client = TextServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.generate_text), '__call__') as call:
        call.return_value = text_service.GenerateTextResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(text_service.GenerateTextResponse())
        response = await client.generate_text(model='model_value', prompt=text_service.TextPrompt(text='text_value'), temperature=0.1198, candidate_count=1573, max_output_tokens=1865, top_p=0.546, top_k=541)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].model
        mock_val = 'model_value'
        assert arg == mock_val
        arg = args[0].prompt
        mock_val = text_service.TextPrompt(text='text_value')
        assert arg == mock_val
        assert math.isclose(args[0].temperature, 0.1198, rel_tol=1e-06)
        arg = args[0].candidate_count
        mock_val = 1573
        assert arg == mock_val
        arg = args[0].max_output_tokens
        mock_val = 1865
        assert arg == mock_val
        assert math.isclose(args[0].top_p, 0.546, rel_tol=1e-06)
        arg = args[0].top_k
        mock_val = 541
        assert arg == mock_val

@pytest.mark.asyncio
async def test_generate_text_flattened_error_async():
    client = TextServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.generate_text(text_service.GenerateTextRequest(), model='model_value', prompt=text_service.TextPrompt(text='text_value'), temperature=0.1198, candidate_count=1573, max_output_tokens=1865, top_p=0.546, top_k=541)

@pytest.mark.parametrize('request_type', [text_service.EmbedTextRequest, dict])
def test_embed_text(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.embed_text), '__call__') as call:
        call.return_value = text_service.EmbedTextResponse()
        response = client.embed_text(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == text_service.EmbedTextRequest()
    assert isinstance(response, text_service.EmbedTextResponse)

def test_embed_text_empty_call():
    if False:
        while True:
            i = 10
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.embed_text), '__call__') as call:
        client.embed_text()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == text_service.EmbedTextRequest()

@pytest.mark.asyncio
async def test_embed_text_async(transport: str='grpc_asyncio', request_type=text_service.EmbedTextRequest):
    client = TextServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.embed_text), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(text_service.EmbedTextResponse())
        response = await client.embed_text(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == text_service.EmbedTextRequest()
    assert isinstance(response, text_service.EmbedTextResponse)

@pytest.mark.asyncio
async def test_embed_text_async_from_dict():
    await test_embed_text_async(request_type=dict)

def test_embed_text_field_headers():
    if False:
        while True:
            i = 10
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = text_service.EmbedTextRequest()
    request.model = 'model_value'
    with mock.patch.object(type(client.transport.embed_text), '__call__') as call:
        call.return_value = text_service.EmbedTextResponse()
        client.embed_text(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'model=model_value') in kw['metadata']

@pytest.mark.asyncio
async def test_embed_text_field_headers_async():
    client = TextServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = text_service.EmbedTextRequest()
    request.model = 'model_value'
    with mock.patch.object(type(client.transport.embed_text), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(text_service.EmbedTextResponse())
        await client.embed_text(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'model=model_value') in kw['metadata']

def test_embed_text_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.embed_text), '__call__') as call:
        call.return_value = text_service.EmbedTextResponse()
        client.embed_text(model='model_value', text='text_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].model
        mock_val = 'model_value'
        assert arg == mock_val
        arg = args[0].text
        mock_val = 'text_value'
        assert arg == mock_val

def test_embed_text_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.embed_text(text_service.EmbedTextRequest(), model='model_value', text='text_value')

@pytest.mark.asyncio
async def test_embed_text_flattened_async():
    client = TextServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.embed_text), '__call__') as call:
        call.return_value = text_service.EmbedTextResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(text_service.EmbedTextResponse())
        response = await client.embed_text(model='model_value', text='text_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].model
        mock_val = 'model_value'
        assert arg == mock_val
        arg = args[0].text
        mock_val = 'text_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_embed_text_flattened_error_async():
    client = TextServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.embed_text(text_service.EmbedTextRequest(), model='model_value', text='text_value')

@pytest.mark.parametrize('request_type', [text_service.GenerateTextRequest, dict])
def test_generate_text_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'model': 'models/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = text_service.GenerateTextResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = text_service.GenerateTextResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.generate_text(request)
    assert isinstance(response, text_service.GenerateTextResponse)

def test_generate_text_rest_required_fields(request_type=text_service.GenerateTextRequest):
    if False:
        print('Hello World!')
    transport_class = transports.TextServiceRestTransport
    request_init = {}
    request_init['model'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_text._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['model'] = 'model_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_text._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'model' in jsonified_request
    assert jsonified_request['model'] == 'model_value'
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = text_service.GenerateTextResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = text_service.GenerateTextResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.generate_text(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_generate_text_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.TextServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.generate_text._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('model', 'prompt'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_generate_text_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.TextServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TextServiceRestInterceptor())
    client = TextServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TextServiceRestInterceptor, 'post_generate_text') as post, mock.patch.object(transports.TextServiceRestInterceptor, 'pre_generate_text') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = text_service.GenerateTextRequest.pb(text_service.GenerateTextRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = text_service.GenerateTextResponse.to_json(text_service.GenerateTextResponse())
        request = text_service.GenerateTextRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = text_service.GenerateTextResponse()
        client.generate_text(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_generate_text_rest_bad_request(transport: str='rest', request_type=text_service.GenerateTextRequest):
    if False:
        i = 10
        return i + 15
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'model': 'models/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.generate_text(request)

def test_generate_text_rest_flattened():
    if False:
        return 10
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = text_service.GenerateTextResponse()
        sample_request = {'model': 'models/sample1'}
        mock_args = dict(model='model_value', prompt=text_service.TextPrompt(text='text_value'), temperature=0.1198, candidate_count=1573, max_output_tokens=1865, top_p=0.546, top_k=541)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = text_service.GenerateTextResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.generate_text(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta2/{model=models/*}:generateText' % client.transport._host, args[1])

def test_generate_text_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.generate_text(text_service.GenerateTextRequest(), model='model_value', prompt=text_service.TextPrompt(text='text_value'), temperature=0.1198, candidate_count=1573, max_output_tokens=1865, top_p=0.546, top_k=541)

def test_generate_text_rest_error():
    if False:
        print('Hello World!')
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [text_service.EmbedTextRequest, dict])
def test_embed_text_rest(request_type):
    if False:
        return 10
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'model': 'models/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = text_service.EmbedTextResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = text_service.EmbedTextResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.embed_text(request)
    assert isinstance(response, text_service.EmbedTextResponse)

def test_embed_text_rest_required_fields(request_type=text_service.EmbedTextRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.TextServiceRestTransport
    request_init = {}
    request_init['model'] = ''
    request_init['text'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).embed_text._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['model'] = 'model_value'
    jsonified_request['text'] = 'text_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).embed_text._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'model' in jsonified_request
    assert jsonified_request['model'] == 'model_value'
    assert 'text' in jsonified_request
    assert jsonified_request['text'] == 'text_value'
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = text_service.EmbedTextResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = text_service.EmbedTextResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.embed_text(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_embed_text_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.TextServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.embed_text._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('model', 'text'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_embed_text_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.TextServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TextServiceRestInterceptor())
    client = TextServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TextServiceRestInterceptor, 'post_embed_text') as post, mock.patch.object(transports.TextServiceRestInterceptor, 'pre_embed_text') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = text_service.EmbedTextRequest.pb(text_service.EmbedTextRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = text_service.EmbedTextResponse.to_json(text_service.EmbedTextResponse())
        request = text_service.EmbedTextRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = text_service.EmbedTextResponse()
        client.embed_text(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_embed_text_rest_bad_request(transport: str='rest', request_type=text_service.EmbedTextRequest):
    if False:
        for i in range(10):
            print('nop')
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'model': 'models/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.embed_text(request)

def test_embed_text_rest_flattened():
    if False:
        while True:
            i = 10
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = text_service.EmbedTextResponse()
        sample_request = {'model': 'models/sample1'}
        mock_args = dict(model='model_value', text='text_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = text_service.EmbedTextResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.embed_text(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta2/{model=models/*}:embedText' % client.transport._host, args[1])

def test_embed_text_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.embed_text(text_service.EmbedTextRequest(), model='model_value', text='text_value')

def test_embed_text_rest_error():
    if False:
        return 10
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.TextServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.TextServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TextServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.TextServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = TextServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = TextServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.TextServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TextServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.TextServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = TextServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.TextServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.TextServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.TextServiceGrpcTransport, transports.TextServiceGrpcAsyncIOTransport, transports.TextServiceRestTransport])
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
        return 10
    transport = TextServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.TextServiceGrpcTransport)

def test_text_service_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.TextServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_text_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.ai.generativelanguage_v1beta2.services.text_service.transports.TextServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.TextServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('generate_text', 'embed_text')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_text_service_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.ai.generativelanguage_v1beta2.services.text_service.transports.TextServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.TextServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=(), quota_project_id='octopus')

def test_text_service_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.ai.generativelanguage_v1beta2.services.text_service.transports.TextServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.TextServiceTransport()
        adc.assert_called_once()

def test_text_service_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        TextServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=(), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.TextServiceGrpcTransport, transports.TextServiceGrpcAsyncIOTransport])
def test_text_service_transport_auth_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=(), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.TextServiceGrpcTransport, transports.TextServiceGrpcAsyncIOTransport, transports.TextServiceRestTransport])
def test_text_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.TextServiceGrpcTransport, grpc_helpers), (transports.TextServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_text_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('generativelanguage.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=(), scopes=['1', '2'], default_host='generativelanguage.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.TextServiceGrpcTransport, transports.TextServiceGrpcAsyncIOTransport])
def test_text_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_text_service_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.TextServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_text_service_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='generativelanguage.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('generativelanguage.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://generativelanguage.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_text_service_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='generativelanguage.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('generativelanguage.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://generativelanguage.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_text_service_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = TextServiceClient(credentials=creds1, transport=transport_name)
    client2 = TextServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.generate_text._session
    session2 = client2.transport.generate_text._session
    assert session1 != session2
    session1 = client1.transport.embed_text._session
    session2 = client2.transport.embed_text._session
    assert session1 != session2

def test_text_service_grpc_transport_channel():
    if False:
        print('Hello World!')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.TextServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_text_service_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.TextServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.TextServiceGrpcTransport, transports.TextServiceGrpcAsyncIOTransport])
def test_text_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.TextServiceGrpcTransport, transports.TextServiceGrpcAsyncIOTransport])
def test_text_service_transport_channel_mtls_with_adc(transport_class):
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

def test_model_path():
    if False:
        print('Hello World!')
    model = 'squid'
    expected = 'models/{model}'.format(model=model)
    actual = TextServiceClient.model_path(model)
    assert expected == actual

def test_parse_model_path():
    if False:
        print('Hello World!')
    expected = {'model': 'clam'}
    path = TextServiceClient.model_path(**expected)
    actual = TextServiceClient.parse_model_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = TextServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'octopus'}
    path = TextServiceClient.common_billing_account_path(**expected)
    actual = TextServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = TextServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'nudibranch'}
    path = TextServiceClient.common_folder_path(**expected)
    actual = TextServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = TextServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'mussel'}
    path = TextServiceClient.common_organization_path(**expected)
    actual = TextServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = TextServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nautilus'}
    path = TextServiceClient.common_project_path(**expected)
    actual = TextServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = TextServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'squid', 'location': 'clam'}
    path = TextServiceClient.common_location_path(**expected)
    actual = TextServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.TextServiceTransport, '_prep_wrapped_messages') as prep:
        client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.TextServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = TextServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = TextServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        return 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = TextServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(TextServiceClient, transports.TextServiceGrpcTransport), (TextServiceAsyncClient, transports.TextServiceGrpcAsyncIOTransport)])
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