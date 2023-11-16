import os
try:
    from unittest import mock
    from unittest.mock import AsyncMock
except ImportError:
    import mock
from collections.abc import Iterable
import json
import math
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.api_core import future, gapic_v1, grpc_helpers, grpc_helpers_async, operation, operations_v1, path_template
from google.api_core import client_options
from google.api_core import exceptions as core_exceptions
from google.api_core import operation_async
import google.auth
from google.auth import credentials as ga_credentials
from google.auth.exceptions import MutualTLSChannelError
from google.cloud.translate_v3beta1.services.translation_service import TranslationServiceAsyncClient, TranslationServiceClient, pagers, transports
from google.cloud.translate_v3beta1.types import translation_service
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import json_format
from google.protobuf import timestamp_pb2

def client_cert_source_callback():
    if False:
        print('Hello World!')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        return 10
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
    assert TranslationServiceClient._get_default_mtls_endpoint(None) is None
    assert TranslationServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert TranslationServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert TranslationServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert TranslationServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert TranslationServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(TranslationServiceClient, 'grpc'), (TranslationServiceAsyncClient, 'grpc_asyncio'), (TranslationServiceClient, 'rest')])
def test_translation_service_client_from_service_account_info(client_class, transport_name):
    if False:
        print('Hello World!')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('translate.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://translate.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.TranslationServiceGrpcTransport, 'grpc'), (transports.TranslationServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.TranslationServiceRestTransport, 'rest')])
def test_translation_service_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(TranslationServiceClient, 'grpc'), (TranslationServiceAsyncClient, 'grpc_asyncio'), (TranslationServiceClient, 'rest')])
def test_translation_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('translate.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://translate.googleapis.com')

def test_translation_service_client_get_transport_class():
    if False:
        return 10
    transport = TranslationServiceClient.get_transport_class()
    available_transports = [transports.TranslationServiceGrpcTransport, transports.TranslationServiceRestTransport]
    assert transport in available_transports
    transport = TranslationServiceClient.get_transport_class('grpc')
    assert transport == transports.TranslationServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(TranslationServiceClient, transports.TranslationServiceGrpcTransport, 'grpc'), (TranslationServiceAsyncClient, transports.TranslationServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (TranslationServiceClient, transports.TranslationServiceRestTransport, 'rest')])
@mock.patch.object(TranslationServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TranslationServiceClient))
@mock.patch.object(TranslationServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TranslationServiceAsyncClient))
def test_translation_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(TranslationServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(TranslationServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(TranslationServiceClient, transports.TranslationServiceGrpcTransport, 'grpc', 'true'), (TranslationServiceAsyncClient, transports.TranslationServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (TranslationServiceClient, transports.TranslationServiceGrpcTransport, 'grpc', 'false'), (TranslationServiceAsyncClient, transports.TranslationServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (TranslationServiceClient, transports.TranslationServiceRestTransport, 'rest', 'true'), (TranslationServiceClient, transports.TranslationServiceRestTransport, 'rest', 'false')])
@mock.patch.object(TranslationServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TranslationServiceClient))
@mock.patch.object(TranslationServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TranslationServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_translation_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [TranslationServiceClient, TranslationServiceAsyncClient])
@mock.patch.object(TranslationServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TranslationServiceClient))
@mock.patch.object(TranslationServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TranslationServiceAsyncClient))
def test_translation_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(TranslationServiceClient, transports.TranslationServiceGrpcTransport, 'grpc'), (TranslationServiceAsyncClient, transports.TranslationServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (TranslationServiceClient, transports.TranslationServiceRestTransport, 'rest')])
def test_translation_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(TranslationServiceClient, transports.TranslationServiceGrpcTransport, 'grpc', grpc_helpers), (TranslationServiceAsyncClient, transports.TranslationServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (TranslationServiceClient, transports.TranslationServiceRestTransport, 'rest', None)])
def test_translation_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_translation_service_client_client_options_from_dict():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.translate_v3beta1.services.translation_service.transports.TranslationServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = TranslationServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(TranslationServiceClient, transports.TranslationServiceGrpcTransport, 'grpc', grpc_helpers), (TranslationServiceAsyncClient, transports.TranslationServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_translation_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('translate.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-translation'), scopes=None, default_host='translate.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [translation_service.TranslateTextRequest, dict])
def test_translate_text(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.translate_text), '__call__') as call:
        call.return_value = translation_service.TranslateTextResponse()
        response = client.translate_text(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.TranslateTextRequest()
    assert isinstance(response, translation_service.TranslateTextResponse)

def test_translate_text_empty_call():
    if False:
        i = 10
        return i + 15
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.translate_text), '__call__') as call:
        client.translate_text()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.TranslateTextRequest()

@pytest.mark.asyncio
async def test_translate_text_async(transport: str='grpc_asyncio', request_type=translation_service.TranslateTextRequest):
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.translate_text), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(translation_service.TranslateTextResponse())
        response = await client.translate_text(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.TranslateTextRequest()
    assert isinstance(response, translation_service.TranslateTextResponse)

@pytest.mark.asyncio
async def test_translate_text_async_from_dict():
    await test_translate_text_async(request_type=dict)

def test_translate_text_field_headers():
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.TranslateTextRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.translate_text), '__call__') as call:
        call.return_value = translation_service.TranslateTextResponse()
        client.translate_text(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_translate_text_field_headers_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.TranslateTextRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.translate_text), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(translation_service.TranslateTextResponse())
        await client.translate_text(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [translation_service.DetectLanguageRequest, dict])
def test_detect_language(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.detect_language), '__call__') as call:
        call.return_value = translation_service.DetectLanguageResponse()
        response = client.detect_language(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.DetectLanguageRequest()
    assert isinstance(response, translation_service.DetectLanguageResponse)

def test_detect_language_empty_call():
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.detect_language), '__call__') as call:
        client.detect_language()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.DetectLanguageRequest()

@pytest.mark.asyncio
async def test_detect_language_async(transport: str='grpc_asyncio', request_type=translation_service.DetectLanguageRequest):
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.detect_language), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(translation_service.DetectLanguageResponse())
        response = await client.detect_language(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.DetectLanguageRequest()
    assert isinstance(response, translation_service.DetectLanguageResponse)

@pytest.mark.asyncio
async def test_detect_language_async_from_dict():
    await test_detect_language_async(request_type=dict)

def test_detect_language_field_headers():
    if False:
        return 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.DetectLanguageRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.detect_language), '__call__') as call:
        call.return_value = translation_service.DetectLanguageResponse()
        client.detect_language(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_detect_language_field_headers_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.DetectLanguageRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.detect_language), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(translation_service.DetectLanguageResponse())
        await client.detect_language(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_detect_language_flattened():
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.detect_language), '__call__') as call:
        call.return_value = translation_service.DetectLanguageResponse()
        client.detect_language(parent='parent_value', model='model_value', mime_type='mime_type_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].model
        mock_val = 'model_value'
        assert arg == mock_val
        arg = args[0].mime_type
        mock_val = 'mime_type_value'
        assert arg == mock_val

def test_detect_language_flattened_error():
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.detect_language(translation_service.DetectLanguageRequest(), parent='parent_value', model='model_value', mime_type='mime_type_value')

@pytest.mark.asyncio
async def test_detect_language_flattened_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.detect_language), '__call__') as call:
        call.return_value = translation_service.DetectLanguageResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(translation_service.DetectLanguageResponse())
        response = await client.detect_language(parent='parent_value', model='model_value', mime_type='mime_type_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].model
        mock_val = 'model_value'
        assert arg == mock_val
        arg = args[0].mime_type
        mock_val = 'mime_type_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_detect_language_flattened_error_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.detect_language(translation_service.DetectLanguageRequest(), parent='parent_value', model='model_value', mime_type='mime_type_value')

@pytest.mark.parametrize('request_type', [translation_service.GetSupportedLanguagesRequest, dict])
def test_get_supported_languages(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_supported_languages), '__call__') as call:
        call.return_value = translation_service.SupportedLanguages()
        response = client.get_supported_languages(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.GetSupportedLanguagesRequest()
    assert isinstance(response, translation_service.SupportedLanguages)

def test_get_supported_languages_empty_call():
    if False:
        i = 10
        return i + 15
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_supported_languages), '__call__') as call:
        client.get_supported_languages()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.GetSupportedLanguagesRequest()

@pytest.mark.asyncio
async def test_get_supported_languages_async(transport: str='grpc_asyncio', request_type=translation_service.GetSupportedLanguagesRequest):
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_supported_languages), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(translation_service.SupportedLanguages())
        response = await client.get_supported_languages(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.GetSupportedLanguagesRequest()
    assert isinstance(response, translation_service.SupportedLanguages)

@pytest.mark.asyncio
async def test_get_supported_languages_async_from_dict():
    await test_get_supported_languages_async(request_type=dict)

def test_get_supported_languages_field_headers():
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.GetSupportedLanguagesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.get_supported_languages), '__call__') as call:
        call.return_value = translation_service.SupportedLanguages()
        client.get_supported_languages(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_supported_languages_field_headers_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.GetSupportedLanguagesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.get_supported_languages), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(translation_service.SupportedLanguages())
        await client.get_supported_languages(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_get_supported_languages_flattened():
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_supported_languages), '__call__') as call:
        call.return_value = translation_service.SupportedLanguages()
        client.get_supported_languages(parent='parent_value', display_language_code='display_language_code_value', model='model_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].display_language_code
        mock_val = 'display_language_code_value'
        assert arg == mock_val
        arg = args[0].model
        mock_val = 'model_value'
        assert arg == mock_val

def test_get_supported_languages_flattened_error():
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_supported_languages(translation_service.GetSupportedLanguagesRequest(), parent='parent_value', display_language_code='display_language_code_value', model='model_value')

@pytest.mark.asyncio
async def test_get_supported_languages_flattened_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_supported_languages), '__call__') as call:
        call.return_value = translation_service.SupportedLanguages()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(translation_service.SupportedLanguages())
        response = await client.get_supported_languages(parent='parent_value', display_language_code='display_language_code_value', model='model_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].display_language_code
        mock_val = 'display_language_code_value'
        assert arg == mock_val
        arg = args[0].model
        mock_val = 'model_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_supported_languages_flattened_error_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_supported_languages(translation_service.GetSupportedLanguagesRequest(), parent='parent_value', display_language_code='display_language_code_value', model='model_value')

@pytest.mark.parametrize('request_type', [translation_service.TranslateDocumentRequest, dict])
def test_translate_document(request_type, transport: str='grpc'):
    if False:
        return 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.translate_document), '__call__') as call:
        call.return_value = translation_service.TranslateDocumentResponse(model='model_value')
        response = client.translate_document(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.TranslateDocumentRequest()
    assert isinstance(response, translation_service.TranslateDocumentResponse)
    assert response.model == 'model_value'

def test_translate_document_empty_call():
    if False:
        i = 10
        return i + 15
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.translate_document), '__call__') as call:
        client.translate_document()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.TranslateDocumentRequest()

@pytest.mark.asyncio
async def test_translate_document_async(transport: str='grpc_asyncio', request_type=translation_service.TranslateDocumentRequest):
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.translate_document), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(translation_service.TranslateDocumentResponse(model='model_value'))
        response = await client.translate_document(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.TranslateDocumentRequest()
    assert isinstance(response, translation_service.TranslateDocumentResponse)
    assert response.model == 'model_value'

@pytest.mark.asyncio
async def test_translate_document_async_from_dict():
    await test_translate_document_async(request_type=dict)

def test_translate_document_field_headers():
    if False:
        return 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.TranslateDocumentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.translate_document), '__call__') as call:
        call.return_value = translation_service.TranslateDocumentResponse()
        client.translate_document(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_translate_document_field_headers_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.TranslateDocumentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.translate_document), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(translation_service.TranslateDocumentResponse())
        await client.translate_document(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [translation_service.BatchTranslateTextRequest, dict])
def test_batch_translate_text(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_translate_text), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.batch_translate_text(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.BatchTranslateTextRequest()
    assert isinstance(response, future.Future)

def test_batch_translate_text_empty_call():
    if False:
        return 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_translate_text), '__call__') as call:
        client.batch_translate_text()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.BatchTranslateTextRequest()

@pytest.mark.asyncio
async def test_batch_translate_text_async(transport: str='grpc_asyncio', request_type=translation_service.BatchTranslateTextRequest):
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_translate_text), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_translate_text(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.BatchTranslateTextRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_batch_translate_text_async_from_dict():
    await test_batch_translate_text_async(request_type=dict)

def test_batch_translate_text_field_headers():
    if False:
        return 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.BatchTranslateTextRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_translate_text), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_translate_text(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_translate_text_field_headers_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.BatchTranslateTextRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_translate_text), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.batch_translate_text(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [translation_service.BatchTranslateDocumentRequest, dict])
def test_batch_translate_document(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_translate_document), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.batch_translate_document(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.BatchTranslateDocumentRequest()
    assert isinstance(response, future.Future)

def test_batch_translate_document_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_translate_document), '__call__') as call:
        client.batch_translate_document()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.BatchTranslateDocumentRequest()

@pytest.mark.asyncio
async def test_batch_translate_document_async(transport: str='grpc_asyncio', request_type=translation_service.BatchTranslateDocumentRequest):
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_translate_document), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_translate_document(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.BatchTranslateDocumentRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_batch_translate_document_async_from_dict():
    await test_batch_translate_document_async(request_type=dict)

def test_batch_translate_document_field_headers():
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.BatchTranslateDocumentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_translate_document), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_translate_document(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_batch_translate_document_field_headers_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.BatchTranslateDocumentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.batch_translate_document), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.batch_translate_document(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_batch_translate_document_flattened():
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_translate_document), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.batch_translate_document(parent='parent_value', source_language_code='source_language_code_value', target_language_codes=['target_language_codes_value'], input_configs=[translation_service.BatchDocumentInputConfig(gcs_source=translation_service.GcsSource(input_uri='input_uri_value'))], output_config=translation_service.BatchDocumentOutputConfig(gcs_destination=translation_service.GcsDestination(output_uri_prefix='output_uri_prefix_value')))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].source_language_code
        mock_val = 'source_language_code_value'
        assert arg == mock_val
        arg = args[0].target_language_codes
        mock_val = ['target_language_codes_value']
        assert arg == mock_val
        arg = args[0].input_configs
        mock_val = [translation_service.BatchDocumentInputConfig(gcs_source=translation_service.GcsSource(input_uri='input_uri_value'))]
        assert arg == mock_val
        arg = args[0].output_config
        mock_val = translation_service.BatchDocumentOutputConfig(gcs_destination=translation_service.GcsDestination(output_uri_prefix='output_uri_prefix_value'))
        assert arg == mock_val

def test_batch_translate_document_flattened_error():
    if False:
        i = 10
        return i + 15
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.batch_translate_document(translation_service.BatchTranslateDocumentRequest(), parent='parent_value', source_language_code='source_language_code_value', target_language_codes=['target_language_codes_value'], input_configs=[translation_service.BatchDocumentInputConfig(gcs_source=translation_service.GcsSource(input_uri='input_uri_value'))], output_config=translation_service.BatchDocumentOutputConfig(gcs_destination=translation_service.GcsDestination(output_uri_prefix='output_uri_prefix_value')))

@pytest.mark.asyncio
async def test_batch_translate_document_flattened_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_translate_document), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.batch_translate_document(parent='parent_value', source_language_code='source_language_code_value', target_language_codes=['target_language_codes_value'], input_configs=[translation_service.BatchDocumentInputConfig(gcs_source=translation_service.GcsSource(input_uri='input_uri_value'))], output_config=translation_service.BatchDocumentOutputConfig(gcs_destination=translation_service.GcsDestination(output_uri_prefix='output_uri_prefix_value')))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].source_language_code
        mock_val = 'source_language_code_value'
        assert arg == mock_val
        arg = args[0].target_language_codes
        mock_val = ['target_language_codes_value']
        assert arg == mock_val
        arg = args[0].input_configs
        mock_val = [translation_service.BatchDocumentInputConfig(gcs_source=translation_service.GcsSource(input_uri='input_uri_value'))]
        assert arg == mock_val
        arg = args[0].output_config
        mock_val = translation_service.BatchDocumentOutputConfig(gcs_destination=translation_service.GcsDestination(output_uri_prefix='output_uri_prefix_value'))
        assert arg == mock_val

@pytest.mark.asyncio
async def test_batch_translate_document_flattened_error_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.batch_translate_document(translation_service.BatchTranslateDocumentRequest(), parent='parent_value', source_language_code='source_language_code_value', target_language_codes=['target_language_codes_value'], input_configs=[translation_service.BatchDocumentInputConfig(gcs_source=translation_service.GcsSource(input_uri='input_uri_value'))], output_config=translation_service.BatchDocumentOutputConfig(gcs_destination=translation_service.GcsDestination(output_uri_prefix='output_uri_prefix_value')))

@pytest.mark.parametrize('request_type', [translation_service.CreateGlossaryRequest, dict])
def test_create_glossary(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_glossary), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_glossary(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.CreateGlossaryRequest()
    assert isinstance(response, future.Future)

def test_create_glossary_empty_call():
    if False:
        return 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_glossary), '__call__') as call:
        client.create_glossary()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.CreateGlossaryRequest()

@pytest.mark.asyncio
async def test_create_glossary_async(transport: str='grpc_asyncio', request_type=translation_service.CreateGlossaryRequest):
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_glossary), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_glossary(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.CreateGlossaryRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_glossary_async_from_dict():
    await test_create_glossary_async(request_type=dict)

def test_create_glossary_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.CreateGlossaryRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_glossary), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_glossary(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_glossary_field_headers_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.CreateGlossaryRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_glossary), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_glossary(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_glossary_flattened():
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_glossary), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_glossary(parent='parent_value', glossary=translation_service.Glossary(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].glossary
        mock_val = translation_service.Glossary(name='name_value')
        assert arg == mock_val

def test_create_glossary_flattened_error():
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_glossary(translation_service.CreateGlossaryRequest(), parent='parent_value', glossary=translation_service.Glossary(name='name_value'))

@pytest.mark.asyncio
async def test_create_glossary_flattened_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_glossary), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_glossary(parent='parent_value', glossary=translation_service.Glossary(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].glossary
        mock_val = translation_service.Glossary(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_glossary_flattened_error_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_glossary(translation_service.CreateGlossaryRequest(), parent='parent_value', glossary=translation_service.Glossary(name='name_value'))

@pytest.mark.parametrize('request_type', [translation_service.ListGlossariesRequest, dict])
def test_list_glossaries(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_glossaries), '__call__') as call:
        call.return_value = translation_service.ListGlossariesResponse(next_page_token='next_page_token_value')
        response = client.list_glossaries(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.ListGlossariesRequest()
    assert isinstance(response, pagers.ListGlossariesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_glossaries_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_glossaries), '__call__') as call:
        client.list_glossaries()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.ListGlossariesRequest()

@pytest.mark.asyncio
async def test_list_glossaries_async(transport: str='grpc_asyncio', request_type=translation_service.ListGlossariesRequest):
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_glossaries), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(translation_service.ListGlossariesResponse(next_page_token='next_page_token_value'))
        response = await client.list_glossaries(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.ListGlossariesRequest()
    assert isinstance(response, pagers.ListGlossariesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_glossaries_async_from_dict():
    await test_list_glossaries_async(request_type=dict)

def test_list_glossaries_field_headers():
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.ListGlossariesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_glossaries), '__call__') as call:
        call.return_value = translation_service.ListGlossariesResponse()
        client.list_glossaries(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_glossaries_field_headers_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.ListGlossariesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_glossaries), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(translation_service.ListGlossariesResponse())
        await client.list_glossaries(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_glossaries_flattened():
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_glossaries), '__call__') as call:
        call.return_value = translation_service.ListGlossariesResponse()
        client.list_glossaries(parent='parent_value', filter='filter_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

def test_list_glossaries_flattened_error():
    if False:
        i = 10
        return i + 15
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_glossaries(translation_service.ListGlossariesRequest(), parent='parent_value', filter='filter_value')

@pytest.mark.asyncio
async def test_list_glossaries_flattened_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_glossaries), '__call__') as call:
        call.return_value = translation_service.ListGlossariesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(translation_service.ListGlossariesResponse())
        response = await client.list_glossaries(parent='parent_value', filter='filter_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_glossaries_flattened_error_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_glossaries(translation_service.ListGlossariesRequest(), parent='parent_value', filter='filter_value')

def test_list_glossaries_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_glossaries), '__call__') as call:
        call.side_effect = (translation_service.ListGlossariesResponse(glossaries=[translation_service.Glossary(), translation_service.Glossary(), translation_service.Glossary()], next_page_token='abc'), translation_service.ListGlossariesResponse(glossaries=[], next_page_token='def'), translation_service.ListGlossariesResponse(glossaries=[translation_service.Glossary()], next_page_token='ghi'), translation_service.ListGlossariesResponse(glossaries=[translation_service.Glossary(), translation_service.Glossary()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_glossaries(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, translation_service.Glossary) for i in results))

def test_list_glossaries_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_glossaries), '__call__') as call:
        call.side_effect = (translation_service.ListGlossariesResponse(glossaries=[translation_service.Glossary(), translation_service.Glossary(), translation_service.Glossary()], next_page_token='abc'), translation_service.ListGlossariesResponse(glossaries=[], next_page_token='def'), translation_service.ListGlossariesResponse(glossaries=[translation_service.Glossary()], next_page_token='ghi'), translation_service.ListGlossariesResponse(glossaries=[translation_service.Glossary(), translation_service.Glossary()]), RuntimeError)
        pages = list(client.list_glossaries(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_glossaries_async_pager():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_glossaries), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (translation_service.ListGlossariesResponse(glossaries=[translation_service.Glossary(), translation_service.Glossary(), translation_service.Glossary()], next_page_token='abc'), translation_service.ListGlossariesResponse(glossaries=[], next_page_token='def'), translation_service.ListGlossariesResponse(glossaries=[translation_service.Glossary()], next_page_token='ghi'), translation_service.ListGlossariesResponse(glossaries=[translation_service.Glossary(), translation_service.Glossary()]), RuntimeError)
        async_pager = await client.list_glossaries(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, translation_service.Glossary) for i in responses))

@pytest.mark.asyncio
async def test_list_glossaries_async_pages():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_glossaries), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (translation_service.ListGlossariesResponse(glossaries=[translation_service.Glossary(), translation_service.Glossary(), translation_service.Glossary()], next_page_token='abc'), translation_service.ListGlossariesResponse(glossaries=[], next_page_token='def'), translation_service.ListGlossariesResponse(glossaries=[translation_service.Glossary()], next_page_token='ghi'), translation_service.ListGlossariesResponse(glossaries=[translation_service.Glossary(), translation_service.Glossary()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_glossaries(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [translation_service.GetGlossaryRequest, dict])
def test_get_glossary(request_type, transport: str='grpc'):
    if False:
        return 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_glossary), '__call__') as call:
        call.return_value = translation_service.Glossary(name='name_value', entry_count=1210)
        response = client.get_glossary(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.GetGlossaryRequest()
    assert isinstance(response, translation_service.Glossary)
    assert response.name == 'name_value'
    assert response.entry_count == 1210

def test_get_glossary_empty_call():
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_glossary), '__call__') as call:
        client.get_glossary()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.GetGlossaryRequest()

@pytest.mark.asyncio
async def test_get_glossary_async(transport: str='grpc_asyncio', request_type=translation_service.GetGlossaryRequest):
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_glossary), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(translation_service.Glossary(name='name_value', entry_count=1210))
        response = await client.get_glossary(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.GetGlossaryRequest()
    assert isinstance(response, translation_service.Glossary)
    assert response.name == 'name_value'
    assert response.entry_count == 1210

@pytest.mark.asyncio
async def test_get_glossary_async_from_dict():
    await test_get_glossary_async(request_type=dict)

def test_get_glossary_field_headers():
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.GetGlossaryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_glossary), '__call__') as call:
        call.return_value = translation_service.Glossary()
        client.get_glossary(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_glossary_field_headers_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.GetGlossaryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_glossary), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(translation_service.Glossary())
        await client.get_glossary(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_glossary_flattened():
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_glossary), '__call__') as call:
        call.return_value = translation_service.Glossary()
        client.get_glossary(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_glossary_flattened_error():
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_glossary(translation_service.GetGlossaryRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_glossary_flattened_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_glossary), '__call__') as call:
        call.return_value = translation_service.Glossary()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(translation_service.Glossary())
        response = await client.get_glossary(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_glossary_flattened_error_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_glossary(translation_service.GetGlossaryRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [translation_service.DeleteGlossaryRequest, dict])
def test_delete_glossary(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_glossary), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_glossary(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.DeleteGlossaryRequest()
    assert isinstance(response, future.Future)

def test_delete_glossary_empty_call():
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_glossary), '__call__') as call:
        client.delete_glossary()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.DeleteGlossaryRequest()

@pytest.mark.asyncio
async def test_delete_glossary_async(transport: str='grpc_asyncio', request_type=translation_service.DeleteGlossaryRequest):
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_glossary), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_glossary(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == translation_service.DeleteGlossaryRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_glossary_async_from_dict():
    await test_delete_glossary_async(request_type=dict)

def test_delete_glossary_field_headers():
    if False:
        i = 10
        return i + 15
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.DeleteGlossaryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_glossary), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_glossary(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_glossary_field_headers_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = translation_service.DeleteGlossaryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_glossary), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_glossary(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_glossary_flattened():
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_glossary), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_glossary(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_glossary_flattened_error():
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_glossary(translation_service.DeleteGlossaryRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_glossary_flattened_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_glossary), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_glossary(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_glossary_flattened_error_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_glossary(translation_service.DeleteGlossaryRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [translation_service.TranslateTextRequest, dict])
def test_translate_text_rest(request_type):
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = translation_service.TranslateTextResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = translation_service.TranslateTextResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.translate_text(request)
    assert isinstance(response, translation_service.TranslateTextResponse)

def test_translate_text_rest_required_fields(request_type=translation_service.TranslateTextRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.TranslationServiceRestTransport
    request_init = {}
    request_init['contents'] = ''
    request_init['target_language_code'] = ''
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).translate_text._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['contents'] = 'contents_value'
    jsonified_request['targetLanguageCode'] = 'target_language_code_value'
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).translate_text._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'contents' in jsonified_request
    assert jsonified_request['contents'] == 'contents_value'
    assert 'targetLanguageCode' in jsonified_request
    assert jsonified_request['targetLanguageCode'] == 'target_language_code_value'
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = translation_service.TranslateTextResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = translation_service.TranslateTextResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.translate_text(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_translate_text_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.translate_text._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('contents', 'targetLanguageCode', 'parent'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_translate_text_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TranslationServiceRestInterceptor())
    client = TranslationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TranslationServiceRestInterceptor, 'post_translate_text') as post, mock.patch.object(transports.TranslationServiceRestInterceptor, 'pre_translate_text') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = translation_service.TranslateTextRequest.pb(translation_service.TranslateTextRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = translation_service.TranslateTextResponse.to_json(translation_service.TranslateTextResponse())
        request = translation_service.TranslateTextRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = translation_service.TranslateTextResponse()
        client.translate_text(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_translate_text_rest_bad_request(transport: str='rest', request_type=translation_service.TranslateTextRequest):
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.translate_text(request)

def test_translate_text_rest_error():
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [translation_service.DetectLanguageRequest, dict])
def test_detect_language_rest(request_type):
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = translation_service.DetectLanguageResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = translation_service.DetectLanguageResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.detect_language(request)
    assert isinstance(response, translation_service.DetectLanguageResponse)

def test_detect_language_rest_required_fields(request_type=translation_service.DetectLanguageRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.TranslationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).detect_language._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).detect_language._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = translation_service.DetectLanguageResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = translation_service.DetectLanguageResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.detect_language(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_detect_language_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.detect_language._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_detect_language_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TranslationServiceRestInterceptor())
    client = TranslationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TranslationServiceRestInterceptor, 'post_detect_language') as post, mock.patch.object(transports.TranslationServiceRestInterceptor, 'pre_detect_language') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = translation_service.DetectLanguageRequest.pb(translation_service.DetectLanguageRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = translation_service.DetectLanguageResponse.to_json(translation_service.DetectLanguageResponse())
        request = translation_service.DetectLanguageRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = translation_service.DetectLanguageResponse()
        client.detect_language(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_detect_language_rest_bad_request(transport: str='rest', request_type=translation_service.DetectLanguageRequest):
    if False:
        for i in range(10):
            print('nop')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.detect_language(request)

def test_detect_language_rest_flattened():
    if False:
        return 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = translation_service.DetectLanguageResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', model='model_value', mime_type='mime_type_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = translation_service.DetectLanguageResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.detect_language(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{parent=projects/*/locations/*}:detectLanguage' % client.transport._host, args[1])

def test_detect_language_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.detect_language(translation_service.DetectLanguageRequest(), parent='parent_value', model='model_value', mime_type='mime_type_value')

def test_detect_language_rest_error():
    if False:
        return 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [translation_service.GetSupportedLanguagesRequest, dict])
def test_get_supported_languages_rest(request_type):
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = translation_service.SupportedLanguages()
        response_value = Response()
        response_value.status_code = 200
        return_value = translation_service.SupportedLanguages.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_supported_languages(request)
    assert isinstance(response, translation_service.SupportedLanguages)

def test_get_supported_languages_rest_required_fields(request_type=translation_service.GetSupportedLanguagesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.TranslationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_supported_languages._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_supported_languages._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('display_language_code', 'model'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = translation_service.SupportedLanguages()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = translation_service.SupportedLanguages.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_supported_languages(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_supported_languages_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_supported_languages._get_unset_required_fields({})
    assert set(unset_fields) == set(('displayLanguageCode', 'model')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_supported_languages_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TranslationServiceRestInterceptor())
    client = TranslationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TranslationServiceRestInterceptor, 'post_get_supported_languages') as post, mock.patch.object(transports.TranslationServiceRestInterceptor, 'pre_get_supported_languages') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = translation_service.GetSupportedLanguagesRequest.pb(translation_service.GetSupportedLanguagesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = translation_service.SupportedLanguages.to_json(translation_service.SupportedLanguages())
        request = translation_service.GetSupportedLanguagesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = translation_service.SupportedLanguages()
        client.get_supported_languages(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_supported_languages_rest_bad_request(transport: str='rest', request_type=translation_service.GetSupportedLanguagesRequest):
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_supported_languages(request)

def test_get_supported_languages_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = translation_service.SupportedLanguages()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', display_language_code='display_language_code_value', model='model_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = translation_service.SupportedLanguages.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_supported_languages(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{parent=projects/*/locations/*}/supportedLanguages' % client.transport._host, args[1])

def test_get_supported_languages_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_supported_languages(translation_service.GetSupportedLanguagesRequest(), parent='parent_value', display_language_code='display_language_code_value', model='model_value')

def test_get_supported_languages_rest_error():
    if False:
        i = 10
        return i + 15
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [translation_service.TranslateDocumentRequest, dict])
def test_translate_document_rest(request_type):
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = translation_service.TranslateDocumentResponse(model='model_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = translation_service.TranslateDocumentResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.translate_document(request)
    assert isinstance(response, translation_service.TranslateDocumentResponse)
    assert response.model == 'model_value'

def test_translate_document_rest_required_fields(request_type=translation_service.TranslateDocumentRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.TranslationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['target_language_code'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).translate_document._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['targetLanguageCode'] = 'target_language_code_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).translate_document._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'targetLanguageCode' in jsonified_request
    assert jsonified_request['targetLanguageCode'] == 'target_language_code_value'
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = translation_service.TranslateDocumentResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = translation_service.TranslateDocumentResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.translate_document(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_translate_document_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.translate_document._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'targetLanguageCode', 'documentInputConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_translate_document_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TranslationServiceRestInterceptor())
    client = TranslationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TranslationServiceRestInterceptor, 'post_translate_document') as post, mock.patch.object(transports.TranslationServiceRestInterceptor, 'pre_translate_document') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = translation_service.TranslateDocumentRequest.pb(translation_service.TranslateDocumentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = translation_service.TranslateDocumentResponse.to_json(translation_service.TranslateDocumentResponse())
        request = translation_service.TranslateDocumentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = translation_service.TranslateDocumentResponse()
        client.translate_document(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_translate_document_rest_bad_request(transport: str='rest', request_type=translation_service.TranslateDocumentRequest):
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.translate_document(request)

def test_translate_document_rest_error():
    if False:
        return 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [translation_service.BatchTranslateTextRequest, dict])
def test_batch_translate_text_rest(request_type):
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_translate_text(request)
    assert response.operation.name == 'operations/spam'

def test_batch_translate_text_rest_required_fields(request_type=translation_service.BatchTranslateTextRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.TranslationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['source_language_code'] = ''
    request_init['target_language_codes'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_translate_text._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['sourceLanguageCode'] = 'source_language_code_value'
    jsonified_request['targetLanguageCodes'] = 'target_language_codes_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_translate_text._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'sourceLanguageCode' in jsonified_request
    assert jsonified_request['sourceLanguageCode'] == 'source_language_code_value'
    assert 'targetLanguageCodes' in jsonified_request
    assert jsonified_request['targetLanguageCodes'] == 'target_language_codes_value'
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.batch_translate_text(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_translate_text_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_translate_text._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'sourceLanguageCode', 'targetLanguageCodes', 'inputConfigs', 'outputConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_translate_text_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TranslationServiceRestInterceptor())
    client = TranslationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.TranslationServiceRestInterceptor, 'post_batch_translate_text') as post, mock.patch.object(transports.TranslationServiceRestInterceptor, 'pre_batch_translate_text') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = translation_service.BatchTranslateTextRequest.pb(translation_service.BatchTranslateTextRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = translation_service.BatchTranslateTextRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.batch_translate_text(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_translate_text_rest_bad_request(transport: str='rest', request_type=translation_service.BatchTranslateTextRequest):
    if False:
        return 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_translate_text(request)

def test_batch_translate_text_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [translation_service.BatchTranslateDocumentRequest, dict])
def test_batch_translate_document_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_translate_document(request)
    assert response.operation.name == 'operations/spam'

def test_batch_translate_document_rest_required_fields(request_type=translation_service.BatchTranslateDocumentRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.TranslationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['source_language_code'] = ''
    request_init['target_language_codes'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_translate_document._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['sourceLanguageCode'] = 'source_language_code_value'
    jsonified_request['targetLanguageCodes'] = 'target_language_codes_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_translate_document._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'sourceLanguageCode' in jsonified_request
    assert jsonified_request['sourceLanguageCode'] == 'source_language_code_value'
    assert 'targetLanguageCodes' in jsonified_request
    assert jsonified_request['targetLanguageCodes'] == 'target_language_codes_value'
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.batch_translate_document(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_translate_document_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_translate_document._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'sourceLanguageCode', 'targetLanguageCodes', 'inputConfigs', 'outputConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_translate_document_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TranslationServiceRestInterceptor())
    client = TranslationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.TranslationServiceRestInterceptor, 'post_batch_translate_document') as post, mock.patch.object(transports.TranslationServiceRestInterceptor, 'pre_batch_translate_document') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = translation_service.BatchTranslateDocumentRequest.pb(translation_service.BatchTranslateDocumentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = translation_service.BatchTranslateDocumentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.batch_translate_document(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_translate_document_rest_bad_request(transport: str='rest', request_type=translation_service.BatchTranslateDocumentRequest):
    if False:
        return 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_translate_document(request)

def test_batch_translate_document_rest_flattened():
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', source_language_code='source_language_code_value', target_language_codes=['target_language_codes_value'], input_configs=[translation_service.BatchDocumentInputConfig(gcs_source=translation_service.GcsSource(input_uri='input_uri_value'))], output_config=translation_service.BatchDocumentOutputConfig(gcs_destination=translation_service.GcsDestination(output_uri_prefix='output_uri_prefix_value')))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.batch_translate_document(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{parent=projects/*/locations/*}:batchTranslateDocument' % client.transport._host, args[1])

def test_batch_translate_document_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.batch_translate_document(translation_service.BatchTranslateDocumentRequest(), parent='parent_value', source_language_code='source_language_code_value', target_language_codes=['target_language_codes_value'], input_configs=[translation_service.BatchDocumentInputConfig(gcs_source=translation_service.GcsSource(input_uri='input_uri_value'))], output_config=translation_service.BatchDocumentOutputConfig(gcs_destination=translation_service.GcsDestination(output_uri_prefix='output_uri_prefix_value')))

def test_batch_translate_document_rest_error():
    if False:
        return 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [translation_service.CreateGlossaryRequest, dict])
def test_create_glossary_rest(request_type):
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['glossary'] = {'name': 'name_value', 'language_pair': {'source_language_code': 'source_language_code_value', 'target_language_code': 'target_language_code_value'}, 'language_codes_set': {'language_codes': ['language_codes_value1', 'language_codes_value2']}, 'input_config': {'gcs_source': {'input_uri': 'input_uri_value'}}, 'entry_count': 1210, 'submit_time': {'seconds': 751, 'nanos': 543}, 'end_time': {}}
    test_field = translation_service.CreateGlossaryRequest.meta.fields['glossary']

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
    for (field, value) in request_init['glossary'].items():
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
                for i in range(0, len(request_init['glossary'][field])):
                    del request_init['glossary'][field][i][subfield]
            else:
                del request_init['glossary'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_glossary(request)
    assert response.operation.name == 'operations/spam'

def test_create_glossary_rest_required_fields(request_type=translation_service.CreateGlossaryRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.TranslationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_glossary._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_glossary._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_glossary(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_glossary_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_glossary._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'glossary'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_glossary_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TranslationServiceRestInterceptor())
    client = TranslationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.TranslationServiceRestInterceptor, 'post_create_glossary') as post, mock.patch.object(transports.TranslationServiceRestInterceptor, 'pre_create_glossary') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = translation_service.CreateGlossaryRequest.pb(translation_service.CreateGlossaryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = translation_service.CreateGlossaryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_glossary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_glossary_rest_bad_request(transport: str='rest', request_type=translation_service.CreateGlossaryRequest):
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_glossary(request)

def test_create_glossary_rest_flattened():
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', glossary=translation_service.Glossary(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_glossary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{parent=projects/*/locations/*}/glossaries' % client.transport._host, args[1])

def test_create_glossary_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_glossary(translation_service.CreateGlossaryRequest(), parent='parent_value', glossary=translation_service.Glossary(name='name_value'))

def test_create_glossary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [translation_service.ListGlossariesRequest, dict])
def test_list_glossaries_rest(request_type):
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = translation_service.ListGlossariesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = translation_service.ListGlossariesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_glossaries(request)
    assert isinstance(response, pagers.ListGlossariesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_glossaries_rest_required_fields(request_type=translation_service.ListGlossariesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.TranslationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_glossaries._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_glossaries._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = translation_service.ListGlossariesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = translation_service.ListGlossariesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_glossaries(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_glossaries_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_glossaries._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_glossaries_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TranslationServiceRestInterceptor())
    client = TranslationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TranslationServiceRestInterceptor, 'post_list_glossaries') as post, mock.patch.object(transports.TranslationServiceRestInterceptor, 'pre_list_glossaries') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = translation_service.ListGlossariesRequest.pb(translation_service.ListGlossariesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = translation_service.ListGlossariesResponse.to_json(translation_service.ListGlossariesResponse())
        request = translation_service.ListGlossariesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = translation_service.ListGlossariesResponse()
        client.list_glossaries(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_glossaries_rest_bad_request(transport: str='rest', request_type=translation_service.ListGlossariesRequest):
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_glossaries(request)

def test_list_glossaries_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = translation_service.ListGlossariesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', filter='filter_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = translation_service.ListGlossariesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_glossaries(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{parent=projects/*/locations/*}/glossaries' % client.transport._host, args[1])

def test_list_glossaries_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_glossaries(translation_service.ListGlossariesRequest(), parent='parent_value', filter='filter_value')

def test_list_glossaries_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (translation_service.ListGlossariesResponse(glossaries=[translation_service.Glossary(), translation_service.Glossary(), translation_service.Glossary()], next_page_token='abc'), translation_service.ListGlossariesResponse(glossaries=[], next_page_token='def'), translation_service.ListGlossariesResponse(glossaries=[translation_service.Glossary()], next_page_token='ghi'), translation_service.ListGlossariesResponse(glossaries=[translation_service.Glossary(), translation_service.Glossary()]))
        response = response + response
        response = tuple((translation_service.ListGlossariesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_glossaries(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, translation_service.Glossary) for i in results))
        pages = list(client.list_glossaries(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [translation_service.GetGlossaryRequest, dict])
def test_get_glossary_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/glossaries/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = translation_service.Glossary(name='name_value', entry_count=1210)
        response_value = Response()
        response_value.status_code = 200
        return_value = translation_service.Glossary.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_glossary(request)
    assert isinstance(response, translation_service.Glossary)
    assert response.name == 'name_value'
    assert response.entry_count == 1210

def test_get_glossary_rest_required_fields(request_type=translation_service.GetGlossaryRequest):
    if False:
        return 10
    transport_class = transports.TranslationServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_glossary._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_glossary._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = translation_service.Glossary()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = translation_service.Glossary.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_glossary(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_glossary_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_glossary._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_glossary_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TranslationServiceRestInterceptor())
    client = TranslationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TranslationServiceRestInterceptor, 'post_get_glossary') as post, mock.patch.object(transports.TranslationServiceRestInterceptor, 'pre_get_glossary') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = translation_service.GetGlossaryRequest.pb(translation_service.GetGlossaryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = translation_service.Glossary.to_json(translation_service.Glossary())
        request = translation_service.GetGlossaryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = translation_service.Glossary()
        client.get_glossary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_glossary_rest_bad_request(transport: str='rest', request_type=translation_service.GetGlossaryRequest):
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/glossaries/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_glossary(request)

def test_get_glossary_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = translation_service.Glossary()
        sample_request = {'name': 'projects/sample1/locations/sample2/glossaries/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = translation_service.Glossary.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_glossary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{name=projects/*/locations/*/glossaries/*}' % client.transport._host, args[1])

def test_get_glossary_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_glossary(translation_service.GetGlossaryRequest(), name='name_value')

def test_get_glossary_rest_error():
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [translation_service.DeleteGlossaryRequest, dict])
def test_delete_glossary_rest(request_type):
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/glossaries/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_glossary(request)
    assert response.operation.name == 'operations/spam'

def test_delete_glossary_rest_required_fields(request_type=translation_service.DeleteGlossaryRequest):
    if False:
        return 10
    transport_class = transports.TranslationServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_glossary._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_glossary._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_glossary(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_glossary_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_glossary._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_glossary_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.TranslationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TranslationServiceRestInterceptor())
    client = TranslationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.TranslationServiceRestInterceptor, 'post_delete_glossary') as post, mock.patch.object(transports.TranslationServiceRestInterceptor, 'pre_delete_glossary') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = translation_service.DeleteGlossaryRequest.pb(translation_service.DeleteGlossaryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = translation_service.DeleteGlossaryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_glossary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_glossary_rest_bad_request(transport: str='rest', request_type=translation_service.DeleteGlossaryRequest):
    if False:
        for i in range(10):
            print('nop')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/glossaries/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_glossary(request)

def test_delete_glossary_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/glossaries/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_glossary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{name=projects/*/locations/*/glossaries/*}' % client.transport._host, args[1])

def test_delete_glossary_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_glossary(translation_service.DeleteGlossaryRequest(), name='name_value')

def test_delete_glossary_rest_error():
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.TranslationServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.TranslationServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TranslationServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.TranslationServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = TranslationServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = TranslationServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.TranslationServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TranslationServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.TranslationServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = TranslationServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TranslationServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.TranslationServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.TranslationServiceGrpcTransport, transports.TranslationServiceGrpcAsyncIOTransport, transports.TranslationServiceRestTransport])
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
        for i in range(10):
            print('nop')
    transport = TranslationServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.TranslationServiceGrpcTransport)

def test_translation_service_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.TranslationServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_translation_service_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.translate_v3beta1.services.translation_service.transports.TranslationServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.TranslationServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('translate_text', 'detect_language', 'get_supported_languages', 'translate_document', 'batch_translate_text', 'batch_translate_document', 'create_glossary', 'list_glossaries', 'get_glossary', 'delete_glossary')
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

def test_translation_service_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.translate_v3beta1.services.translation_service.transports.TranslationServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.TranslationServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-translation'), quota_project_id='octopus')

def test_translation_service_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.translate_v3beta1.services.translation_service.transports.TranslationServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.TranslationServiceTransport()
        adc.assert_called_once()

def test_translation_service_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        TranslationServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-translation'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.TranslationServiceGrpcTransport, transports.TranslationServiceGrpcAsyncIOTransport])
def test_translation_service_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-translation'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.TranslationServiceGrpcTransport, transports.TranslationServiceGrpcAsyncIOTransport, transports.TranslationServiceRestTransport])
def test_translation_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.TranslationServiceGrpcTransport, grpc_helpers), (transports.TranslationServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_translation_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('translate.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-translation'), scopes=['1', '2'], default_host='translate.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.TranslationServiceGrpcTransport, transports.TranslationServiceGrpcAsyncIOTransport])
def test_translation_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_translation_service_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.TranslationServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_translation_service_rest_lro_client():
    if False:
        print('Hello World!')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_translation_service_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='translate.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('translate.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://translate.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_translation_service_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='translate.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('translate.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://translate.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_translation_service_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = TranslationServiceClient(credentials=creds1, transport=transport_name)
    client2 = TranslationServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.translate_text._session
    session2 = client2.transport.translate_text._session
    assert session1 != session2
    session1 = client1.transport.detect_language._session
    session2 = client2.transport.detect_language._session
    assert session1 != session2
    session1 = client1.transport.get_supported_languages._session
    session2 = client2.transport.get_supported_languages._session
    assert session1 != session2
    session1 = client1.transport.translate_document._session
    session2 = client2.transport.translate_document._session
    assert session1 != session2
    session1 = client1.transport.batch_translate_text._session
    session2 = client2.transport.batch_translate_text._session
    assert session1 != session2
    session1 = client1.transport.batch_translate_document._session
    session2 = client2.transport.batch_translate_document._session
    assert session1 != session2
    session1 = client1.transport.create_glossary._session
    session2 = client2.transport.create_glossary._session
    assert session1 != session2
    session1 = client1.transport.list_glossaries._session
    session2 = client2.transport.list_glossaries._session
    assert session1 != session2
    session1 = client1.transport.get_glossary._session
    session2 = client2.transport.get_glossary._session
    assert session1 != session2
    session1 = client1.transport.delete_glossary._session
    session2 = client2.transport.delete_glossary._session
    assert session1 != session2

def test_translation_service_grpc_transport_channel():
    if False:
        print('Hello World!')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.TranslationServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_translation_service_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.TranslationServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.TranslationServiceGrpcTransport, transports.TranslationServiceGrpcAsyncIOTransport])
def test_translation_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.TranslationServiceGrpcTransport, transports.TranslationServiceGrpcAsyncIOTransport])
def test_translation_service_transport_channel_mtls_with_adc(transport_class):
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

def test_translation_service_grpc_lro_client():
    if False:
        return 10
    client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_translation_service_grpc_lro_async_client():
    if False:
        for i in range(10):
            print('nop')
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_glossary_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    glossary = 'whelk'
    expected = 'projects/{project}/locations/{location}/glossaries/{glossary}'.format(project=project, location=location, glossary=glossary)
    actual = TranslationServiceClient.glossary_path(project, location, glossary)
    assert expected == actual

def test_parse_glossary_path():
    if False:
        print('Hello World!')
    expected = {'project': 'octopus', 'location': 'oyster', 'glossary': 'nudibranch'}
    path = TranslationServiceClient.glossary_path(**expected)
    actual = TranslationServiceClient.parse_glossary_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = TranslationServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'mussel'}
    path = TranslationServiceClient.common_billing_account_path(**expected)
    actual = TranslationServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = TranslationServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'nautilus'}
    path = TranslationServiceClient.common_folder_path(**expected)
    actual = TranslationServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = TranslationServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'abalone'}
    path = TranslationServiceClient.common_organization_path(**expected)
    actual = TranslationServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = TranslationServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'clam'}
    path = TranslationServiceClient.common_project_path(**expected)
    actual = TranslationServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = TranslationServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = TranslationServiceClient.common_location_path(**expected)
    actual = TranslationServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.TranslationServiceTransport, '_prep_wrapped_messages') as prep:
        client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.TranslationServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = TranslationServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = TranslationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = TranslationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(TranslationServiceClient, transports.TranslationServiceGrpcTransport), (TranslationServiceAsyncClient, transports.TranslationServiceGrpcAsyncIOTransport)])
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