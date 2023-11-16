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
from google.cloud.language_v1beta2.services.language_service import LanguageServiceAsyncClient, LanguageServiceClient, transports
from google.cloud.language_v1beta2.types import language_service

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        print('Hello World!')
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
    assert LanguageServiceClient._get_default_mtls_endpoint(None) is None
    assert LanguageServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert LanguageServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert LanguageServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert LanguageServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert LanguageServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(LanguageServiceClient, 'grpc'), (LanguageServiceAsyncClient, 'grpc_asyncio'), (LanguageServiceClient, 'rest')])
def test_language_service_client_from_service_account_info(client_class, transport_name):
    if False:
        print('Hello World!')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('language.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://language.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.LanguageServiceGrpcTransport, 'grpc'), (transports.LanguageServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.LanguageServiceRestTransport, 'rest')])
def test_language_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(LanguageServiceClient, 'grpc'), (LanguageServiceAsyncClient, 'grpc_asyncio'), (LanguageServiceClient, 'rest')])
def test_language_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('language.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://language.googleapis.com')

def test_language_service_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = LanguageServiceClient.get_transport_class()
    available_transports = [transports.LanguageServiceGrpcTransport, transports.LanguageServiceRestTransport]
    assert transport in available_transports
    transport = LanguageServiceClient.get_transport_class('grpc')
    assert transport == transports.LanguageServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(LanguageServiceClient, transports.LanguageServiceGrpcTransport, 'grpc'), (LanguageServiceAsyncClient, transports.LanguageServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (LanguageServiceClient, transports.LanguageServiceRestTransport, 'rest')])
@mock.patch.object(LanguageServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(LanguageServiceClient))
@mock.patch.object(LanguageServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(LanguageServiceAsyncClient))
def test_language_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(LanguageServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(LanguageServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(LanguageServiceClient, transports.LanguageServiceGrpcTransport, 'grpc', 'true'), (LanguageServiceAsyncClient, transports.LanguageServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (LanguageServiceClient, transports.LanguageServiceGrpcTransport, 'grpc', 'false'), (LanguageServiceAsyncClient, transports.LanguageServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (LanguageServiceClient, transports.LanguageServiceRestTransport, 'rest', 'true'), (LanguageServiceClient, transports.LanguageServiceRestTransport, 'rest', 'false')])
@mock.patch.object(LanguageServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(LanguageServiceClient))
@mock.patch.object(LanguageServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(LanguageServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_language_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [LanguageServiceClient, LanguageServiceAsyncClient])
@mock.patch.object(LanguageServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(LanguageServiceClient))
@mock.patch.object(LanguageServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(LanguageServiceAsyncClient))
def test_language_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(LanguageServiceClient, transports.LanguageServiceGrpcTransport, 'grpc'), (LanguageServiceAsyncClient, transports.LanguageServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (LanguageServiceClient, transports.LanguageServiceRestTransport, 'rest')])
def test_language_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(LanguageServiceClient, transports.LanguageServiceGrpcTransport, 'grpc', grpc_helpers), (LanguageServiceAsyncClient, transports.LanguageServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (LanguageServiceClient, transports.LanguageServiceRestTransport, 'rest', None)])
def test_language_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_language_service_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.language_v1beta2.services.language_service.transports.LanguageServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = LanguageServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(LanguageServiceClient, transports.LanguageServiceGrpcTransport, 'grpc', grpc_helpers), (LanguageServiceAsyncClient, transports.LanguageServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_language_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('language.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-language', 'https://www.googleapis.com/auth/cloud-platform'), scopes=None, default_host='language.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [language_service.AnalyzeSentimentRequest, dict])
def test_analyze_sentiment(request_type, transport: str='grpc'):
    if False:
        return 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_sentiment), '__call__') as call:
        call.return_value = language_service.AnalyzeSentimentResponse(language='language_value')
        response = client.analyze_sentiment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.AnalyzeSentimentRequest()
    assert isinstance(response, language_service.AnalyzeSentimentResponse)
    assert response.language == 'language_value'

def test_analyze_sentiment_empty_call():
    if False:
        print('Hello World!')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.analyze_sentiment), '__call__') as call:
        client.analyze_sentiment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.AnalyzeSentimentRequest()

@pytest.mark.asyncio
async def test_analyze_sentiment_async(transport: str='grpc_asyncio', request_type=language_service.AnalyzeSentimentRequest):
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_sentiment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(language_service.AnalyzeSentimentResponse(language='language_value'))
        response = await client.analyze_sentiment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.AnalyzeSentimentRequest()
    assert isinstance(response, language_service.AnalyzeSentimentResponse)
    assert response.language == 'language_value'

@pytest.mark.asyncio
async def test_analyze_sentiment_async_from_dict():
    await test_analyze_sentiment_async(request_type=dict)

def test_analyze_sentiment_flattened():
    if False:
        print('Hello World!')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.analyze_sentiment), '__call__') as call:
        call.return_value = language_service.AnalyzeSentimentResponse()
        client.analyze_sentiment(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].document
        mock_val = language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT)
        assert arg == mock_val
        arg = args[0].encoding_type
        mock_val = language_service.EncodingType.UTF8
        assert arg == mock_val

def test_analyze_sentiment_flattened_error():
    if False:
        i = 10
        return i + 15
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.analyze_sentiment(language_service.AnalyzeSentimentRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)

@pytest.mark.asyncio
async def test_analyze_sentiment_flattened_async():
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.analyze_sentiment), '__call__') as call:
        call.return_value = language_service.AnalyzeSentimentResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(language_service.AnalyzeSentimentResponse())
        response = await client.analyze_sentiment(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].document
        mock_val = language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT)
        assert arg == mock_val
        arg = args[0].encoding_type
        mock_val = language_service.EncodingType.UTF8
        assert arg == mock_val

@pytest.mark.asyncio
async def test_analyze_sentiment_flattened_error_async():
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.analyze_sentiment(language_service.AnalyzeSentimentRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)

@pytest.mark.parametrize('request_type', [language_service.AnalyzeEntitiesRequest, dict])
def test_analyze_entities(request_type, transport: str='grpc'):
    if False:
        return 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_entities), '__call__') as call:
        call.return_value = language_service.AnalyzeEntitiesResponse(language='language_value')
        response = client.analyze_entities(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.AnalyzeEntitiesRequest()
    assert isinstance(response, language_service.AnalyzeEntitiesResponse)
    assert response.language == 'language_value'

def test_analyze_entities_empty_call():
    if False:
        i = 10
        return i + 15
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.analyze_entities), '__call__') as call:
        client.analyze_entities()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.AnalyzeEntitiesRequest()

@pytest.mark.asyncio
async def test_analyze_entities_async(transport: str='grpc_asyncio', request_type=language_service.AnalyzeEntitiesRequest):
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_entities), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(language_service.AnalyzeEntitiesResponse(language='language_value'))
        response = await client.analyze_entities(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.AnalyzeEntitiesRequest()
    assert isinstance(response, language_service.AnalyzeEntitiesResponse)
    assert response.language == 'language_value'

@pytest.mark.asyncio
async def test_analyze_entities_async_from_dict():
    await test_analyze_entities_async(request_type=dict)

def test_analyze_entities_flattened():
    if False:
        i = 10
        return i + 15
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.analyze_entities), '__call__') as call:
        call.return_value = language_service.AnalyzeEntitiesResponse()
        client.analyze_entities(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].document
        mock_val = language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT)
        assert arg == mock_val
        arg = args[0].encoding_type
        mock_val = language_service.EncodingType.UTF8
        assert arg == mock_val

def test_analyze_entities_flattened_error():
    if False:
        i = 10
        return i + 15
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.analyze_entities(language_service.AnalyzeEntitiesRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)

@pytest.mark.asyncio
async def test_analyze_entities_flattened_async():
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.analyze_entities), '__call__') as call:
        call.return_value = language_service.AnalyzeEntitiesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(language_service.AnalyzeEntitiesResponse())
        response = await client.analyze_entities(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].document
        mock_val = language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT)
        assert arg == mock_val
        arg = args[0].encoding_type
        mock_val = language_service.EncodingType.UTF8
        assert arg == mock_val

@pytest.mark.asyncio
async def test_analyze_entities_flattened_error_async():
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.analyze_entities(language_service.AnalyzeEntitiesRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)

@pytest.mark.parametrize('request_type', [language_service.AnalyzeEntitySentimentRequest, dict])
def test_analyze_entity_sentiment(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_entity_sentiment), '__call__') as call:
        call.return_value = language_service.AnalyzeEntitySentimentResponse(language='language_value')
        response = client.analyze_entity_sentiment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.AnalyzeEntitySentimentRequest()
    assert isinstance(response, language_service.AnalyzeEntitySentimentResponse)
    assert response.language == 'language_value'

def test_analyze_entity_sentiment_empty_call():
    if False:
        return 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.analyze_entity_sentiment), '__call__') as call:
        client.analyze_entity_sentiment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.AnalyzeEntitySentimentRequest()

@pytest.mark.asyncio
async def test_analyze_entity_sentiment_async(transport: str='grpc_asyncio', request_type=language_service.AnalyzeEntitySentimentRequest):
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_entity_sentiment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(language_service.AnalyzeEntitySentimentResponse(language='language_value'))
        response = await client.analyze_entity_sentiment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.AnalyzeEntitySentimentRequest()
    assert isinstance(response, language_service.AnalyzeEntitySentimentResponse)
    assert response.language == 'language_value'

@pytest.mark.asyncio
async def test_analyze_entity_sentiment_async_from_dict():
    await test_analyze_entity_sentiment_async(request_type=dict)

def test_analyze_entity_sentiment_flattened():
    if False:
        print('Hello World!')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.analyze_entity_sentiment), '__call__') as call:
        call.return_value = language_service.AnalyzeEntitySentimentResponse()
        client.analyze_entity_sentiment(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].document
        mock_val = language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT)
        assert arg == mock_val
        arg = args[0].encoding_type
        mock_val = language_service.EncodingType.UTF8
        assert arg == mock_val

def test_analyze_entity_sentiment_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.analyze_entity_sentiment(language_service.AnalyzeEntitySentimentRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)

@pytest.mark.asyncio
async def test_analyze_entity_sentiment_flattened_async():
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.analyze_entity_sentiment), '__call__') as call:
        call.return_value = language_service.AnalyzeEntitySentimentResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(language_service.AnalyzeEntitySentimentResponse())
        response = await client.analyze_entity_sentiment(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].document
        mock_val = language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT)
        assert arg == mock_val
        arg = args[0].encoding_type
        mock_val = language_service.EncodingType.UTF8
        assert arg == mock_val

@pytest.mark.asyncio
async def test_analyze_entity_sentiment_flattened_error_async():
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.analyze_entity_sentiment(language_service.AnalyzeEntitySentimentRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)

@pytest.mark.parametrize('request_type', [language_service.AnalyzeSyntaxRequest, dict])
def test_analyze_syntax(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_syntax), '__call__') as call:
        call.return_value = language_service.AnalyzeSyntaxResponse(language='language_value')
        response = client.analyze_syntax(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.AnalyzeSyntaxRequest()
    assert isinstance(response, language_service.AnalyzeSyntaxResponse)
    assert response.language == 'language_value'

def test_analyze_syntax_empty_call():
    if False:
        return 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.analyze_syntax), '__call__') as call:
        client.analyze_syntax()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.AnalyzeSyntaxRequest()

@pytest.mark.asyncio
async def test_analyze_syntax_async(transport: str='grpc_asyncio', request_type=language_service.AnalyzeSyntaxRequest):
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.analyze_syntax), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(language_service.AnalyzeSyntaxResponse(language='language_value'))
        response = await client.analyze_syntax(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.AnalyzeSyntaxRequest()
    assert isinstance(response, language_service.AnalyzeSyntaxResponse)
    assert response.language == 'language_value'

@pytest.mark.asyncio
async def test_analyze_syntax_async_from_dict():
    await test_analyze_syntax_async(request_type=dict)

def test_analyze_syntax_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.analyze_syntax), '__call__') as call:
        call.return_value = language_service.AnalyzeSyntaxResponse()
        client.analyze_syntax(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].document
        mock_val = language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT)
        assert arg == mock_val
        arg = args[0].encoding_type
        mock_val = language_service.EncodingType.UTF8
        assert arg == mock_val

def test_analyze_syntax_flattened_error():
    if False:
        while True:
            i = 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.analyze_syntax(language_service.AnalyzeSyntaxRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)

@pytest.mark.asyncio
async def test_analyze_syntax_flattened_async():
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.analyze_syntax), '__call__') as call:
        call.return_value = language_service.AnalyzeSyntaxResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(language_service.AnalyzeSyntaxResponse())
        response = await client.analyze_syntax(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].document
        mock_val = language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT)
        assert arg == mock_val
        arg = args[0].encoding_type
        mock_val = language_service.EncodingType.UTF8
        assert arg == mock_val

@pytest.mark.asyncio
async def test_analyze_syntax_flattened_error_async():
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.analyze_syntax(language_service.AnalyzeSyntaxRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)

@pytest.mark.parametrize('request_type', [language_service.ClassifyTextRequest, dict])
def test_classify_text(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.classify_text), '__call__') as call:
        call.return_value = language_service.ClassifyTextResponse()
        response = client.classify_text(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.ClassifyTextRequest()
    assert isinstance(response, language_service.ClassifyTextResponse)

def test_classify_text_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.classify_text), '__call__') as call:
        client.classify_text()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.ClassifyTextRequest()

@pytest.mark.asyncio
async def test_classify_text_async(transport: str='grpc_asyncio', request_type=language_service.ClassifyTextRequest):
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.classify_text), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(language_service.ClassifyTextResponse())
        response = await client.classify_text(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.ClassifyTextRequest()
    assert isinstance(response, language_service.ClassifyTextResponse)

@pytest.mark.asyncio
async def test_classify_text_async_from_dict():
    await test_classify_text_async(request_type=dict)

def test_classify_text_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.classify_text), '__call__') as call:
        call.return_value = language_service.ClassifyTextResponse()
        client.classify_text(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].document
        mock_val = language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT)
        assert arg == mock_val

def test_classify_text_flattened_error():
    if False:
        print('Hello World!')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.classify_text(language_service.ClassifyTextRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT))

@pytest.mark.asyncio
async def test_classify_text_flattened_async():
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.classify_text), '__call__') as call:
        call.return_value = language_service.ClassifyTextResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(language_service.ClassifyTextResponse())
        response = await client.classify_text(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].document
        mock_val = language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT)
        assert arg == mock_val

@pytest.mark.asyncio
async def test_classify_text_flattened_error_async():
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.classify_text(language_service.ClassifyTextRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT))

@pytest.mark.parametrize('request_type', [language_service.ModerateTextRequest, dict])
def test_moderate_text(request_type, transport: str='grpc'):
    if False:
        return 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.moderate_text), '__call__') as call:
        call.return_value = language_service.ModerateTextResponse()
        response = client.moderate_text(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.ModerateTextRequest()
    assert isinstance(response, language_service.ModerateTextResponse)

def test_moderate_text_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.moderate_text), '__call__') as call:
        client.moderate_text()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.ModerateTextRequest()

@pytest.mark.asyncio
async def test_moderate_text_async(transport: str='grpc_asyncio', request_type=language_service.ModerateTextRequest):
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.moderate_text), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(language_service.ModerateTextResponse())
        response = await client.moderate_text(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.ModerateTextRequest()
    assert isinstance(response, language_service.ModerateTextResponse)

@pytest.mark.asyncio
async def test_moderate_text_async_from_dict():
    await test_moderate_text_async(request_type=dict)

def test_moderate_text_flattened():
    if False:
        i = 10
        return i + 15
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.moderate_text), '__call__') as call:
        call.return_value = language_service.ModerateTextResponse()
        client.moderate_text(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].document
        mock_val = language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT)
        assert arg == mock_val

def test_moderate_text_flattened_error():
    if False:
        return 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.moderate_text(language_service.ModerateTextRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT))

@pytest.mark.asyncio
async def test_moderate_text_flattened_async():
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.moderate_text), '__call__') as call:
        call.return_value = language_service.ModerateTextResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(language_service.ModerateTextResponse())
        response = await client.moderate_text(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].document
        mock_val = language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT)
        assert arg == mock_val

@pytest.mark.asyncio
async def test_moderate_text_flattened_error_async():
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.moderate_text(language_service.ModerateTextRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT))

@pytest.mark.parametrize('request_type', [language_service.AnnotateTextRequest, dict])
def test_annotate_text(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.annotate_text), '__call__') as call:
        call.return_value = language_service.AnnotateTextResponse(language='language_value')
        response = client.annotate_text(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.AnnotateTextRequest()
    assert isinstance(response, language_service.AnnotateTextResponse)
    assert response.language == 'language_value'

def test_annotate_text_empty_call():
    if False:
        while True:
            i = 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.annotate_text), '__call__') as call:
        client.annotate_text()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.AnnotateTextRequest()

@pytest.mark.asyncio
async def test_annotate_text_async(transport: str='grpc_asyncio', request_type=language_service.AnnotateTextRequest):
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.annotate_text), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(language_service.AnnotateTextResponse(language='language_value'))
        response = await client.annotate_text(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == language_service.AnnotateTextRequest()
    assert isinstance(response, language_service.AnnotateTextResponse)
    assert response.language == 'language_value'

@pytest.mark.asyncio
async def test_annotate_text_async_from_dict():
    await test_annotate_text_async(request_type=dict)

def test_annotate_text_flattened():
    if False:
        return 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.annotate_text), '__call__') as call:
        call.return_value = language_service.AnnotateTextResponse()
        client.annotate_text(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), features=language_service.AnnotateTextRequest.Features(extract_syntax=True), encoding_type=language_service.EncodingType.UTF8)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].document
        mock_val = language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT)
        assert arg == mock_val
        arg = args[0].features
        mock_val = language_service.AnnotateTextRequest.Features(extract_syntax=True)
        assert arg == mock_val
        arg = args[0].encoding_type
        mock_val = language_service.EncodingType.UTF8
        assert arg == mock_val

def test_annotate_text_flattened_error():
    if False:
        print('Hello World!')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.annotate_text(language_service.AnnotateTextRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), features=language_service.AnnotateTextRequest.Features(extract_syntax=True), encoding_type=language_service.EncodingType.UTF8)

@pytest.mark.asyncio
async def test_annotate_text_flattened_async():
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.annotate_text), '__call__') as call:
        call.return_value = language_service.AnnotateTextResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(language_service.AnnotateTextResponse())
        response = await client.annotate_text(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), features=language_service.AnnotateTextRequest.Features(extract_syntax=True), encoding_type=language_service.EncodingType.UTF8)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].document
        mock_val = language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT)
        assert arg == mock_val
        arg = args[0].features
        mock_val = language_service.AnnotateTextRequest.Features(extract_syntax=True)
        assert arg == mock_val
        arg = args[0].encoding_type
        mock_val = language_service.EncodingType.UTF8
        assert arg == mock_val

@pytest.mark.asyncio
async def test_annotate_text_flattened_error_async():
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.annotate_text(language_service.AnnotateTextRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), features=language_service.AnnotateTextRequest.Features(extract_syntax=True), encoding_type=language_service.EncodingType.UTF8)

@pytest.mark.parametrize('request_type', [language_service.AnalyzeSentimentRequest, dict])
def test_analyze_sentiment_rest(request_type):
    if False:
        return 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = language_service.AnalyzeSentimentResponse(language='language_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = language_service.AnalyzeSentimentResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.analyze_sentiment(request)
    assert isinstance(response, language_service.AnalyzeSentimentResponse)
    assert response.language == 'language_value'

def test_analyze_sentiment_rest_required_fields(request_type=language_service.AnalyzeSentimentRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.LanguageServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_sentiment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_sentiment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = language_service.AnalyzeSentimentResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = language_service.AnalyzeSentimentResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.analyze_sentiment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_analyze_sentiment_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.LanguageServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.analyze_sentiment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('document',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_analyze_sentiment_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.LanguageServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LanguageServiceRestInterceptor())
    client = LanguageServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LanguageServiceRestInterceptor, 'post_analyze_sentiment') as post, mock.patch.object(transports.LanguageServiceRestInterceptor, 'pre_analyze_sentiment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = language_service.AnalyzeSentimentRequest.pb(language_service.AnalyzeSentimentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = language_service.AnalyzeSentimentResponse.to_json(language_service.AnalyzeSentimentResponse())
        request = language_service.AnalyzeSentimentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = language_service.AnalyzeSentimentResponse()
        client.analyze_sentiment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_analyze_sentiment_rest_bad_request(transport: str='rest', request_type=language_service.AnalyzeSentimentRequest):
    if False:
        return 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.analyze_sentiment(request)

def test_analyze_sentiment_rest_flattened():
    if False:
        return 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = language_service.AnalyzeSentimentResponse()
        sample_request = {}
        mock_args = dict(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = language_service.AnalyzeSentimentResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.analyze_sentiment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta2/documents:analyzeSentiment' % client.transport._host, args[1])

def test_analyze_sentiment_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.analyze_sentiment(language_service.AnalyzeSentimentRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)

def test_analyze_sentiment_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [language_service.AnalyzeEntitiesRequest, dict])
def test_analyze_entities_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = language_service.AnalyzeEntitiesResponse(language='language_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = language_service.AnalyzeEntitiesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.analyze_entities(request)
    assert isinstance(response, language_service.AnalyzeEntitiesResponse)
    assert response.language == 'language_value'

def test_analyze_entities_rest_required_fields(request_type=language_service.AnalyzeEntitiesRequest):
    if False:
        return 10
    transport_class = transports.LanguageServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_entities._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_entities._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = language_service.AnalyzeEntitiesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = language_service.AnalyzeEntitiesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.analyze_entities(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_analyze_entities_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.LanguageServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.analyze_entities._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('document',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_analyze_entities_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.LanguageServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LanguageServiceRestInterceptor())
    client = LanguageServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LanguageServiceRestInterceptor, 'post_analyze_entities') as post, mock.patch.object(transports.LanguageServiceRestInterceptor, 'pre_analyze_entities') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = language_service.AnalyzeEntitiesRequest.pb(language_service.AnalyzeEntitiesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = language_service.AnalyzeEntitiesResponse.to_json(language_service.AnalyzeEntitiesResponse())
        request = language_service.AnalyzeEntitiesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = language_service.AnalyzeEntitiesResponse()
        client.analyze_entities(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_analyze_entities_rest_bad_request(transport: str='rest', request_type=language_service.AnalyzeEntitiesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.analyze_entities(request)

def test_analyze_entities_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = language_service.AnalyzeEntitiesResponse()
        sample_request = {}
        mock_args = dict(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = language_service.AnalyzeEntitiesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.analyze_entities(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta2/documents:analyzeEntities' % client.transport._host, args[1])

def test_analyze_entities_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.analyze_entities(language_service.AnalyzeEntitiesRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)

def test_analyze_entities_rest_error():
    if False:
        i = 10
        return i + 15
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [language_service.AnalyzeEntitySentimentRequest, dict])
def test_analyze_entity_sentiment_rest(request_type):
    if False:
        while True:
            i = 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = language_service.AnalyzeEntitySentimentResponse(language='language_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = language_service.AnalyzeEntitySentimentResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.analyze_entity_sentiment(request)
    assert isinstance(response, language_service.AnalyzeEntitySentimentResponse)
    assert response.language == 'language_value'

def test_analyze_entity_sentiment_rest_required_fields(request_type=language_service.AnalyzeEntitySentimentRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.LanguageServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_entity_sentiment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_entity_sentiment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = language_service.AnalyzeEntitySentimentResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = language_service.AnalyzeEntitySentimentResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.analyze_entity_sentiment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_analyze_entity_sentiment_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.LanguageServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.analyze_entity_sentiment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('document',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_analyze_entity_sentiment_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.LanguageServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LanguageServiceRestInterceptor())
    client = LanguageServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LanguageServiceRestInterceptor, 'post_analyze_entity_sentiment') as post, mock.patch.object(transports.LanguageServiceRestInterceptor, 'pre_analyze_entity_sentiment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = language_service.AnalyzeEntitySentimentRequest.pb(language_service.AnalyzeEntitySentimentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = language_service.AnalyzeEntitySentimentResponse.to_json(language_service.AnalyzeEntitySentimentResponse())
        request = language_service.AnalyzeEntitySentimentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = language_service.AnalyzeEntitySentimentResponse()
        client.analyze_entity_sentiment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_analyze_entity_sentiment_rest_bad_request(transport: str='rest', request_type=language_service.AnalyzeEntitySentimentRequest):
    if False:
        i = 10
        return i + 15
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.analyze_entity_sentiment(request)

def test_analyze_entity_sentiment_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = language_service.AnalyzeEntitySentimentResponse()
        sample_request = {}
        mock_args = dict(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = language_service.AnalyzeEntitySentimentResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.analyze_entity_sentiment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta2/documents:analyzeEntitySentiment' % client.transport._host, args[1])

def test_analyze_entity_sentiment_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.analyze_entity_sentiment(language_service.AnalyzeEntitySentimentRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)

def test_analyze_entity_sentiment_rest_error():
    if False:
        i = 10
        return i + 15
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [language_service.AnalyzeSyntaxRequest, dict])
def test_analyze_syntax_rest(request_type):
    if False:
        while True:
            i = 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = language_service.AnalyzeSyntaxResponse(language='language_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = language_service.AnalyzeSyntaxResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.analyze_syntax(request)
    assert isinstance(response, language_service.AnalyzeSyntaxResponse)
    assert response.language == 'language_value'

def test_analyze_syntax_rest_required_fields(request_type=language_service.AnalyzeSyntaxRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.LanguageServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_syntax._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).analyze_syntax._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = language_service.AnalyzeSyntaxResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = language_service.AnalyzeSyntaxResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.analyze_syntax(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_analyze_syntax_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.LanguageServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.analyze_syntax._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('document',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_analyze_syntax_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.LanguageServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LanguageServiceRestInterceptor())
    client = LanguageServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LanguageServiceRestInterceptor, 'post_analyze_syntax') as post, mock.patch.object(transports.LanguageServiceRestInterceptor, 'pre_analyze_syntax') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = language_service.AnalyzeSyntaxRequest.pb(language_service.AnalyzeSyntaxRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = language_service.AnalyzeSyntaxResponse.to_json(language_service.AnalyzeSyntaxResponse())
        request = language_service.AnalyzeSyntaxRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = language_service.AnalyzeSyntaxResponse()
        client.analyze_syntax(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_analyze_syntax_rest_bad_request(transport: str='rest', request_type=language_service.AnalyzeSyntaxRequest):
    if False:
        print('Hello World!')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.analyze_syntax(request)

def test_analyze_syntax_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = language_service.AnalyzeSyntaxResponse()
        sample_request = {}
        mock_args = dict(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = language_service.AnalyzeSyntaxResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.analyze_syntax(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta2/documents:analyzeSyntax' % client.transport._host, args[1])

def test_analyze_syntax_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.analyze_syntax(language_service.AnalyzeSyntaxRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), encoding_type=language_service.EncodingType.UTF8)

def test_analyze_syntax_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [language_service.ClassifyTextRequest, dict])
def test_classify_text_rest(request_type):
    if False:
        while True:
            i = 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = language_service.ClassifyTextResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = language_service.ClassifyTextResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.classify_text(request)
    assert isinstance(response, language_service.ClassifyTextResponse)

def test_classify_text_rest_required_fields(request_type=language_service.ClassifyTextRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.LanguageServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).classify_text._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).classify_text._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = language_service.ClassifyTextResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = language_service.ClassifyTextResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.classify_text(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_classify_text_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.LanguageServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.classify_text._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('document',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_classify_text_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.LanguageServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LanguageServiceRestInterceptor())
    client = LanguageServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LanguageServiceRestInterceptor, 'post_classify_text') as post, mock.patch.object(transports.LanguageServiceRestInterceptor, 'pre_classify_text') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = language_service.ClassifyTextRequest.pb(language_service.ClassifyTextRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = language_service.ClassifyTextResponse.to_json(language_service.ClassifyTextResponse())
        request = language_service.ClassifyTextRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = language_service.ClassifyTextResponse()
        client.classify_text(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_classify_text_rest_bad_request(transport: str='rest', request_type=language_service.ClassifyTextRequest):
    if False:
        while True:
            i = 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.classify_text(request)

def test_classify_text_rest_flattened():
    if False:
        while True:
            i = 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = language_service.ClassifyTextResponse()
        sample_request = {}
        mock_args = dict(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = language_service.ClassifyTextResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.classify_text(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta2/documents:classifyText' % client.transport._host, args[1])

def test_classify_text_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.classify_text(language_service.ClassifyTextRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT))

def test_classify_text_rest_error():
    if False:
        print('Hello World!')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [language_service.ModerateTextRequest, dict])
def test_moderate_text_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = language_service.ModerateTextResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = language_service.ModerateTextResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.moderate_text(request)
    assert isinstance(response, language_service.ModerateTextResponse)

def test_moderate_text_rest_required_fields(request_type=language_service.ModerateTextRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.LanguageServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).moderate_text._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).moderate_text._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = language_service.ModerateTextResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = language_service.ModerateTextResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.moderate_text(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_moderate_text_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.LanguageServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.moderate_text._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('document',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_moderate_text_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.LanguageServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LanguageServiceRestInterceptor())
    client = LanguageServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LanguageServiceRestInterceptor, 'post_moderate_text') as post, mock.patch.object(transports.LanguageServiceRestInterceptor, 'pre_moderate_text') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = language_service.ModerateTextRequest.pb(language_service.ModerateTextRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = language_service.ModerateTextResponse.to_json(language_service.ModerateTextResponse())
        request = language_service.ModerateTextRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = language_service.ModerateTextResponse()
        client.moderate_text(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_moderate_text_rest_bad_request(transport: str='rest', request_type=language_service.ModerateTextRequest):
    if False:
        for i in range(10):
            print('nop')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.moderate_text(request)

def test_moderate_text_rest_flattened():
    if False:
        print('Hello World!')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = language_service.ModerateTextResponse()
        sample_request = {}
        mock_args = dict(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = language_service.ModerateTextResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.moderate_text(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta2/documents:moderateText' % client.transport._host, args[1])

def test_moderate_text_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.moderate_text(language_service.ModerateTextRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT))

def test_moderate_text_rest_error():
    if False:
        while True:
            i = 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [language_service.AnnotateTextRequest, dict])
def test_annotate_text_rest(request_type):
    if False:
        while True:
            i = 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = language_service.AnnotateTextResponse(language='language_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = language_service.AnnotateTextResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.annotate_text(request)
    assert isinstance(response, language_service.AnnotateTextResponse)
    assert response.language == 'language_value'

def test_annotate_text_rest_required_fields(request_type=language_service.AnnotateTextRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.LanguageServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).annotate_text._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).annotate_text._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = language_service.AnnotateTextResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = language_service.AnnotateTextResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.annotate_text(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_annotate_text_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.LanguageServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.annotate_text._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('document', 'features'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_annotate_text_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.LanguageServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LanguageServiceRestInterceptor())
    client = LanguageServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LanguageServiceRestInterceptor, 'post_annotate_text') as post, mock.patch.object(transports.LanguageServiceRestInterceptor, 'pre_annotate_text') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = language_service.AnnotateTextRequest.pb(language_service.AnnotateTextRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = language_service.AnnotateTextResponse.to_json(language_service.AnnotateTextResponse())
        request = language_service.AnnotateTextRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = language_service.AnnotateTextResponse()
        client.annotate_text(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_annotate_text_rest_bad_request(transport: str='rest', request_type=language_service.AnnotateTextRequest):
    if False:
        print('Hello World!')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.annotate_text(request)

def test_annotate_text_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = language_service.AnnotateTextResponse()
        sample_request = {}
        mock_args = dict(document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), features=language_service.AnnotateTextRequest.Features(extract_syntax=True), encoding_type=language_service.EncodingType.UTF8)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = language_service.AnnotateTextResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.annotate_text(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta2/documents:annotateText' % client.transport._host, args[1])

def test_annotate_text_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.annotate_text(language_service.AnnotateTextRequest(), document=language_service.Document(type_=language_service.Document.Type.PLAIN_TEXT), features=language_service.AnnotateTextRequest.Features(extract_syntax=True), encoding_type=language_service.EncodingType.UTF8)

def test_annotate_text_rest_error():
    if False:
        i = 10
        return i + 15
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.LanguageServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.LanguageServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = LanguageServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.LanguageServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = LanguageServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = LanguageServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.LanguageServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = LanguageServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.LanguageServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = LanguageServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.LanguageServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.LanguageServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.LanguageServiceGrpcTransport, transports.LanguageServiceGrpcAsyncIOTransport, transports.LanguageServiceRestTransport])
def test_transport_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        return 10
    transport = LanguageServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.LanguageServiceGrpcTransport)

def test_language_service_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.LanguageServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_language_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.language_v1beta2.services.language_service.transports.LanguageServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.LanguageServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('analyze_sentiment', 'analyze_entities', 'analyze_entity_sentiment', 'analyze_syntax', 'classify_text', 'moderate_text', 'annotate_text')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_language_service_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.language_v1beta2.services.language_service.transports.LanguageServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.LanguageServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-language', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id='octopus')

def test_language_service_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.language_v1beta2.services.language_service.transports.LanguageServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.LanguageServiceTransport()
        adc.assert_called_once()

def test_language_service_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        LanguageServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-language', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.LanguageServiceGrpcTransport, transports.LanguageServiceGrpcAsyncIOTransport])
def test_language_service_transport_auth_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-language', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.LanguageServiceGrpcTransport, transports.LanguageServiceGrpcAsyncIOTransport, transports.LanguageServiceRestTransport])
def test_language_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.LanguageServiceGrpcTransport, grpc_helpers), (transports.LanguageServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_language_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('language.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-language', 'https://www.googleapis.com/auth/cloud-platform'), scopes=['1', '2'], default_host='language.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.LanguageServiceGrpcTransport, transports.LanguageServiceGrpcAsyncIOTransport])
def test_language_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_language_service_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.LanguageServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_language_service_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='language.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('language.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://language.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_language_service_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='language.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('language.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://language.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_language_service_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = LanguageServiceClient(credentials=creds1, transport=transport_name)
    client2 = LanguageServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.analyze_sentiment._session
    session2 = client2.transport.analyze_sentiment._session
    assert session1 != session2
    session1 = client1.transport.analyze_entities._session
    session2 = client2.transport.analyze_entities._session
    assert session1 != session2
    session1 = client1.transport.analyze_entity_sentiment._session
    session2 = client2.transport.analyze_entity_sentiment._session
    assert session1 != session2
    session1 = client1.transport.analyze_syntax._session
    session2 = client2.transport.analyze_syntax._session
    assert session1 != session2
    session1 = client1.transport.classify_text._session
    session2 = client2.transport.classify_text._session
    assert session1 != session2
    session1 = client1.transport.moderate_text._session
    session2 = client2.transport.moderate_text._session
    assert session1 != session2
    session1 = client1.transport.annotate_text._session
    session2 = client2.transport.annotate_text._session
    assert session1 != session2

def test_language_service_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.LanguageServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_language_service_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.LanguageServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.LanguageServiceGrpcTransport, transports.LanguageServiceGrpcAsyncIOTransport])
def test_language_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.LanguageServiceGrpcTransport, transports.LanguageServiceGrpcAsyncIOTransport])
def test_language_service_transport_channel_mtls_with_adc(transport_class):
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

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = LanguageServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'clam'}
    path = LanguageServiceClient.common_billing_account_path(**expected)
    actual = LanguageServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = LanguageServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'octopus'}
    path = LanguageServiceClient.common_folder_path(**expected)
    actual = LanguageServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = LanguageServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        i = 10
        return i + 15
    expected = {'organization': 'nudibranch'}
    path = LanguageServiceClient.common_organization_path(**expected)
    actual = LanguageServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = LanguageServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'mussel'}
    path = LanguageServiceClient.common_project_path(**expected)
    actual = LanguageServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = LanguageServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = LanguageServiceClient.common_location_path(**expected)
    actual = LanguageServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.LanguageServiceTransport, '_prep_wrapped_messages') as prep:
        client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.LanguageServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = LanguageServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = LanguageServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
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
        client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest', 'grpc']
    for transport in transports:
        client = LanguageServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(LanguageServiceClient, transports.LanguageServiceGrpcTransport), (LanguageServiceAsyncClient, transports.LanguageServiceGrpcAsyncIOTransport)])
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