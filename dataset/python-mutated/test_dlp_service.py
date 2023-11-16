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
from google.cloud.location import locations_pb2
from google.oauth2 import service_account
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import empty_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.rpc import status_pb2
from google.type import date_pb2
from google.type import dayofweek_pb2
from google.type import timeofday_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.dlp_v2.services.dlp_service import DlpServiceAsyncClient, DlpServiceClient, pagers, transports
from google.cloud.dlp_v2.types import dlp, storage

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
        print('Hello World!')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert DlpServiceClient._get_default_mtls_endpoint(None) is None
    assert DlpServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert DlpServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert DlpServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert DlpServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert DlpServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(DlpServiceClient, 'grpc'), (DlpServiceAsyncClient, 'grpc_asyncio'), (DlpServiceClient, 'rest')])
def test_dlp_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('dlp.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dlp.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.DlpServiceGrpcTransport, 'grpc'), (transports.DlpServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.DlpServiceRestTransport, 'rest')])
def test_dlp_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(DlpServiceClient, 'grpc'), (DlpServiceAsyncClient, 'grpc_asyncio'), (DlpServiceClient, 'rest')])
def test_dlp_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('dlp.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dlp.googleapis.com')

def test_dlp_service_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = DlpServiceClient.get_transport_class()
    available_transports = [transports.DlpServiceGrpcTransport, transports.DlpServiceRestTransport]
    assert transport in available_transports
    transport = DlpServiceClient.get_transport_class('grpc')
    assert transport == transports.DlpServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(DlpServiceClient, transports.DlpServiceGrpcTransport, 'grpc'), (DlpServiceAsyncClient, transports.DlpServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (DlpServiceClient, transports.DlpServiceRestTransport, 'rest')])
@mock.patch.object(DlpServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DlpServiceClient))
@mock.patch.object(DlpServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DlpServiceAsyncClient))
def test_dlp_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(DlpServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(DlpServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(DlpServiceClient, transports.DlpServiceGrpcTransport, 'grpc', 'true'), (DlpServiceAsyncClient, transports.DlpServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (DlpServiceClient, transports.DlpServiceGrpcTransport, 'grpc', 'false'), (DlpServiceAsyncClient, transports.DlpServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (DlpServiceClient, transports.DlpServiceRestTransport, 'rest', 'true'), (DlpServiceClient, transports.DlpServiceRestTransport, 'rest', 'false')])
@mock.patch.object(DlpServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DlpServiceClient))
@mock.patch.object(DlpServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DlpServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_dlp_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [DlpServiceClient, DlpServiceAsyncClient])
@mock.patch.object(DlpServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DlpServiceClient))
@mock.patch.object(DlpServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(DlpServiceAsyncClient))
def test_dlp_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(DlpServiceClient, transports.DlpServiceGrpcTransport, 'grpc'), (DlpServiceAsyncClient, transports.DlpServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (DlpServiceClient, transports.DlpServiceRestTransport, 'rest')])
def test_dlp_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(DlpServiceClient, transports.DlpServiceGrpcTransport, 'grpc', grpc_helpers), (DlpServiceAsyncClient, transports.DlpServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (DlpServiceClient, transports.DlpServiceRestTransport, 'rest', None)])
def test_dlp_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_dlp_service_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.dlp_v2.services.dlp_service.transports.DlpServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = DlpServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(DlpServiceClient, transports.DlpServiceGrpcTransport, 'grpc', grpc_helpers), (DlpServiceAsyncClient, transports.DlpServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_dlp_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('dlp.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='dlp.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [dlp.InspectContentRequest, dict])
def test_inspect_content(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.inspect_content), '__call__') as call:
        call.return_value = dlp.InspectContentResponse()
        response = client.inspect_content(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.InspectContentRequest()
    assert isinstance(response, dlp.InspectContentResponse)

def test_inspect_content_empty_call():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.inspect_content), '__call__') as call:
        client.inspect_content()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.InspectContentRequest()

@pytest.mark.asyncio
async def test_inspect_content_async(transport: str='grpc_asyncio', request_type=dlp.InspectContentRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.inspect_content), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.InspectContentResponse())
        response = await client.inspect_content(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.InspectContentRequest()
    assert isinstance(response, dlp.InspectContentResponse)

@pytest.mark.asyncio
async def test_inspect_content_async_from_dict():
    await test_inspect_content_async(request_type=dict)

def test_inspect_content_field_headers():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.InspectContentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.inspect_content), '__call__') as call:
        call.return_value = dlp.InspectContentResponse()
        client.inspect_content(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_inspect_content_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.InspectContentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.inspect_content), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.InspectContentResponse())
        await client.inspect_content(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [dlp.RedactImageRequest, dict])
def test_redact_image(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.redact_image), '__call__') as call:
        call.return_value = dlp.RedactImageResponse(redacted_image=b'redacted_image_blob', extracted_text='extracted_text_value')
        response = client.redact_image(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.RedactImageRequest()
    assert isinstance(response, dlp.RedactImageResponse)
    assert response.redacted_image == b'redacted_image_blob'
    assert response.extracted_text == 'extracted_text_value'

def test_redact_image_empty_call():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.redact_image), '__call__') as call:
        client.redact_image()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.RedactImageRequest()

@pytest.mark.asyncio
async def test_redact_image_async(transport: str='grpc_asyncio', request_type=dlp.RedactImageRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.redact_image), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.RedactImageResponse(redacted_image=b'redacted_image_blob', extracted_text='extracted_text_value'))
        response = await client.redact_image(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.RedactImageRequest()
    assert isinstance(response, dlp.RedactImageResponse)
    assert response.redacted_image == b'redacted_image_blob'
    assert response.extracted_text == 'extracted_text_value'

@pytest.mark.asyncio
async def test_redact_image_async_from_dict():
    await test_redact_image_async(request_type=dict)

def test_redact_image_field_headers():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.RedactImageRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.redact_image), '__call__') as call:
        call.return_value = dlp.RedactImageResponse()
        client.redact_image(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_redact_image_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.RedactImageRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.redact_image), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.RedactImageResponse())
        await client.redact_image(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [dlp.DeidentifyContentRequest, dict])
def test_deidentify_content(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.deidentify_content), '__call__') as call:
        call.return_value = dlp.DeidentifyContentResponse()
        response = client.deidentify_content(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeidentifyContentRequest()
    assert isinstance(response, dlp.DeidentifyContentResponse)

def test_deidentify_content_empty_call():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.deidentify_content), '__call__') as call:
        client.deidentify_content()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeidentifyContentRequest()

@pytest.mark.asyncio
async def test_deidentify_content_async(transport: str='grpc_asyncio', request_type=dlp.DeidentifyContentRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.deidentify_content), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DeidentifyContentResponse())
        response = await client.deidentify_content(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeidentifyContentRequest()
    assert isinstance(response, dlp.DeidentifyContentResponse)

@pytest.mark.asyncio
async def test_deidentify_content_async_from_dict():
    await test_deidentify_content_async(request_type=dict)

def test_deidentify_content_field_headers():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.DeidentifyContentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.deidentify_content), '__call__') as call:
        call.return_value = dlp.DeidentifyContentResponse()
        client.deidentify_content(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_deidentify_content_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.DeidentifyContentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.deidentify_content), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DeidentifyContentResponse())
        await client.deidentify_content(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [dlp.ReidentifyContentRequest, dict])
def test_reidentify_content(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reidentify_content), '__call__') as call:
        call.return_value = dlp.ReidentifyContentResponse()
        response = client.reidentify_content(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ReidentifyContentRequest()
    assert isinstance(response, dlp.ReidentifyContentResponse)

def test_reidentify_content_empty_call():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.reidentify_content), '__call__') as call:
        client.reidentify_content()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ReidentifyContentRequest()

@pytest.mark.asyncio
async def test_reidentify_content_async(transport: str='grpc_asyncio', request_type=dlp.ReidentifyContentRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reidentify_content), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ReidentifyContentResponse())
        response = await client.reidentify_content(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ReidentifyContentRequest()
    assert isinstance(response, dlp.ReidentifyContentResponse)

@pytest.mark.asyncio
async def test_reidentify_content_async_from_dict():
    await test_reidentify_content_async(request_type=dict)

def test_reidentify_content_field_headers():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.ReidentifyContentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.reidentify_content), '__call__') as call:
        call.return_value = dlp.ReidentifyContentResponse()
        client.reidentify_content(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_reidentify_content_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.ReidentifyContentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.reidentify_content), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ReidentifyContentResponse())
        await client.reidentify_content(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [dlp.ListInfoTypesRequest, dict])
def test_list_info_types(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_info_types), '__call__') as call:
        call.return_value = dlp.ListInfoTypesResponse()
        response = client.list_info_types(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListInfoTypesRequest()
    assert isinstance(response, dlp.ListInfoTypesResponse)

def test_list_info_types_empty_call():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_info_types), '__call__') as call:
        client.list_info_types()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListInfoTypesRequest()

@pytest.mark.asyncio
async def test_list_info_types_async(transport: str='grpc_asyncio', request_type=dlp.ListInfoTypesRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_info_types), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListInfoTypesResponse())
        response = await client.list_info_types(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListInfoTypesRequest()
    assert isinstance(response, dlp.ListInfoTypesResponse)

@pytest.mark.asyncio
async def test_list_info_types_async_from_dict():
    await test_list_info_types_async(request_type=dict)

def test_list_info_types_flattened():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_info_types), '__call__') as call:
        call.return_value = dlp.ListInfoTypesResponse()
        client.list_info_types(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_info_types_flattened_error():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_info_types(dlp.ListInfoTypesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_info_types_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_info_types), '__call__') as call:
        call.return_value = dlp.ListInfoTypesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListInfoTypesResponse())
        response = await client.list_info_types(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_info_types_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_info_types(dlp.ListInfoTypesRequest(), parent='parent_value')

@pytest.mark.parametrize('request_type', [dlp.CreateInspectTemplateRequest, dict])
def test_create_inspect_template(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_inspect_template), '__call__') as call:
        call.return_value = dlp.InspectTemplate(name='name_value', display_name='display_name_value', description='description_value')
        response = client.create_inspect_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CreateInspectTemplateRequest()
    assert isinstance(response, dlp.InspectTemplate)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'

def test_create_inspect_template_empty_call():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_inspect_template), '__call__') as call:
        client.create_inspect_template()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CreateInspectTemplateRequest()

@pytest.mark.asyncio
async def test_create_inspect_template_async(transport: str='grpc_asyncio', request_type=dlp.CreateInspectTemplateRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_inspect_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.InspectTemplate(name='name_value', display_name='display_name_value', description='description_value'))
        response = await client.create_inspect_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CreateInspectTemplateRequest()
    assert isinstance(response, dlp.InspectTemplate)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'

@pytest.mark.asyncio
async def test_create_inspect_template_async_from_dict():
    await test_create_inspect_template_async(request_type=dict)

def test_create_inspect_template_field_headers():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.CreateInspectTemplateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_inspect_template), '__call__') as call:
        call.return_value = dlp.InspectTemplate()
        client.create_inspect_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_inspect_template_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.CreateInspectTemplateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_inspect_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.InspectTemplate())
        await client.create_inspect_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_inspect_template_flattened():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_inspect_template), '__call__') as call:
        call.return_value = dlp.InspectTemplate()
        client.create_inspect_template(parent='parent_value', inspect_template=dlp.InspectTemplate(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].inspect_template
        mock_val = dlp.InspectTemplate(name='name_value')
        assert arg == mock_val

def test_create_inspect_template_flattened_error():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_inspect_template(dlp.CreateInspectTemplateRequest(), parent='parent_value', inspect_template=dlp.InspectTemplate(name='name_value'))

@pytest.mark.asyncio
async def test_create_inspect_template_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_inspect_template), '__call__') as call:
        call.return_value = dlp.InspectTemplate()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.InspectTemplate())
        response = await client.create_inspect_template(parent='parent_value', inspect_template=dlp.InspectTemplate(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].inspect_template
        mock_val = dlp.InspectTemplate(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_inspect_template_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_inspect_template(dlp.CreateInspectTemplateRequest(), parent='parent_value', inspect_template=dlp.InspectTemplate(name='name_value'))

@pytest.mark.parametrize('request_type', [dlp.UpdateInspectTemplateRequest, dict])
def test_update_inspect_template(request_type, transport: str='grpc'):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_inspect_template), '__call__') as call:
        call.return_value = dlp.InspectTemplate(name='name_value', display_name='display_name_value', description='description_value')
        response = client.update_inspect_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.UpdateInspectTemplateRequest()
    assert isinstance(response, dlp.InspectTemplate)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'

def test_update_inspect_template_empty_call():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_inspect_template), '__call__') as call:
        client.update_inspect_template()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.UpdateInspectTemplateRequest()

@pytest.mark.asyncio
async def test_update_inspect_template_async(transport: str='grpc_asyncio', request_type=dlp.UpdateInspectTemplateRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_inspect_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.InspectTemplate(name='name_value', display_name='display_name_value', description='description_value'))
        response = await client.update_inspect_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.UpdateInspectTemplateRequest()
    assert isinstance(response, dlp.InspectTemplate)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'

@pytest.mark.asyncio
async def test_update_inspect_template_async_from_dict():
    await test_update_inspect_template_async(request_type=dict)

def test_update_inspect_template_field_headers():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.UpdateInspectTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_inspect_template), '__call__') as call:
        call.return_value = dlp.InspectTemplate()
        client.update_inspect_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_inspect_template_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.UpdateInspectTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_inspect_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.InspectTemplate())
        await client.update_inspect_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_update_inspect_template_flattened():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_inspect_template), '__call__') as call:
        call.return_value = dlp.InspectTemplate()
        client.update_inspect_template(name='name_value', inspect_template=dlp.InspectTemplate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].inspect_template
        mock_val = dlp.InspectTemplate(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_inspect_template_flattened_error():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_inspect_template(dlp.UpdateInspectTemplateRequest(), name='name_value', inspect_template=dlp.InspectTemplate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_inspect_template_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_inspect_template), '__call__') as call:
        call.return_value = dlp.InspectTemplate()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.InspectTemplate())
        response = await client.update_inspect_template(name='name_value', inspect_template=dlp.InspectTemplate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].inspect_template
        mock_val = dlp.InspectTemplate(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_inspect_template_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_inspect_template(dlp.UpdateInspectTemplateRequest(), name='name_value', inspect_template=dlp.InspectTemplate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [dlp.GetInspectTemplateRequest, dict])
def test_get_inspect_template(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_inspect_template), '__call__') as call:
        call.return_value = dlp.InspectTemplate(name='name_value', display_name='display_name_value', description='description_value')
        response = client.get_inspect_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.GetInspectTemplateRequest()
    assert isinstance(response, dlp.InspectTemplate)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'

def test_get_inspect_template_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_inspect_template), '__call__') as call:
        client.get_inspect_template()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.GetInspectTemplateRequest()

@pytest.mark.asyncio
async def test_get_inspect_template_async(transport: str='grpc_asyncio', request_type=dlp.GetInspectTemplateRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_inspect_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.InspectTemplate(name='name_value', display_name='display_name_value', description='description_value'))
        response = await client.get_inspect_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.GetInspectTemplateRequest()
    assert isinstance(response, dlp.InspectTemplate)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'

@pytest.mark.asyncio
async def test_get_inspect_template_async_from_dict():
    await test_get_inspect_template_async(request_type=dict)

def test_get_inspect_template_field_headers():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.GetInspectTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_inspect_template), '__call__') as call:
        call.return_value = dlp.InspectTemplate()
        client.get_inspect_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_inspect_template_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.GetInspectTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_inspect_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.InspectTemplate())
        await client.get_inspect_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_inspect_template_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_inspect_template), '__call__') as call:
        call.return_value = dlp.InspectTemplate()
        client.get_inspect_template(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_inspect_template_flattened_error():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_inspect_template(dlp.GetInspectTemplateRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_inspect_template_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_inspect_template), '__call__') as call:
        call.return_value = dlp.InspectTemplate()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.InspectTemplate())
        response = await client.get_inspect_template(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_inspect_template_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_inspect_template(dlp.GetInspectTemplateRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [dlp.ListInspectTemplatesRequest, dict])
def test_list_inspect_templates(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_inspect_templates), '__call__') as call:
        call.return_value = dlp.ListInspectTemplatesResponse(next_page_token='next_page_token_value')
        response = client.list_inspect_templates(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListInspectTemplatesRequest()
    assert isinstance(response, pagers.ListInspectTemplatesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_inspect_templates_empty_call():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_inspect_templates), '__call__') as call:
        client.list_inspect_templates()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListInspectTemplatesRequest()

@pytest.mark.asyncio
async def test_list_inspect_templates_async(transport: str='grpc_asyncio', request_type=dlp.ListInspectTemplatesRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_inspect_templates), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListInspectTemplatesResponse(next_page_token='next_page_token_value'))
        response = await client.list_inspect_templates(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListInspectTemplatesRequest()
    assert isinstance(response, pagers.ListInspectTemplatesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_inspect_templates_async_from_dict():
    await test_list_inspect_templates_async(request_type=dict)

def test_list_inspect_templates_field_headers():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.ListInspectTemplatesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_inspect_templates), '__call__') as call:
        call.return_value = dlp.ListInspectTemplatesResponse()
        client.list_inspect_templates(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_inspect_templates_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.ListInspectTemplatesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_inspect_templates), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListInspectTemplatesResponse())
        await client.list_inspect_templates(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_inspect_templates_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_inspect_templates), '__call__') as call:
        call.return_value = dlp.ListInspectTemplatesResponse()
        client.list_inspect_templates(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_inspect_templates_flattened_error():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_inspect_templates(dlp.ListInspectTemplatesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_inspect_templates_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_inspect_templates), '__call__') as call:
        call.return_value = dlp.ListInspectTemplatesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListInspectTemplatesResponse())
        response = await client.list_inspect_templates(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_inspect_templates_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_inspect_templates(dlp.ListInspectTemplatesRequest(), parent='parent_value')

def test_list_inspect_templates_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_inspect_templates), '__call__') as call:
        call.side_effect = (dlp.ListInspectTemplatesResponse(inspect_templates=[dlp.InspectTemplate(), dlp.InspectTemplate(), dlp.InspectTemplate()], next_page_token='abc'), dlp.ListInspectTemplatesResponse(inspect_templates=[], next_page_token='def'), dlp.ListInspectTemplatesResponse(inspect_templates=[dlp.InspectTemplate()], next_page_token='ghi'), dlp.ListInspectTemplatesResponse(inspect_templates=[dlp.InspectTemplate(), dlp.InspectTemplate()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_inspect_templates(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, dlp.InspectTemplate) for i in results))

def test_list_inspect_templates_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_inspect_templates), '__call__') as call:
        call.side_effect = (dlp.ListInspectTemplatesResponse(inspect_templates=[dlp.InspectTemplate(), dlp.InspectTemplate(), dlp.InspectTemplate()], next_page_token='abc'), dlp.ListInspectTemplatesResponse(inspect_templates=[], next_page_token='def'), dlp.ListInspectTemplatesResponse(inspect_templates=[dlp.InspectTemplate()], next_page_token='ghi'), dlp.ListInspectTemplatesResponse(inspect_templates=[dlp.InspectTemplate(), dlp.InspectTemplate()]), RuntimeError)
        pages = list(client.list_inspect_templates(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_inspect_templates_async_pager():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_inspect_templates), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (dlp.ListInspectTemplatesResponse(inspect_templates=[dlp.InspectTemplate(), dlp.InspectTemplate(), dlp.InspectTemplate()], next_page_token='abc'), dlp.ListInspectTemplatesResponse(inspect_templates=[], next_page_token='def'), dlp.ListInspectTemplatesResponse(inspect_templates=[dlp.InspectTemplate()], next_page_token='ghi'), dlp.ListInspectTemplatesResponse(inspect_templates=[dlp.InspectTemplate(), dlp.InspectTemplate()]), RuntimeError)
        async_pager = await client.list_inspect_templates(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, dlp.InspectTemplate) for i in responses))

@pytest.mark.asyncio
async def test_list_inspect_templates_async_pages():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_inspect_templates), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (dlp.ListInspectTemplatesResponse(inspect_templates=[dlp.InspectTemplate(), dlp.InspectTemplate(), dlp.InspectTemplate()], next_page_token='abc'), dlp.ListInspectTemplatesResponse(inspect_templates=[], next_page_token='def'), dlp.ListInspectTemplatesResponse(inspect_templates=[dlp.InspectTemplate()], next_page_token='ghi'), dlp.ListInspectTemplatesResponse(inspect_templates=[dlp.InspectTemplate(), dlp.InspectTemplate()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_inspect_templates(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [dlp.DeleteInspectTemplateRequest, dict])
def test_delete_inspect_template(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_inspect_template), '__call__') as call:
        call.return_value = None
        response = client.delete_inspect_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeleteInspectTemplateRequest()
    assert response is None

def test_delete_inspect_template_empty_call():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_inspect_template), '__call__') as call:
        client.delete_inspect_template()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeleteInspectTemplateRequest()

@pytest.mark.asyncio
async def test_delete_inspect_template_async(transport: str='grpc_asyncio', request_type=dlp.DeleteInspectTemplateRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_inspect_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_inspect_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeleteInspectTemplateRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_inspect_template_async_from_dict():
    await test_delete_inspect_template_async(request_type=dict)

def test_delete_inspect_template_field_headers():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.DeleteInspectTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_inspect_template), '__call__') as call:
        call.return_value = None
        client.delete_inspect_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_inspect_template_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.DeleteInspectTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_inspect_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_inspect_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_inspect_template_flattened():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_inspect_template), '__call__') as call:
        call.return_value = None
        client.delete_inspect_template(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_inspect_template_flattened_error():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_inspect_template(dlp.DeleteInspectTemplateRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_inspect_template_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_inspect_template), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_inspect_template(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_inspect_template_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_inspect_template(dlp.DeleteInspectTemplateRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [dlp.CreateDeidentifyTemplateRequest, dict])
def test_create_deidentify_template(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_deidentify_template), '__call__') as call:
        call.return_value = dlp.DeidentifyTemplate(name='name_value', display_name='display_name_value', description='description_value')
        response = client.create_deidentify_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CreateDeidentifyTemplateRequest()
    assert isinstance(response, dlp.DeidentifyTemplate)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'

def test_create_deidentify_template_empty_call():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_deidentify_template), '__call__') as call:
        client.create_deidentify_template()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CreateDeidentifyTemplateRequest()

@pytest.mark.asyncio
async def test_create_deidentify_template_async(transport: str='grpc_asyncio', request_type=dlp.CreateDeidentifyTemplateRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_deidentify_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DeidentifyTemplate(name='name_value', display_name='display_name_value', description='description_value'))
        response = await client.create_deidentify_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CreateDeidentifyTemplateRequest()
    assert isinstance(response, dlp.DeidentifyTemplate)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'

@pytest.mark.asyncio
async def test_create_deidentify_template_async_from_dict():
    await test_create_deidentify_template_async(request_type=dict)

def test_create_deidentify_template_field_headers():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.CreateDeidentifyTemplateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_deidentify_template), '__call__') as call:
        call.return_value = dlp.DeidentifyTemplate()
        client.create_deidentify_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_deidentify_template_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.CreateDeidentifyTemplateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_deidentify_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DeidentifyTemplate())
        await client.create_deidentify_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_deidentify_template_flattened():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_deidentify_template), '__call__') as call:
        call.return_value = dlp.DeidentifyTemplate()
        client.create_deidentify_template(parent='parent_value', deidentify_template=dlp.DeidentifyTemplate(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].deidentify_template
        mock_val = dlp.DeidentifyTemplate(name='name_value')
        assert arg == mock_val

def test_create_deidentify_template_flattened_error():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_deidentify_template(dlp.CreateDeidentifyTemplateRequest(), parent='parent_value', deidentify_template=dlp.DeidentifyTemplate(name='name_value'))

@pytest.mark.asyncio
async def test_create_deidentify_template_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_deidentify_template), '__call__') as call:
        call.return_value = dlp.DeidentifyTemplate()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DeidentifyTemplate())
        response = await client.create_deidentify_template(parent='parent_value', deidentify_template=dlp.DeidentifyTemplate(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].deidentify_template
        mock_val = dlp.DeidentifyTemplate(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_deidentify_template_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_deidentify_template(dlp.CreateDeidentifyTemplateRequest(), parent='parent_value', deidentify_template=dlp.DeidentifyTemplate(name='name_value'))

@pytest.mark.parametrize('request_type', [dlp.UpdateDeidentifyTemplateRequest, dict])
def test_update_deidentify_template(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_deidentify_template), '__call__') as call:
        call.return_value = dlp.DeidentifyTemplate(name='name_value', display_name='display_name_value', description='description_value')
        response = client.update_deidentify_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.UpdateDeidentifyTemplateRequest()
    assert isinstance(response, dlp.DeidentifyTemplate)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'

def test_update_deidentify_template_empty_call():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_deidentify_template), '__call__') as call:
        client.update_deidentify_template()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.UpdateDeidentifyTemplateRequest()

@pytest.mark.asyncio
async def test_update_deidentify_template_async(transport: str='grpc_asyncio', request_type=dlp.UpdateDeidentifyTemplateRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_deidentify_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DeidentifyTemplate(name='name_value', display_name='display_name_value', description='description_value'))
        response = await client.update_deidentify_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.UpdateDeidentifyTemplateRequest()
    assert isinstance(response, dlp.DeidentifyTemplate)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'

@pytest.mark.asyncio
async def test_update_deidentify_template_async_from_dict():
    await test_update_deidentify_template_async(request_type=dict)

def test_update_deidentify_template_field_headers():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.UpdateDeidentifyTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_deidentify_template), '__call__') as call:
        call.return_value = dlp.DeidentifyTemplate()
        client.update_deidentify_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_deidentify_template_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.UpdateDeidentifyTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_deidentify_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DeidentifyTemplate())
        await client.update_deidentify_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_update_deidentify_template_flattened():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_deidentify_template), '__call__') as call:
        call.return_value = dlp.DeidentifyTemplate()
        client.update_deidentify_template(name='name_value', deidentify_template=dlp.DeidentifyTemplate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].deidentify_template
        mock_val = dlp.DeidentifyTemplate(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_deidentify_template_flattened_error():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_deidentify_template(dlp.UpdateDeidentifyTemplateRequest(), name='name_value', deidentify_template=dlp.DeidentifyTemplate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_deidentify_template_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_deidentify_template), '__call__') as call:
        call.return_value = dlp.DeidentifyTemplate()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DeidentifyTemplate())
        response = await client.update_deidentify_template(name='name_value', deidentify_template=dlp.DeidentifyTemplate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].deidentify_template
        mock_val = dlp.DeidentifyTemplate(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_deidentify_template_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_deidentify_template(dlp.UpdateDeidentifyTemplateRequest(), name='name_value', deidentify_template=dlp.DeidentifyTemplate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [dlp.GetDeidentifyTemplateRequest, dict])
def test_get_deidentify_template(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_deidentify_template), '__call__') as call:
        call.return_value = dlp.DeidentifyTemplate(name='name_value', display_name='display_name_value', description='description_value')
        response = client.get_deidentify_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.GetDeidentifyTemplateRequest()
    assert isinstance(response, dlp.DeidentifyTemplate)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'

def test_get_deidentify_template_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_deidentify_template), '__call__') as call:
        client.get_deidentify_template()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.GetDeidentifyTemplateRequest()

@pytest.mark.asyncio
async def test_get_deidentify_template_async(transport: str='grpc_asyncio', request_type=dlp.GetDeidentifyTemplateRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_deidentify_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DeidentifyTemplate(name='name_value', display_name='display_name_value', description='description_value'))
        response = await client.get_deidentify_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.GetDeidentifyTemplateRequest()
    assert isinstance(response, dlp.DeidentifyTemplate)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'

@pytest.mark.asyncio
async def test_get_deidentify_template_async_from_dict():
    await test_get_deidentify_template_async(request_type=dict)

def test_get_deidentify_template_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.GetDeidentifyTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_deidentify_template), '__call__') as call:
        call.return_value = dlp.DeidentifyTemplate()
        client.get_deidentify_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_deidentify_template_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.GetDeidentifyTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_deidentify_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DeidentifyTemplate())
        await client.get_deidentify_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_deidentify_template_flattened():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_deidentify_template), '__call__') as call:
        call.return_value = dlp.DeidentifyTemplate()
        client.get_deidentify_template(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_deidentify_template_flattened_error():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_deidentify_template(dlp.GetDeidentifyTemplateRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_deidentify_template_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_deidentify_template), '__call__') as call:
        call.return_value = dlp.DeidentifyTemplate()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DeidentifyTemplate())
        response = await client.get_deidentify_template(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_deidentify_template_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_deidentify_template(dlp.GetDeidentifyTemplateRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [dlp.ListDeidentifyTemplatesRequest, dict])
def test_list_deidentify_templates(request_type, transport: str='grpc'):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_deidentify_templates), '__call__') as call:
        call.return_value = dlp.ListDeidentifyTemplatesResponse(next_page_token='next_page_token_value')
        response = client.list_deidentify_templates(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListDeidentifyTemplatesRequest()
    assert isinstance(response, pagers.ListDeidentifyTemplatesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_deidentify_templates_empty_call():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_deidentify_templates), '__call__') as call:
        client.list_deidentify_templates()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListDeidentifyTemplatesRequest()

@pytest.mark.asyncio
async def test_list_deidentify_templates_async(transport: str='grpc_asyncio', request_type=dlp.ListDeidentifyTemplatesRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_deidentify_templates), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListDeidentifyTemplatesResponse(next_page_token='next_page_token_value'))
        response = await client.list_deidentify_templates(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListDeidentifyTemplatesRequest()
    assert isinstance(response, pagers.ListDeidentifyTemplatesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_deidentify_templates_async_from_dict():
    await test_list_deidentify_templates_async(request_type=dict)

def test_list_deidentify_templates_field_headers():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.ListDeidentifyTemplatesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_deidentify_templates), '__call__') as call:
        call.return_value = dlp.ListDeidentifyTemplatesResponse()
        client.list_deidentify_templates(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_deidentify_templates_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.ListDeidentifyTemplatesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_deidentify_templates), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListDeidentifyTemplatesResponse())
        await client.list_deidentify_templates(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_deidentify_templates_flattened():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_deidentify_templates), '__call__') as call:
        call.return_value = dlp.ListDeidentifyTemplatesResponse()
        client.list_deidentify_templates(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_deidentify_templates_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_deidentify_templates(dlp.ListDeidentifyTemplatesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_deidentify_templates_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_deidentify_templates), '__call__') as call:
        call.return_value = dlp.ListDeidentifyTemplatesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListDeidentifyTemplatesResponse())
        response = await client.list_deidentify_templates(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_deidentify_templates_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_deidentify_templates(dlp.ListDeidentifyTemplatesRequest(), parent='parent_value')

def test_list_deidentify_templates_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_deidentify_templates), '__call__') as call:
        call.side_effect = (dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[dlp.DeidentifyTemplate(), dlp.DeidentifyTemplate(), dlp.DeidentifyTemplate()], next_page_token='abc'), dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[], next_page_token='def'), dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[dlp.DeidentifyTemplate()], next_page_token='ghi'), dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[dlp.DeidentifyTemplate(), dlp.DeidentifyTemplate()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_deidentify_templates(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, dlp.DeidentifyTemplate) for i in results))

def test_list_deidentify_templates_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_deidentify_templates), '__call__') as call:
        call.side_effect = (dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[dlp.DeidentifyTemplate(), dlp.DeidentifyTemplate(), dlp.DeidentifyTemplate()], next_page_token='abc'), dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[], next_page_token='def'), dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[dlp.DeidentifyTemplate()], next_page_token='ghi'), dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[dlp.DeidentifyTemplate(), dlp.DeidentifyTemplate()]), RuntimeError)
        pages = list(client.list_deidentify_templates(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_deidentify_templates_async_pager():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_deidentify_templates), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[dlp.DeidentifyTemplate(), dlp.DeidentifyTemplate(), dlp.DeidentifyTemplate()], next_page_token='abc'), dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[], next_page_token='def'), dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[dlp.DeidentifyTemplate()], next_page_token='ghi'), dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[dlp.DeidentifyTemplate(), dlp.DeidentifyTemplate()]), RuntimeError)
        async_pager = await client.list_deidentify_templates(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, dlp.DeidentifyTemplate) for i in responses))

@pytest.mark.asyncio
async def test_list_deidentify_templates_async_pages():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_deidentify_templates), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[dlp.DeidentifyTemplate(), dlp.DeidentifyTemplate(), dlp.DeidentifyTemplate()], next_page_token='abc'), dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[], next_page_token='def'), dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[dlp.DeidentifyTemplate()], next_page_token='ghi'), dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[dlp.DeidentifyTemplate(), dlp.DeidentifyTemplate()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_deidentify_templates(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [dlp.DeleteDeidentifyTemplateRequest, dict])
def test_delete_deidentify_template(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_deidentify_template), '__call__') as call:
        call.return_value = None
        response = client.delete_deidentify_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeleteDeidentifyTemplateRequest()
    assert response is None

def test_delete_deidentify_template_empty_call():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_deidentify_template), '__call__') as call:
        client.delete_deidentify_template()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeleteDeidentifyTemplateRequest()

@pytest.mark.asyncio
async def test_delete_deidentify_template_async(transport: str='grpc_asyncio', request_type=dlp.DeleteDeidentifyTemplateRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_deidentify_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_deidentify_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeleteDeidentifyTemplateRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_deidentify_template_async_from_dict():
    await test_delete_deidentify_template_async(request_type=dict)

def test_delete_deidentify_template_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.DeleteDeidentifyTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_deidentify_template), '__call__') as call:
        call.return_value = None
        client.delete_deidentify_template(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_deidentify_template_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.DeleteDeidentifyTemplateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_deidentify_template), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_deidentify_template(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_deidentify_template_flattened():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_deidentify_template), '__call__') as call:
        call.return_value = None
        client.delete_deidentify_template(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_deidentify_template_flattened_error():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_deidentify_template(dlp.DeleteDeidentifyTemplateRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_deidentify_template_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_deidentify_template), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_deidentify_template(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_deidentify_template_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_deidentify_template(dlp.DeleteDeidentifyTemplateRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [dlp.CreateJobTriggerRequest, dict])
def test_create_job_trigger(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_job_trigger), '__call__') as call:
        call.return_value = dlp.JobTrigger(name='name_value', display_name='display_name_value', description='description_value', status=dlp.JobTrigger.Status.HEALTHY)
        response = client.create_job_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CreateJobTriggerRequest()
    assert isinstance(response, dlp.JobTrigger)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.status == dlp.JobTrigger.Status.HEALTHY

def test_create_job_trigger_empty_call():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_job_trigger), '__call__') as call:
        client.create_job_trigger()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CreateJobTriggerRequest()

@pytest.mark.asyncio
async def test_create_job_trigger_async(transport: str='grpc_asyncio', request_type=dlp.CreateJobTriggerRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_job_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.JobTrigger(name='name_value', display_name='display_name_value', description='description_value', status=dlp.JobTrigger.Status.HEALTHY))
        response = await client.create_job_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CreateJobTriggerRequest()
    assert isinstance(response, dlp.JobTrigger)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.status == dlp.JobTrigger.Status.HEALTHY

@pytest.mark.asyncio
async def test_create_job_trigger_async_from_dict():
    await test_create_job_trigger_async(request_type=dict)

def test_create_job_trigger_field_headers():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.CreateJobTriggerRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_job_trigger), '__call__') as call:
        call.return_value = dlp.JobTrigger()
        client.create_job_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_job_trigger_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.CreateJobTriggerRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_job_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.JobTrigger())
        await client.create_job_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_job_trigger_flattened():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_job_trigger), '__call__') as call:
        call.return_value = dlp.JobTrigger()
        client.create_job_trigger(parent='parent_value', job_trigger=dlp.JobTrigger(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].job_trigger
        mock_val = dlp.JobTrigger(name='name_value')
        assert arg == mock_val

def test_create_job_trigger_flattened_error():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_job_trigger(dlp.CreateJobTriggerRequest(), parent='parent_value', job_trigger=dlp.JobTrigger(name='name_value'))

@pytest.mark.asyncio
async def test_create_job_trigger_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_job_trigger), '__call__') as call:
        call.return_value = dlp.JobTrigger()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.JobTrigger())
        response = await client.create_job_trigger(parent='parent_value', job_trigger=dlp.JobTrigger(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].job_trigger
        mock_val = dlp.JobTrigger(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_job_trigger_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_job_trigger(dlp.CreateJobTriggerRequest(), parent='parent_value', job_trigger=dlp.JobTrigger(name='name_value'))

@pytest.mark.parametrize('request_type', [dlp.UpdateJobTriggerRequest, dict])
def test_update_job_trigger(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_job_trigger), '__call__') as call:
        call.return_value = dlp.JobTrigger(name='name_value', display_name='display_name_value', description='description_value', status=dlp.JobTrigger.Status.HEALTHY)
        response = client.update_job_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.UpdateJobTriggerRequest()
    assert isinstance(response, dlp.JobTrigger)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.status == dlp.JobTrigger.Status.HEALTHY

def test_update_job_trigger_empty_call():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_job_trigger), '__call__') as call:
        client.update_job_trigger()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.UpdateJobTriggerRequest()

@pytest.mark.asyncio
async def test_update_job_trigger_async(transport: str='grpc_asyncio', request_type=dlp.UpdateJobTriggerRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_job_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.JobTrigger(name='name_value', display_name='display_name_value', description='description_value', status=dlp.JobTrigger.Status.HEALTHY))
        response = await client.update_job_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.UpdateJobTriggerRequest()
    assert isinstance(response, dlp.JobTrigger)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.status == dlp.JobTrigger.Status.HEALTHY

@pytest.mark.asyncio
async def test_update_job_trigger_async_from_dict():
    await test_update_job_trigger_async(request_type=dict)

def test_update_job_trigger_field_headers():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.UpdateJobTriggerRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_job_trigger), '__call__') as call:
        call.return_value = dlp.JobTrigger()
        client.update_job_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_job_trigger_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.UpdateJobTriggerRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_job_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.JobTrigger())
        await client.update_job_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_update_job_trigger_flattened():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_job_trigger), '__call__') as call:
        call.return_value = dlp.JobTrigger()
        client.update_job_trigger(name='name_value', job_trigger=dlp.JobTrigger(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].job_trigger
        mock_val = dlp.JobTrigger(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_job_trigger_flattened_error():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_job_trigger(dlp.UpdateJobTriggerRequest(), name='name_value', job_trigger=dlp.JobTrigger(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_job_trigger_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_job_trigger), '__call__') as call:
        call.return_value = dlp.JobTrigger()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.JobTrigger())
        response = await client.update_job_trigger(name='name_value', job_trigger=dlp.JobTrigger(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].job_trigger
        mock_val = dlp.JobTrigger(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_job_trigger_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_job_trigger(dlp.UpdateJobTriggerRequest(), name='name_value', job_trigger=dlp.JobTrigger(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [dlp.HybridInspectJobTriggerRequest, dict])
def test_hybrid_inspect_job_trigger(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.hybrid_inspect_job_trigger), '__call__') as call:
        call.return_value = dlp.HybridInspectResponse()
        response = client.hybrid_inspect_job_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.HybridInspectJobTriggerRequest()
    assert isinstance(response, dlp.HybridInspectResponse)

def test_hybrid_inspect_job_trigger_empty_call():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.hybrid_inspect_job_trigger), '__call__') as call:
        client.hybrid_inspect_job_trigger()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.HybridInspectJobTriggerRequest()

@pytest.mark.asyncio
async def test_hybrid_inspect_job_trigger_async(transport: str='grpc_asyncio', request_type=dlp.HybridInspectJobTriggerRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.hybrid_inspect_job_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.HybridInspectResponse())
        response = await client.hybrid_inspect_job_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.HybridInspectJobTriggerRequest()
    assert isinstance(response, dlp.HybridInspectResponse)

@pytest.mark.asyncio
async def test_hybrid_inspect_job_trigger_async_from_dict():
    await test_hybrid_inspect_job_trigger_async(request_type=dict)

def test_hybrid_inspect_job_trigger_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.HybridInspectJobTriggerRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.hybrid_inspect_job_trigger), '__call__') as call:
        call.return_value = dlp.HybridInspectResponse()
        client.hybrid_inspect_job_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_hybrid_inspect_job_trigger_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.HybridInspectJobTriggerRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.hybrid_inspect_job_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.HybridInspectResponse())
        await client.hybrid_inspect_job_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_hybrid_inspect_job_trigger_flattened():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.hybrid_inspect_job_trigger), '__call__') as call:
        call.return_value = dlp.HybridInspectResponse()
        client.hybrid_inspect_job_trigger(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_hybrid_inspect_job_trigger_flattened_error():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.hybrid_inspect_job_trigger(dlp.HybridInspectJobTriggerRequest(), name='name_value')

@pytest.mark.asyncio
async def test_hybrid_inspect_job_trigger_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.hybrid_inspect_job_trigger), '__call__') as call:
        call.return_value = dlp.HybridInspectResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.HybridInspectResponse())
        response = await client.hybrid_inspect_job_trigger(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_hybrid_inspect_job_trigger_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.hybrid_inspect_job_trigger(dlp.HybridInspectJobTriggerRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [dlp.GetJobTriggerRequest, dict])
def test_get_job_trigger(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_job_trigger), '__call__') as call:
        call.return_value = dlp.JobTrigger(name='name_value', display_name='display_name_value', description='description_value', status=dlp.JobTrigger.Status.HEALTHY)
        response = client.get_job_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.GetJobTriggerRequest()
    assert isinstance(response, dlp.JobTrigger)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.status == dlp.JobTrigger.Status.HEALTHY

def test_get_job_trigger_empty_call():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_job_trigger), '__call__') as call:
        client.get_job_trigger()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.GetJobTriggerRequest()

@pytest.mark.asyncio
async def test_get_job_trigger_async(transport: str='grpc_asyncio', request_type=dlp.GetJobTriggerRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_job_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.JobTrigger(name='name_value', display_name='display_name_value', description='description_value', status=dlp.JobTrigger.Status.HEALTHY))
        response = await client.get_job_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.GetJobTriggerRequest()
    assert isinstance(response, dlp.JobTrigger)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.status == dlp.JobTrigger.Status.HEALTHY

@pytest.mark.asyncio
async def test_get_job_trigger_async_from_dict():
    await test_get_job_trigger_async(request_type=dict)

def test_get_job_trigger_field_headers():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.GetJobTriggerRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_job_trigger), '__call__') as call:
        call.return_value = dlp.JobTrigger()
        client.get_job_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_job_trigger_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.GetJobTriggerRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_job_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.JobTrigger())
        await client.get_job_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_job_trigger_flattened():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_job_trigger), '__call__') as call:
        call.return_value = dlp.JobTrigger()
        client.get_job_trigger(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_job_trigger_flattened_error():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_job_trigger(dlp.GetJobTriggerRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_job_trigger_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_job_trigger), '__call__') as call:
        call.return_value = dlp.JobTrigger()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.JobTrigger())
        response = await client.get_job_trigger(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_job_trigger_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_job_trigger(dlp.GetJobTriggerRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [dlp.ListJobTriggersRequest, dict])
def test_list_job_triggers(request_type, transport: str='grpc'):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_job_triggers), '__call__') as call:
        call.return_value = dlp.ListJobTriggersResponse(next_page_token='next_page_token_value')
        response = client.list_job_triggers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListJobTriggersRequest()
    assert isinstance(response, pagers.ListJobTriggersPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_job_triggers_empty_call():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_job_triggers), '__call__') as call:
        client.list_job_triggers()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListJobTriggersRequest()

@pytest.mark.asyncio
async def test_list_job_triggers_async(transport: str='grpc_asyncio', request_type=dlp.ListJobTriggersRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_job_triggers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListJobTriggersResponse(next_page_token='next_page_token_value'))
        response = await client.list_job_triggers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListJobTriggersRequest()
    assert isinstance(response, pagers.ListJobTriggersAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_job_triggers_async_from_dict():
    await test_list_job_triggers_async(request_type=dict)

def test_list_job_triggers_field_headers():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.ListJobTriggersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_job_triggers), '__call__') as call:
        call.return_value = dlp.ListJobTriggersResponse()
        client.list_job_triggers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_job_triggers_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.ListJobTriggersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_job_triggers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListJobTriggersResponse())
        await client.list_job_triggers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_job_triggers_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_job_triggers), '__call__') as call:
        call.return_value = dlp.ListJobTriggersResponse()
        client.list_job_triggers(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_job_triggers_flattened_error():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_job_triggers(dlp.ListJobTriggersRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_job_triggers_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_job_triggers), '__call__') as call:
        call.return_value = dlp.ListJobTriggersResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListJobTriggersResponse())
        response = await client.list_job_triggers(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_job_triggers_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_job_triggers(dlp.ListJobTriggersRequest(), parent='parent_value')

def test_list_job_triggers_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_job_triggers), '__call__') as call:
        call.side_effect = (dlp.ListJobTriggersResponse(job_triggers=[dlp.JobTrigger(), dlp.JobTrigger(), dlp.JobTrigger()], next_page_token='abc'), dlp.ListJobTriggersResponse(job_triggers=[], next_page_token='def'), dlp.ListJobTriggersResponse(job_triggers=[dlp.JobTrigger()], next_page_token='ghi'), dlp.ListJobTriggersResponse(job_triggers=[dlp.JobTrigger(), dlp.JobTrigger()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_job_triggers(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, dlp.JobTrigger) for i in results))

def test_list_job_triggers_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_job_triggers), '__call__') as call:
        call.side_effect = (dlp.ListJobTriggersResponse(job_triggers=[dlp.JobTrigger(), dlp.JobTrigger(), dlp.JobTrigger()], next_page_token='abc'), dlp.ListJobTriggersResponse(job_triggers=[], next_page_token='def'), dlp.ListJobTriggersResponse(job_triggers=[dlp.JobTrigger()], next_page_token='ghi'), dlp.ListJobTriggersResponse(job_triggers=[dlp.JobTrigger(), dlp.JobTrigger()]), RuntimeError)
        pages = list(client.list_job_triggers(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_job_triggers_async_pager():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_job_triggers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (dlp.ListJobTriggersResponse(job_triggers=[dlp.JobTrigger(), dlp.JobTrigger(), dlp.JobTrigger()], next_page_token='abc'), dlp.ListJobTriggersResponse(job_triggers=[], next_page_token='def'), dlp.ListJobTriggersResponse(job_triggers=[dlp.JobTrigger()], next_page_token='ghi'), dlp.ListJobTriggersResponse(job_triggers=[dlp.JobTrigger(), dlp.JobTrigger()]), RuntimeError)
        async_pager = await client.list_job_triggers(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, dlp.JobTrigger) for i in responses))

@pytest.mark.asyncio
async def test_list_job_triggers_async_pages():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_job_triggers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (dlp.ListJobTriggersResponse(job_triggers=[dlp.JobTrigger(), dlp.JobTrigger(), dlp.JobTrigger()], next_page_token='abc'), dlp.ListJobTriggersResponse(job_triggers=[], next_page_token='def'), dlp.ListJobTriggersResponse(job_triggers=[dlp.JobTrigger()], next_page_token='ghi'), dlp.ListJobTriggersResponse(job_triggers=[dlp.JobTrigger(), dlp.JobTrigger()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_job_triggers(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [dlp.DeleteJobTriggerRequest, dict])
def test_delete_job_trigger(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_job_trigger), '__call__') as call:
        call.return_value = None
        response = client.delete_job_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeleteJobTriggerRequest()
    assert response is None

def test_delete_job_trigger_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_job_trigger), '__call__') as call:
        client.delete_job_trigger()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeleteJobTriggerRequest()

@pytest.mark.asyncio
async def test_delete_job_trigger_async(transport: str='grpc_asyncio', request_type=dlp.DeleteJobTriggerRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_job_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_job_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeleteJobTriggerRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_job_trigger_async_from_dict():
    await test_delete_job_trigger_async(request_type=dict)

def test_delete_job_trigger_field_headers():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.DeleteJobTriggerRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_job_trigger), '__call__') as call:
        call.return_value = None
        client.delete_job_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_job_trigger_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.DeleteJobTriggerRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_job_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_job_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_job_trigger_flattened():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_job_trigger), '__call__') as call:
        call.return_value = None
        client.delete_job_trigger(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_job_trigger_flattened_error():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_job_trigger(dlp.DeleteJobTriggerRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_job_trigger_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_job_trigger), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_job_trigger(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_job_trigger_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_job_trigger(dlp.DeleteJobTriggerRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [dlp.ActivateJobTriggerRequest, dict])
def test_activate_job_trigger(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.activate_job_trigger), '__call__') as call:
        call.return_value = dlp.DlpJob(name='name_value', type_=dlp.DlpJobType.INSPECT_JOB, state=dlp.DlpJob.JobState.PENDING, job_trigger_name='job_trigger_name_value')
        response = client.activate_job_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ActivateJobTriggerRequest()
    assert isinstance(response, dlp.DlpJob)
    assert response.name == 'name_value'
    assert response.type_ == dlp.DlpJobType.INSPECT_JOB
    assert response.state == dlp.DlpJob.JobState.PENDING
    assert response.job_trigger_name == 'job_trigger_name_value'

def test_activate_job_trigger_empty_call():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.activate_job_trigger), '__call__') as call:
        client.activate_job_trigger()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ActivateJobTriggerRequest()

@pytest.mark.asyncio
async def test_activate_job_trigger_async(transport: str='grpc_asyncio', request_type=dlp.ActivateJobTriggerRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.activate_job_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DlpJob(name='name_value', type_=dlp.DlpJobType.INSPECT_JOB, state=dlp.DlpJob.JobState.PENDING, job_trigger_name='job_trigger_name_value'))
        response = await client.activate_job_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ActivateJobTriggerRequest()
    assert isinstance(response, dlp.DlpJob)
    assert response.name == 'name_value'
    assert response.type_ == dlp.DlpJobType.INSPECT_JOB
    assert response.state == dlp.DlpJob.JobState.PENDING
    assert response.job_trigger_name == 'job_trigger_name_value'

@pytest.mark.asyncio
async def test_activate_job_trigger_async_from_dict():
    await test_activate_job_trigger_async(request_type=dict)

def test_activate_job_trigger_field_headers():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.ActivateJobTriggerRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.activate_job_trigger), '__call__') as call:
        call.return_value = dlp.DlpJob()
        client.activate_job_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_activate_job_trigger_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.ActivateJobTriggerRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.activate_job_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DlpJob())
        await client.activate_job_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [dlp.CreateDiscoveryConfigRequest, dict])
def test_create_discovery_config(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_discovery_config), '__call__') as call:
        call.return_value = dlp.DiscoveryConfig(name='name_value', display_name='display_name_value', inspect_templates=['inspect_templates_value'], status=dlp.DiscoveryConfig.Status.RUNNING)
        response = client.create_discovery_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CreateDiscoveryConfigRequest()
    assert isinstance(response, dlp.DiscoveryConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.inspect_templates == ['inspect_templates_value']
    assert response.status == dlp.DiscoveryConfig.Status.RUNNING

def test_create_discovery_config_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_discovery_config), '__call__') as call:
        client.create_discovery_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CreateDiscoveryConfigRequest()

@pytest.mark.asyncio
async def test_create_discovery_config_async(transport: str='grpc_asyncio', request_type=dlp.CreateDiscoveryConfigRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_discovery_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DiscoveryConfig(name='name_value', display_name='display_name_value', inspect_templates=['inspect_templates_value'], status=dlp.DiscoveryConfig.Status.RUNNING))
        response = await client.create_discovery_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CreateDiscoveryConfigRequest()
    assert isinstance(response, dlp.DiscoveryConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.inspect_templates == ['inspect_templates_value']
    assert response.status == dlp.DiscoveryConfig.Status.RUNNING

@pytest.mark.asyncio
async def test_create_discovery_config_async_from_dict():
    await test_create_discovery_config_async(request_type=dict)

def test_create_discovery_config_field_headers():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.CreateDiscoveryConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_discovery_config), '__call__') as call:
        call.return_value = dlp.DiscoveryConfig()
        client.create_discovery_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_discovery_config_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.CreateDiscoveryConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_discovery_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DiscoveryConfig())
        await client.create_discovery_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_discovery_config_flattened():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_discovery_config), '__call__') as call:
        call.return_value = dlp.DiscoveryConfig()
        client.create_discovery_config(parent='parent_value', discovery_config=dlp.DiscoveryConfig(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].discovery_config
        mock_val = dlp.DiscoveryConfig(name='name_value')
        assert arg == mock_val

def test_create_discovery_config_flattened_error():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_discovery_config(dlp.CreateDiscoveryConfigRequest(), parent='parent_value', discovery_config=dlp.DiscoveryConfig(name='name_value'))

@pytest.mark.asyncio
async def test_create_discovery_config_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_discovery_config), '__call__') as call:
        call.return_value = dlp.DiscoveryConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DiscoveryConfig())
        response = await client.create_discovery_config(parent='parent_value', discovery_config=dlp.DiscoveryConfig(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].discovery_config
        mock_val = dlp.DiscoveryConfig(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_discovery_config_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_discovery_config(dlp.CreateDiscoveryConfigRequest(), parent='parent_value', discovery_config=dlp.DiscoveryConfig(name='name_value'))

@pytest.mark.parametrize('request_type', [dlp.UpdateDiscoveryConfigRequest, dict])
def test_update_discovery_config(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_discovery_config), '__call__') as call:
        call.return_value = dlp.DiscoveryConfig(name='name_value', display_name='display_name_value', inspect_templates=['inspect_templates_value'], status=dlp.DiscoveryConfig.Status.RUNNING)
        response = client.update_discovery_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.UpdateDiscoveryConfigRequest()
    assert isinstance(response, dlp.DiscoveryConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.inspect_templates == ['inspect_templates_value']
    assert response.status == dlp.DiscoveryConfig.Status.RUNNING

def test_update_discovery_config_empty_call():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_discovery_config), '__call__') as call:
        client.update_discovery_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.UpdateDiscoveryConfigRequest()

@pytest.mark.asyncio
async def test_update_discovery_config_async(transport: str='grpc_asyncio', request_type=dlp.UpdateDiscoveryConfigRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_discovery_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DiscoveryConfig(name='name_value', display_name='display_name_value', inspect_templates=['inspect_templates_value'], status=dlp.DiscoveryConfig.Status.RUNNING))
        response = await client.update_discovery_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.UpdateDiscoveryConfigRequest()
    assert isinstance(response, dlp.DiscoveryConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.inspect_templates == ['inspect_templates_value']
    assert response.status == dlp.DiscoveryConfig.Status.RUNNING

@pytest.mark.asyncio
async def test_update_discovery_config_async_from_dict():
    await test_update_discovery_config_async(request_type=dict)

def test_update_discovery_config_field_headers():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.UpdateDiscoveryConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_discovery_config), '__call__') as call:
        call.return_value = dlp.DiscoveryConfig()
        client.update_discovery_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_discovery_config_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.UpdateDiscoveryConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_discovery_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DiscoveryConfig())
        await client.update_discovery_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_update_discovery_config_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_discovery_config), '__call__') as call:
        call.return_value = dlp.DiscoveryConfig()
        client.update_discovery_config(name='name_value', discovery_config=dlp.DiscoveryConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].discovery_config
        mock_val = dlp.DiscoveryConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_discovery_config_flattened_error():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_discovery_config(dlp.UpdateDiscoveryConfigRequest(), name='name_value', discovery_config=dlp.DiscoveryConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_discovery_config_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_discovery_config), '__call__') as call:
        call.return_value = dlp.DiscoveryConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DiscoveryConfig())
        response = await client.update_discovery_config(name='name_value', discovery_config=dlp.DiscoveryConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].discovery_config
        mock_val = dlp.DiscoveryConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_discovery_config_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_discovery_config(dlp.UpdateDiscoveryConfigRequest(), name='name_value', discovery_config=dlp.DiscoveryConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [dlp.GetDiscoveryConfigRequest, dict])
def test_get_discovery_config(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_discovery_config), '__call__') as call:
        call.return_value = dlp.DiscoveryConfig(name='name_value', display_name='display_name_value', inspect_templates=['inspect_templates_value'], status=dlp.DiscoveryConfig.Status.RUNNING)
        response = client.get_discovery_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.GetDiscoveryConfigRequest()
    assert isinstance(response, dlp.DiscoveryConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.inspect_templates == ['inspect_templates_value']
    assert response.status == dlp.DiscoveryConfig.Status.RUNNING

def test_get_discovery_config_empty_call():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_discovery_config), '__call__') as call:
        client.get_discovery_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.GetDiscoveryConfigRequest()

@pytest.mark.asyncio
async def test_get_discovery_config_async(transport: str='grpc_asyncio', request_type=dlp.GetDiscoveryConfigRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_discovery_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DiscoveryConfig(name='name_value', display_name='display_name_value', inspect_templates=['inspect_templates_value'], status=dlp.DiscoveryConfig.Status.RUNNING))
        response = await client.get_discovery_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.GetDiscoveryConfigRequest()
    assert isinstance(response, dlp.DiscoveryConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.inspect_templates == ['inspect_templates_value']
    assert response.status == dlp.DiscoveryConfig.Status.RUNNING

@pytest.mark.asyncio
async def test_get_discovery_config_async_from_dict():
    await test_get_discovery_config_async(request_type=dict)

def test_get_discovery_config_field_headers():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.GetDiscoveryConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_discovery_config), '__call__') as call:
        call.return_value = dlp.DiscoveryConfig()
        client.get_discovery_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_discovery_config_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.GetDiscoveryConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_discovery_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DiscoveryConfig())
        await client.get_discovery_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_discovery_config_flattened():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_discovery_config), '__call__') as call:
        call.return_value = dlp.DiscoveryConfig()
        client.get_discovery_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_discovery_config_flattened_error():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_discovery_config(dlp.GetDiscoveryConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_discovery_config_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_discovery_config), '__call__') as call:
        call.return_value = dlp.DiscoveryConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DiscoveryConfig())
        response = await client.get_discovery_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_discovery_config_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_discovery_config(dlp.GetDiscoveryConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [dlp.ListDiscoveryConfigsRequest, dict])
def test_list_discovery_configs(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_discovery_configs), '__call__') as call:
        call.return_value = dlp.ListDiscoveryConfigsResponse(next_page_token='next_page_token_value')
        response = client.list_discovery_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListDiscoveryConfigsRequest()
    assert isinstance(response, pagers.ListDiscoveryConfigsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_discovery_configs_empty_call():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_discovery_configs), '__call__') as call:
        client.list_discovery_configs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListDiscoveryConfigsRequest()

@pytest.mark.asyncio
async def test_list_discovery_configs_async(transport: str='grpc_asyncio', request_type=dlp.ListDiscoveryConfigsRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_discovery_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListDiscoveryConfigsResponse(next_page_token='next_page_token_value'))
        response = await client.list_discovery_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListDiscoveryConfigsRequest()
    assert isinstance(response, pagers.ListDiscoveryConfigsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_discovery_configs_async_from_dict():
    await test_list_discovery_configs_async(request_type=dict)

def test_list_discovery_configs_field_headers():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.ListDiscoveryConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_discovery_configs), '__call__') as call:
        call.return_value = dlp.ListDiscoveryConfigsResponse()
        client.list_discovery_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_discovery_configs_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.ListDiscoveryConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_discovery_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListDiscoveryConfigsResponse())
        await client.list_discovery_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_discovery_configs_flattened():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_discovery_configs), '__call__') as call:
        call.return_value = dlp.ListDiscoveryConfigsResponse()
        client.list_discovery_configs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_discovery_configs_flattened_error():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_discovery_configs(dlp.ListDiscoveryConfigsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_discovery_configs_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_discovery_configs), '__call__') as call:
        call.return_value = dlp.ListDiscoveryConfigsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListDiscoveryConfigsResponse())
        response = await client.list_discovery_configs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_discovery_configs_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_discovery_configs(dlp.ListDiscoveryConfigsRequest(), parent='parent_value')

def test_list_discovery_configs_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_discovery_configs), '__call__') as call:
        call.side_effect = (dlp.ListDiscoveryConfigsResponse(discovery_configs=[dlp.DiscoveryConfig(), dlp.DiscoveryConfig(), dlp.DiscoveryConfig()], next_page_token='abc'), dlp.ListDiscoveryConfigsResponse(discovery_configs=[], next_page_token='def'), dlp.ListDiscoveryConfigsResponse(discovery_configs=[dlp.DiscoveryConfig()], next_page_token='ghi'), dlp.ListDiscoveryConfigsResponse(discovery_configs=[dlp.DiscoveryConfig(), dlp.DiscoveryConfig()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_discovery_configs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, dlp.DiscoveryConfig) for i in results))

def test_list_discovery_configs_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_discovery_configs), '__call__') as call:
        call.side_effect = (dlp.ListDiscoveryConfigsResponse(discovery_configs=[dlp.DiscoveryConfig(), dlp.DiscoveryConfig(), dlp.DiscoveryConfig()], next_page_token='abc'), dlp.ListDiscoveryConfigsResponse(discovery_configs=[], next_page_token='def'), dlp.ListDiscoveryConfigsResponse(discovery_configs=[dlp.DiscoveryConfig()], next_page_token='ghi'), dlp.ListDiscoveryConfigsResponse(discovery_configs=[dlp.DiscoveryConfig(), dlp.DiscoveryConfig()]), RuntimeError)
        pages = list(client.list_discovery_configs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_discovery_configs_async_pager():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_discovery_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (dlp.ListDiscoveryConfigsResponse(discovery_configs=[dlp.DiscoveryConfig(), dlp.DiscoveryConfig(), dlp.DiscoveryConfig()], next_page_token='abc'), dlp.ListDiscoveryConfigsResponse(discovery_configs=[], next_page_token='def'), dlp.ListDiscoveryConfigsResponse(discovery_configs=[dlp.DiscoveryConfig()], next_page_token='ghi'), dlp.ListDiscoveryConfigsResponse(discovery_configs=[dlp.DiscoveryConfig(), dlp.DiscoveryConfig()]), RuntimeError)
        async_pager = await client.list_discovery_configs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, dlp.DiscoveryConfig) for i in responses))

@pytest.mark.asyncio
async def test_list_discovery_configs_async_pages():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_discovery_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (dlp.ListDiscoveryConfigsResponse(discovery_configs=[dlp.DiscoveryConfig(), dlp.DiscoveryConfig(), dlp.DiscoveryConfig()], next_page_token='abc'), dlp.ListDiscoveryConfigsResponse(discovery_configs=[], next_page_token='def'), dlp.ListDiscoveryConfigsResponse(discovery_configs=[dlp.DiscoveryConfig()], next_page_token='ghi'), dlp.ListDiscoveryConfigsResponse(discovery_configs=[dlp.DiscoveryConfig(), dlp.DiscoveryConfig()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_discovery_configs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [dlp.DeleteDiscoveryConfigRequest, dict])
def test_delete_discovery_config(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_discovery_config), '__call__') as call:
        call.return_value = None
        response = client.delete_discovery_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeleteDiscoveryConfigRequest()
    assert response is None

def test_delete_discovery_config_empty_call():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_discovery_config), '__call__') as call:
        client.delete_discovery_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeleteDiscoveryConfigRequest()

@pytest.mark.asyncio
async def test_delete_discovery_config_async(transport: str='grpc_asyncio', request_type=dlp.DeleteDiscoveryConfigRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_discovery_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_discovery_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeleteDiscoveryConfigRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_discovery_config_async_from_dict():
    await test_delete_discovery_config_async(request_type=dict)

def test_delete_discovery_config_field_headers():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.DeleteDiscoveryConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_discovery_config), '__call__') as call:
        call.return_value = None
        client.delete_discovery_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_discovery_config_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.DeleteDiscoveryConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_discovery_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_discovery_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_discovery_config_flattened():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_discovery_config), '__call__') as call:
        call.return_value = None
        client.delete_discovery_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_discovery_config_flattened_error():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_discovery_config(dlp.DeleteDiscoveryConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_discovery_config_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_discovery_config), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_discovery_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_discovery_config_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_discovery_config(dlp.DeleteDiscoveryConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [dlp.CreateDlpJobRequest, dict])
def test_create_dlp_job(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_dlp_job), '__call__') as call:
        call.return_value = dlp.DlpJob(name='name_value', type_=dlp.DlpJobType.INSPECT_JOB, state=dlp.DlpJob.JobState.PENDING, job_trigger_name='job_trigger_name_value')
        response = client.create_dlp_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CreateDlpJobRequest()
    assert isinstance(response, dlp.DlpJob)
    assert response.name == 'name_value'
    assert response.type_ == dlp.DlpJobType.INSPECT_JOB
    assert response.state == dlp.DlpJob.JobState.PENDING
    assert response.job_trigger_name == 'job_trigger_name_value'

def test_create_dlp_job_empty_call():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_dlp_job), '__call__') as call:
        client.create_dlp_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CreateDlpJobRequest()

@pytest.mark.asyncio
async def test_create_dlp_job_async(transport: str='grpc_asyncio', request_type=dlp.CreateDlpJobRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_dlp_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DlpJob(name='name_value', type_=dlp.DlpJobType.INSPECT_JOB, state=dlp.DlpJob.JobState.PENDING, job_trigger_name='job_trigger_name_value'))
        response = await client.create_dlp_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CreateDlpJobRequest()
    assert isinstance(response, dlp.DlpJob)
    assert response.name == 'name_value'
    assert response.type_ == dlp.DlpJobType.INSPECT_JOB
    assert response.state == dlp.DlpJob.JobState.PENDING
    assert response.job_trigger_name == 'job_trigger_name_value'

@pytest.mark.asyncio
async def test_create_dlp_job_async_from_dict():
    await test_create_dlp_job_async(request_type=dict)

def test_create_dlp_job_field_headers():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.CreateDlpJobRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_dlp_job), '__call__') as call:
        call.return_value = dlp.DlpJob()
        client.create_dlp_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_dlp_job_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.CreateDlpJobRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_dlp_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DlpJob())
        await client.create_dlp_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_dlp_job_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_dlp_job), '__call__') as call:
        call.return_value = dlp.DlpJob()
        client.create_dlp_job(parent='parent_value', inspect_job=dlp.InspectJobConfig(storage_config=storage.StorageConfig(datastore_options=storage.DatastoreOptions(partition_id=storage.PartitionId(project_id='project_id_value')))), risk_job=dlp.RiskAnalysisJobConfig(privacy_metric=dlp.PrivacyMetric(numerical_stats_config=dlp.PrivacyMetric.NumericalStatsConfig(field=storage.FieldId(name='name_value')))))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        assert args[0].risk_job == dlp.RiskAnalysisJobConfig(privacy_metric=dlp.PrivacyMetric(numerical_stats_config=dlp.PrivacyMetric.NumericalStatsConfig(field=storage.FieldId(name='name_value'))))

def test_create_dlp_job_flattened_error():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_dlp_job(dlp.CreateDlpJobRequest(), parent='parent_value', inspect_job=dlp.InspectJobConfig(storage_config=storage.StorageConfig(datastore_options=storage.DatastoreOptions(partition_id=storage.PartitionId(project_id='project_id_value')))), risk_job=dlp.RiskAnalysisJobConfig(privacy_metric=dlp.PrivacyMetric(numerical_stats_config=dlp.PrivacyMetric.NumericalStatsConfig(field=storage.FieldId(name='name_value')))))

@pytest.mark.asyncio
async def test_create_dlp_job_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_dlp_job), '__call__') as call:
        call.return_value = dlp.DlpJob()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DlpJob())
        response = await client.create_dlp_job(parent='parent_value', inspect_job=dlp.InspectJobConfig(storage_config=storage.StorageConfig(datastore_options=storage.DatastoreOptions(partition_id=storage.PartitionId(project_id='project_id_value')))), risk_job=dlp.RiskAnalysisJobConfig(privacy_metric=dlp.PrivacyMetric(numerical_stats_config=dlp.PrivacyMetric.NumericalStatsConfig(field=storage.FieldId(name='name_value')))))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        assert args[0].risk_job == dlp.RiskAnalysisJobConfig(privacy_metric=dlp.PrivacyMetric(numerical_stats_config=dlp.PrivacyMetric.NumericalStatsConfig(field=storage.FieldId(name='name_value'))))

@pytest.mark.asyncio
async def test_create_dlp_job_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_dlp_job(dlp.CreateDlpJobRequest(), parent='parent_value', inspect_job=dlp.InspectJobConfig(storage_config=storage.StorageConfig(datastore_options=storage.DatastoreOptions(partition_id=storage.PartitionId(project_id='project_id_value')))), risk_job=dlp.RiskAnalysisJobConfig(privacy_metric=dlp.PrivacyMetric(numerical_stats_config=dlp.PrivacyMetric.NumericalStatsConfig(field=storage.FieldId(name='name_value')))))

@pytest.mark.parametrize('request_type', [dlp.ListDlpJobsRequest, dict])
def test_list_dlp_jobs(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_dlp_jobs), '__call__') as call:
        call.return_value = dlp.ListDlpJobsResponse(next_page_token='next_page_token_value')
        response = client.list_dlp_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListDlpJobsRequest()
    assert isinstance(response, pagers.ListDlpJobsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_dlp_jobs_empty_call():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_dlp_jobs), '__call__') as call:
        client.list_dlp_jobs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListDlpJobsRequest()

@pytest.mark.asyncio
async def test_list_dlp_jobs_async(transport: str='grpc_asyncio', request_type=dlp.ListDlpJobsRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_dlp_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListDlpJobsResponse(next_page_token='next_page_token_value'))
        response = await client.list_dlp_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListDlpJobsRequest()
    assert isinstance(response, pagers.ListDlpJobsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_dlp_jobs_async_from_dict():
    await test_list_dlp_jobs_async(request_type=dict)

def test_list_dlp_jobs_field_headers():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.ListDlpJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_dlp_jobs), '__call__') as call:
        call.return_value = dlp.ListDlpJobsResponse()
        client.list_dlp_jobs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_dlp_jobs_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.ListDlpJobsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_dlp_jobs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListDlpJobsResponse())
        await client.list_dlp_jobs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_dlp_jobs_flattened():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_dlp_jobs), '__call__') as call:
        call.return_value = dlp.ListDlpJobsResponse()
        client.list_dlp_jobs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_dlp_jobs_flattened_error():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_dlp_jobs(dlp.ListDlpJobsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_dlp_jobs_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_dlp_jobs), '__call__') as call:
        call.return_value = dlp.ListDlpJobsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListDlpJobsResponse())
        response = await client.list_dlp_jobs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_dlp_jobs_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_dlp_jobs(dlp.ListDlpJobsRequest(), parent='parent_value')

def test_list_dlp_jobs_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_dlp_jobs), '__call__') as call:
        call.side_effect = (dlp.ListDlpJobsResponse(jobs=[dlp.DlpJob(), dlp.DlpJob(), dlp.DlpJob()], next_page_token='abc'), dlp.ListDlpJobsResponse(jobs=[], next_page_token='def'), dlp.ListDlpJobsResponse(jobs=[dlp.DlpJob()], next_page_token='ghi'), dlp.ListDlpJobsResponse(jobs=[dlp.DlpJob(), dlp.DlpJob()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_dlp_jobs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, dlp.DlpJob) for i in results))

def test_list_dlp_jobs_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_dlp_jobs), '__call__') as call:
        call.side_effect = (dlp.ListDlpJobsResponse(jobs=[dlp.DlpJob(), dlp.DlpJob(), dlp.DlpJob()], next_page_token='abc'), dlp.ListDlpJobsResponse(jobs=[], next_page_token='def'), dlp.ListDlpJobsResponse(jobs=[dlp.DlpJob()], next_page_token='ghi'), dlp.ListDlpJobsResponse(jobs=[dlp.DlpJob(), dlp.DlpJob()]), RuntimeError)
        pages = list(client.list_dlp_jobs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_dlp_jobs_async_pager():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_dlp_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (dlp.ListDlpJobsResponse(jobs=[dlp.DlpJob(), dlp.DlpJob(), dlp.DlpJob()], next_page_token='abc'), dlp.ListDlpJobsResponse(jobs=[], next_page_token='def'), dlp.ListDlpJobsResponse(jobs=[dlp.DlpJob()], next_page_token='ghi'), dlp.ListDlpJobsResponse(jobs=[dlp.DlpJob(), dlp.DlpJob()]), RuntimeError)
        async_pager = await client.list_dlp_jobs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, dlp.DlpJob) for i in responses))

@pytest.mark.asyncio
async def test_list_dlp_jobs_async_pages():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_dlp_jobs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (dlp.ListDlpJobsResponse(jobs=[dlp.DlpJob(), dlp.DlpJob(), dlp.DlpJob()], next_page_token='abc'), dlp.ListDlpJobsResponse(jobs=[], next_page_token='def'), dlp.ListDlpJobsResponse(jobs=[dlp.DlpJob()], next_page_token='ghi'), dlp.ListDlpJobsResponse(jobs=[dlp.DlpJob(), dlp.DlpJob()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_dlp_jobs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [dlp.GetDlpJobRequest, dict])
def test_get_dlp_job(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_dlp_job), '__call__') as call:
        call.return_value = dlp.DlpJob(name='name_value', type_=dlp.DlpJobType.INSPECT_JOB, state=dlp.DlpJob.JobState.PENDING, job_trigger_name='job_trigger_name_value')
        response = client.get_dlp_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.GetDlpJobRequest()
    assert isinstance(response, dlp.DlpJob)
    assert response.name == 'name_value'
    assert response.type_ == dlp.DlpJobType.INSPECT_JOB
    assert response.state == dlp.DlpJob.JobState.PENDING
    assert response.job_trigger_name == 'job_trigger_name_value'

def test_get_dlp_job_empty_call():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_dlp_job), '__call__') as call:
        client.get_dlp_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.GetDlpJobRequest()

@pytest.mark.asyncio
async def test_get_dlp_job_async(transport: str='grpc_asyncio', request_type=dlp.GetDlpJobRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_dlp_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DlpJob(name='name_value', type_=dlp.DlpJobType.INSPECT_JOB, state=dlp.DlpJob.JobState.PENDING, job_trigger_name='job_trigger_name_value'))
        response = await client.get_dlp_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.GetDlpJobRequest()
    assert isinstance(response, dlp.DlpJob)
    assert response.name == 'name_value'
    assert response.type_ == dlp.DlpJobType.INSPECT_JOB
    assert response.state == dlp.DlpJob.JobState.PENDING
    assert response.job_trigger_name == 'job_trigger_name_value'

@pytest.mark.asyncio
async def test_get_dlp_job_async_from_dict():
    await test_get_dlp_job_async(request_type=dict)

def test_get_dlp_job_field_headers():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.GetDlpJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_dlp_job), '__call__') as call:
        call.return_value = dlp.DlpJob()
        client.get_dlp_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_dlp_job_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.GetDlpJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_dlp_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DlpJob())
        await client.get_dlp_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_dlp_job_flattened():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_dlp_job), '__call__') as call:
        call.return_value = dlp.DlpJob()
        client.get_dlp_job(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_dlp_job_flattened_error():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_dlp_job(dlp.GetDlpJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_dlp_job_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_dlp_job), '__call__') as call:
        call.return_value = dlp.DlpJob()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.DlpJob())
        response = await client.get_dlp_job(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_dlp_job_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_dlp_job(dlp.GetDlpJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [dlp.DeleteDlpJobRequest, dict])
def test_delete_dlp_job(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_dlp_job), '__call__') as call:
        call.return_value = None
        response = client.delete_dlp_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeleteDlpJobRequest()
    assert response is None

def test_delete_dlp_job_empty_call():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_dlp_job), '__call__') as call:
        client.delete_dlp_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeleteDlpJobRequest()

@pytest.mark.asyncio
async def test_delete_dlp_job_async(transport: str='grpc_asyncio', request_type=dlp.DeleteDlpJobRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_dlp_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_dlp_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeleteDlpJobRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_dlp_job_async_from_dict():
    await test_delete_dlp_job_async(request_type=dict)

def test_delete_dlp_job_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.DeleteDlpJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_dlp_job), '__call__') as call:
        call.return_value = None
        client.delete_dlp_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_dlp_job_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.DeleteDlpJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_dlp_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_dlp_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_dlp_job_flattened():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_dlp_job), '__call__') as call:
        call.return_value = None
        client.delete_dlp_job(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_dlp_job_flattened_error():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_dlp_job(dlp.DeleteDlpJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_dlp_job_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_dlp_job), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_dlp_job(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_dlp_job_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_dlp_job(dlp.DeleteDlpJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [dlp.CancelDlpJobRequest, dict])
def test_cancel_dlp_job(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.cancel_dlp_job), '__call__') as call:
        call.return_value = None
        response = client.cancel_dlp_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CancelDlpJobRequest()
    assert response is None

def test_cancel_dlp_job_empty_call():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.cancel_dlp_job), '__call__') as call:
        client.cancel_dlp_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CancelDlpJobRequest()

@pytest.mark.asyncio
async def test_cancel_dlp_job_async(transport: str='grpc_asyncio', request_type=dlp.CancelDlpJobRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.cancel_dlp_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_dlp_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CancelDlpJobRequest()
    assert response is None

@pytest.mark.asyncio
async def test_cancel_dlp_job_async_from_dict():
    await test_cancel_dlp_job_async(request_type=dict)

def test_cancel_dlp_job_field_headers():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.CancelDlpJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.cancel_dlp_job), '__call__') as call:
        call.return_value = None
        client.cancel_dlp_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_cancel_dlp_job_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.CancelDlpJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.cancel_dlp_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.cancel_dlp_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [dlp.CreateStoredInfoTypeRequest, dict])
def test_create_stored_info_type(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_stored_info_type), '__call__') as call:
        call.return_value = dlp.StoredInfoType(name='name_value')
        response = client.create_stored_info_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CreateStoredInfoTypeRequest()
    assert isinstance(response, dlp.StoredInfoType)
    assert response.name == 'name_value'

def test_create_stored_info_type_empty_call():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_stored_info_type), '__call__') as call:
        client.create_stored_info_type()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CreateStoredInfoTypeRequest()

@pytest.mark.asyncio
async def test_create_stored_info_type_async(transport: str='grpc_asyncio', request_type=dlp.CreateStoredInfoTypeRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_stored_info_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.StoredInfoType(name='name_value'))
        response = await client.create_stored_info_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.CreateStoredInfoTypeRequest()
    assert isinstance(response, dlp.StoredInfoType)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_create_stored_info_type_async_from_dict():
    await test_create_stored_info_type_async(request_type=dict)

def test_create_stored_info_type_field_headers():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.CreateStoredInfoTypeRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_stored_info_type), '__call__') as call:
        call.return_value = dlp.StoredInfoType()
        client.create_stored_info_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_stored_info_type_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.CreateStoredInfoTypeRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_stored_info_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.StoredInfoType())
        await client.create_stored_info_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_stored_info_type_flattened():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_stored_info_type), '__call__') as call:
        call.return_value = dlp.StoredInfoType()
        client.create_stored_info_type(parent='parent_value', config=dlp.StoredInfoTypeConfig(display_name='display_name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].config
        mock_val = dlp.StoredInfoTypeConfig(display_name='display_name_value')
        assert arg == mock_val

def test_create_stored_info_type_flattened_error():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_stored_info_type(dlp.CreateStoredInfoTypeRequest(), parent='parent_value', config=dlp.StoredInfoTypeConfig(display_name='display_name_value'))

@pytest.mark.asyncio
async def test_create_stored_info_type_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_stored_info_type), '__call__') as call:
        call.return_value = dlp.StoredInfoType()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.StoredInfoType())
        response = await client.create_stored_info_type(parent='parent_value', config=dlp.StoredInfoTypeConfig(display_name='display_name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].config
        mock_val = dlp.StoredInfoTypeConfig(display_name='display_name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_stored_info_type_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_stored_info_type(dlp.CreateStoredInfoTypeRequest(), parent='parent_value', config=dlp.StoredInfoTypeConfig(display_name='display_name_value'))

@pytest.mark.parametrize('request_type', [dlp.UpdateStoredInfoTypeRequest, dict])
def test_update_stored_info_type(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_stored_info_type), '__call__') as call:
        call.return_value = dlp.StoredInfoType(name='name_value')
        response = client.update_stored_info_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.UpdateStoredInfoTypeRequest()
    assert isinstance(response, dlp.StoredInfoType)
    assert response.name == 'name_value'

def test_update_stored_info_type_empty_call():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_stored_info_type), '__call__') as call:
        client.update_stored_info_type()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.UpdateStoredInfoTypeRequest()

@pytest.mark.asyncio
async def test_update_stored_info_type_async(transport: str='grpc_asyncio', request_type=dlp.UpdateStoredInfoTypeRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_stored_info_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.StoredInfoType(name='name_value'))
        response = await client.update_stored_info_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.UpdateStoredInfoTypeRequest()
    assert isinstance(response, dlp.StoredInfoType)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_update_stored_info_type_async_from_dict():
    await test_update_stored_info_type_async(request_type=dict)

def test_update_stored_info_type_field_headers():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.UpdateStoredInfoTypeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_stored_info_type), '__call__') as call:
        call.return_value = dlp.StoredInfoType()
        client.update_stored_info_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_stored_info_type_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.UpdateStoredInfoTypeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_stored_info_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.StoredInfoType())
        await client.update_stored_info_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_update_stored_info_type_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_stored_info_type), '__call__') as call:
        call.return_value = dlp.StoredInfoType()
        client.update_stored_info_type(name='name_value', config=dlp.StoredInfoTypeConfig(display_name='display_name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].config
        mock_val = dlp.StoredInfoTypeConfig(display_name='display_name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_stored_info_type_flattened_error():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_stored_info_type(dlp.UpdateStoredInfoTypeRequest(), name='name_value', config=dlp.StoredInfoTypeConfig(display_name='display_name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_stored_info_type_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_stored_info_type), '__call__') as call:
        call.return_value = dlp.StoredInfoType()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.StoredInfoType())
        response = await client.update_stored_info_type(name='name_value', config=dlp.StoredInfoTypeConfig(display_name='display_name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].config
        mock_val = dlp.StoredInfoTypeConfig(display_name='display_name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_stored_info_type_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_stored_info_type(dlp.UpdateStoredInfoTypeRequest(), name='name_value', config=dlp.StoredInfoTypeConfig(display_name='display_name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [dlp.GetStoredInfoTypeRequest, dict])
def test_get_stored_info_type(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_stored_info_type), '__call__') as call:
        call.return_value = dlp.StoredInfoType(name='name_value')
        response = client.get_stored_info_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.GetStoredInfoTypeRequest()
    assert isinstance(response, dlp.StoredInfoType)
    assert response.name == 'name_value'

def test_get_stored_info_type_empty_call():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_stored_info_type), '__call__') as call:
        client.get_stored_info_type()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.GetStoredInfoTypeRequest()

@pytest.mark.asyncio
async def test_get_stored_info_type_async(transport: str='grpc_asyncio', request_type=dlp.GetStoredInfoTypeRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_stored_info_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.StoredInfoType(name='name_value'))
        response = await client.get_stored_info_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.GetStoredInfoTypeRequest()
    assert isinstance(response, dlp.StoredInfoType)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_stored_info_type_async_from_dict():
    await test_get_stored_info_type_async(request_type=dict)

def test_get_stored_info_type_field_headers():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.GetStoredInfoTypeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_stored_info_type), '__call__') as call:
        call.return_value = dlp.StoredInfoType()
        client.get_stored_info_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_stored_info_type_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.GetStoredInfoTypeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_stored_info_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.StoredInfoType())
        await client.get_stored_info_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_stored_info_type_flattened():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_stored_info_type), '__call__') as call:
        call.return_value = dlp.StoredInfoType()
        client.get_stored_info_type(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_stored_info_type_flattened_error():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_stored_info_type(dlp.GetStoredInfoTypeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_stored_info_type_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_stored_info_type), '__call__') as call:
        call.return_value = dlp.StoredInfoType()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.StoredInfoType())
        response = await client.get_stored_info_type(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_stored_info_type_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_stored_info_type(dlp.GetStoredInfoTypeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [dlp.ListStoredInfoTypesRequest, dict])
def test_list_stored_info_types(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_stored_info_types), '__call__') as call:
        call.return_value = dlp.ListStoredInfoTypesResponse(next_page_token='next_page_token_value')
        response = client.list_stored_info_types(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListStoredInfoTypesRequest()
    assert isinstance(response, pagers.ListStoredInfoTypesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_stored_info_types_empty_call():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_stored_info_types), '__call__') as call:
        client.list_stored_info_types()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListStoredInfoTypesRequest()

@pytest.mark.asyncio
async def test_list_stored_info_types_async(transport: str='grpc_asyncio', request_type=dlp.ListStoredInfoTypesRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_stored_info_types), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListStoredInfoTypesResponse(next_page_token='next_page_token_value'))
        response = await client.list_stored_info_types(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.ListStoredInfoTypesRequest()
    assert isinstance(response, pagers.ListStoredInfoTypesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_stored_info_types_async_from_dict():
    await test_list_stored_info_types_async(request_type=dict)

def test_list_stored_info_types_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.ListStoredInfoTypesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_stored_info_types), '__call__') as call:
        call.return_value = dlp.ListStoredInfoTypesResponse()
        client.list_stored_info_types(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_stored_info_types_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.ListStoredInfoTypesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_stored_info_types), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListStoredInfoTypesResponse())
        await client.list_stored_info_types(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_stored_info_types_flattened():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_stored_info_types), '__call__') as call:
        call.return_value = dlp.ListStoredInfoTypesResponse()
        client.list_stored_info_types(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_stored_info_types_flattened_error():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_stored_info_types(dlp.ListStoredInfoTypesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_stored_info_types_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_stored_info_types), '__call__') as call:
        call.return_value = dlp.ListStoredInfoTypesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.ListStoredInfoTypesResponse())
        response = await client.list_stored_info_types(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_stored_info_types_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_stored_info_types(dlp.ListStoredInfoTypesRequest(), parent='parent_value')

def test_list_stored_info_types_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_stored_info_types), '__call__') as call:
        call.side_effect = (dlp.ListStoredInfoTypesResponse(stored_info_types=[dlp.StoredInfoType(), dlp.StoredInfoType(), dlp.StoredInfoType()], next_page_token='abc'), dlp.ListStoredInfoTypesResponse(stored_info_types=[], next_page_token='def'), dlp.ListStoredInfoTypesResponse(stored_info_types=[dlp.StoredInfoType()], next_page_token='ghi'), dlp.ListStoredInfoTypesResponse(stored_info_types=[dlp.StoredInfoType(), dlp.StoredInfoType()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_stored_info_types(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, dlp.StoredInfoType) for i in results))

def test_list_stored_info_types_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_stored_info_types), '__call__') as call:
        call.side_effect = (dlp.ListStoredInfoTypesResponse(stored_info_types=[dlp.StoredInfoType(), dlp.StoredInfoType(), dlp.StoredInfoType()], next_page_token='abc'), dlp.ListStoredInfoTypesResponse(stored_info_types=[], next_page_token='def'), dlp.ListStoredInfoTypesResponse(stored_info_types=[dlp.StoredInfoType()], next_page_token='ghi'), dlp.ListStoredInfoTypesResponse(stored_info_types=[dlp.StoredInfoType(), dlp.StoredInfoType()]), RuntimeError)
        pages = list(client.list_stored_info_types(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_stored_info_types_async_pager():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_stored_info_types), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (dlp.ListStoredInfoTypesResponse(stored_info_types=[dlp.StoredInfoType(), dlp.StoredInfoType(), dlp.StoredInfoType()], next_page_token='abc'), dlp.ListStoredInfoTypesResponse(stored_info_types=[], next_page_token='def'), dlp.ListStoredInfoTypesResponse(stored_info_types=[dlp.StoredInfoType()], next_page_token='ghi'), dlp.ListStoredInfoTypesResponse(stored_info_types=[dlp.StoredInfoType(), dlp.StoredInfoType()]), RuntimeError)
        async_pager = await client.list_stored_info_types(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, dlp.StoredInfoType) for i in responses))

@pytest.mark.asyncio
async def test_list_stored_info_types_async_pages():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_stored_info_types), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (dlp.ListStoredInfoTypesResponse(stored_info_types=[dlp.StoredInfoType(), dlp.StoredInfoType(), dlp.StoredInfoType()], next_page_token='abc'), dlp.ListStoredInfoTypesResponse(stored_info_types=[], next_page_token='def'), dlp.ListStoredInfoTypesResponse(stored_info_types=[dlp.StoredInfoType()], next_page_token='ghi'), dlp.ListStoredInfoTypesResponse(stored_info_types=[dlp.StoredInfoType(), dlp.StoredInfoType()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_stored_info_types(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [dlp.DeleteStoredInfoTypeRequest, dict])
def test_delete_stored_info_type(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_stored_info_type), '__call__') as call:
        call.return_value = None
        response = client.delete_stored_info_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeleteStoredInfoTypeRequest()
    assert response is None

def test_delete_stored_info_type_empty_call():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_stored_info_type), '__call__') as call:
        client.delete_stored_info_type()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeleteStoredInfoTypeRequest()

@pytest.mark.asyncio
async def test_delete_stored_info_type_async(transport: str='grpc_asyncio', request_type=dlp.DeleteStoredInfoTypeRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_stored_info_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_stored_info_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.DeleteStoredInfoTypeRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_stored_info_type_async_from_dict():
    await test_delete_stored_info_type_async(request_type=dict)

def test_delete_stored_info_type_field_headers():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.DeleteStoredInfoTypeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_stored_info_type), '__call__') as call:
        call.return_value = None
        client.delete_stored_info_type(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_stored_info_type_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.DeleteStoredInfoTypeRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_stored_info_type), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_stored_info_type(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_stored_info_type_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_stored_info_type), '__call__') as call:
        call.return_value = None
        client.delete_stored_info_type(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_stored_info_type_flattened_error():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_stored_info_type(dlp.DeleteStoredInfoTypeRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_stored_info_type_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_stored_info_type), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_stored_info_type(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_stored_info_type_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_stored_info_type(dlp.DeleteStoredInfoTypeRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [dlp.HybridInspectDlpJobRequest, dict])
def test_hybrid_inspect_dlp_job(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.hybrid_inspect_dlp_job), '__call__') as call:
        call.return_value = dlp.HybridInspectResponse()
        response = client.hybrid_inspect_dlp_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.HybridInspectDlpJobRequest()
    assert isinstance(response, dlp.HybridInspectResponse)

def test_hybrid_inspect_dlp_job_empty_call():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.hybrid_inspect_dlp_job), '__call__') as call:
        client.hybrid_inspect_dlp_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.HybridInspectDlpJobRequest()

@pytest.mark.asyncio
async def test_hybrid_inspect_dlp_job_async(transport: str='grpc_asyncio', request_type=dlp.HybridInspectDlpJobRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.hybrid_inspect_dlp_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.HybridInspectResponse())
        response = await client.hybrid_inspect_dlp_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.HybridInspectDlpJobRequest()
    assert isinstance(response, dlp.HybridInspectResponse)

@pytest.mark.asyncio
async def test_hybrid_inspect_dlp_job_async_from_dict():
    await test_hybrid_inspect_dlp_job_async(request_type=dict)

def test_hybrid_inspect_dlp_job_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.HybridInspectDlpJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.hybrid_inspect_dlp_job), '__call__') as call:
        call.return_value = dlp.HybridInspectResponse()
        client.hybrid_inspect_dlp_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_hybrid_inspect_dlp_job_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.HybridInspectDlpJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.hybrid_inspect_dlp_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.HybridInspectResponse())
        await client.hybrid_inspect_dlp_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_hybrid_inspect_dlp_job_flattened():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.hybrid_inspect_dlp_job), '__call__') as call:
        call.return_value = dlp.HybridInspectResponse()
        client.hybrid_inspect_dlp_job(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_hybrid_inspect_dlp_job_flattened_error():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.hybrid_inspect_dlp_job(dlp.HybridInspectDlpJobRequest(), name='name_value')

@pytest.mark.asyncio
async def test_hybrid_inspect_dlp_job_flattened_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.hybrid_inspect_dlp_job), '__call__') as call:
        call.return_value = dlp.HybridInspectResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(dlp.HybridInspectResponse())
        response = await client.hybrid_inspect_dlp_job(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_hybrid_inspect_dlp_job_flattened_error_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.hybrid_inspect_dlp_job(dlp.HybridInspectDlpJobRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [dlp.FinishDlpJobRequest, dict])
def test_finish_dlp_job(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.finish_dlp_job), '__call__') as call:
        call.return_value = None
        response = client.finish_dlp_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.FinishDlpJobRequest()
    assert response is None

def test_finish_dlp_job_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.finish_dlp_job), '__call__') as call:
        client.finish_dlp_job()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.FinishDlpJobRequest()

@pytest.mark.asyncio
async def test_finish_dlp_job_async(transport: str='grpc_asyncio', request_type=dlp.FinishDlpJobRequest):
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.finish_dlp_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.finish_dlp_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == dlp.FinishDlpJobRequest()
    assert response is None

@pytest.mark.asyncio
async def test_finish_dlp_job_async_from_dict():
    await test_finish_dlp_job_async(request_type=dict)

def test_finish_dlp_job_field_headers():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.FinishDlpJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.finish_dlp_job), '__call__') as call:
        call.return_value = None
        client.finish_dlp_job(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_finish_dlp_job_field_headers_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = dlp.FinishDlpJobRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.finish_dlp_job), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.finish_dlp_job(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [dlp.InspectContentRequest, dict])
def test_inspect_content_rest(request_type):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.InspectContentResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.InspectContentResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.inspect_content(request)
    assert isinstance(response, dlp.InspectContentResponse)

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_inspect_content_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_inspect_content') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_inspect_content') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.InspectContentRequest.pb(dlp.InspectContentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.InspectContentResponse.to_json(dlp.InspectContentResponse())
        request = dlp.InspectContentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.InspectContentResponse()
        client.inspect_content(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_inspect_content_rest_bad_request(transport: str='rest', request_type=dlp.InspectContentRequest):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.inspect_content(request)

def test_inspect_content_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.RedactImageRequest, dict])
def test_redact_image_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.RedactImageResponse(redacted_image=b'redacted_image_blob', extracted_text='extracted_text_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.RedactImageResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.redact_image(request)
    assert isinstance(response, dlp.RedactImageResponse)
    assert response.redacted_image == b'redacted_image_blob'
    assert response.extracted_text == 'extracted_text_value'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_redact_image_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_redact_image') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_redact_image') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.RedactImageRequest.pb(dlp.RedactImageRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.RedactImageResponse.to_json(dlp.RedactImageResponse())
        request = dlp.RedactImageRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.RedactImageResponse()
        client.redact_image(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_redact_image_rest_bad_request(transport: str='rest', request_type=dlp.RedactImageRequest):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.redact_image(request)

def test_redact_image_rest_error():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.DeidentifyContentRequest, dict])
def test_deidentify_content_rest(request_type):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.DeidentifyContentResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.DeidentifyContentResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.deidentify_content(request)
    assert isinstance(response, dlp.DeidentifyContentResponse)

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_deidentify_content_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_deidentify_content') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_deidentify_content') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.DeidentifyContentRequest.pb(dlp.DeidentifyContentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.DeidentifyContentResponse.to_json(dlp.DeidentifyContentResponse())
        request = dlp.DeidentifyContentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.DeidentifyContentResponse()
        client.deidentify_content(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_deidentify_content_rest_bad_request(transport: str='rest', request_type=dlp.DeidentifyContentRequest):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.deidentify_content(request)

def test_deidentify_content_rest_error():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.ReidentifyContentRequest, dict])
def test_reidentify_content_rest(request_type):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.ReidentifyContentResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.ReidentifyContentResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.reidentify_content(request)
    assert isinstance(response, dlp.ReidentifyContentResponse)

def test_reidentify_content_rest_required_fields(request_type=dlp.ReidentifyContentRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).reidentify_content._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).reidentify_content._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.ReidentifyContentResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.ReidentifyContentResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.reidentify_content(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_reidentify_content_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.reidentify_content._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_reidentify_content_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_reidentify_content') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_reidentify_content') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.ReidentifyContentRequest.pb(dlp.ReidentifyContentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.ReidentifyContentResponse.to_json(dlp.ReidentifyContentResponse())
        request = dlp.ReidentifyContentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.ReidentifyContentResponse()
        client.reidentify_content(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_reidentify_content_rest_bad_request(transport: str='rest', request_type=dlp.ReidentifyContentRequest):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.reidentify_content(request)

def test_reidentify_content_rest_error():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.ListInfoTypesRequest, dict])
def test_list_info_types_rest(request_type):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.ListInfoTypesResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.ListInfoTypesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_info_types(request)
    assert isinstance(response, dlp.ListInfoTypesResponse)

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_info_types_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_list_info_types') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_list_info_types') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.ListInfoTypesRequest.pb(dlp.ListInfoTypesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.ListInfoTypesResponse.to_json(dlp.ListInfoTypesResponse())
        request = dlp.ListInfoTypesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.ListInfoTypesResponse()
        client.list_info_types(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_info_types_rest_bad_request(transport: str='rest', request_type=dlp.ListInfoTypesRequest):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_info_types(request)

def test_list_info_types_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.ListInfoTypesResponse()
        sample_request = {}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.ListInfoTypesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_info_types(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/infoTypes' % client.transport._host, args[1])

def test_list_info_types_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_info_types(dlp.ListInfoTypesRequest(), parent='parent_value')

def test_list_info_types_rest_error():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.CreateInspectTemplateRequest, dict])
def test_create_inspect_template_rest(request_type):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'organizations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.InspectTemplate(name='name_value', display_name='display_name_value', description='description_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.InspectTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_inspect_template(request)
    assert isinstance(response, dlp.InspectTemplate)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'

def test_create_inspect_template_rest_required_fields(request_type=dlp.CreateInspectTemplateRequest):
    if False:
        return 10
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_inspect_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_inspect_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.InspectTemplate()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.InspectTemplate.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_inspect_template(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_inspect_template_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_inspect_template._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'inspectTemplate'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_inspect_template_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_create_inspect_template') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_create_inspect_template') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.CreateInspectTemplateRequest.pb(dlp.CreateInspectTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.InspectTemplate.to_json(dlp.InspectTemplate())
        request = dlp.CreateInspectTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.InspectTemplate()
        client.create_inspect_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_inspect_template_rest_bad_request(transport: str='rest', request_type=dlp.CreateInspectTemplateRequest):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'organizations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_inspect_template(request)

def test_create_inspect_template_rest_flattened():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.InspectTemplate()
        sample_request = {'parent': 'organizations/sample1'}
        mock_args = dict(parent='parent_value', inspect_template=dlp.InspectTemplate(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.InspectTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_inspect_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=organizations/*}/inspectTemplates' % client.transport._host, args[1])

def test_create_inspect_template_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_inspect_template(dlp.CreateInspectTemplateRequest(), parent='parent_value', inspect_template=dlp.InspectTemplate(name='name_value'))

def test_create_inspect_template_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.UpdateInspectTemplateRequest, dict])
def test_update_inspect_template_rest(request_type):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'organizations/sample1/inspectTemplates/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.InspectTemplate(name='name_value', display_name='display_name_value', description='description_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.InspectTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_inspect_template(request)
    assert isinstance(response, dlp.InspectTemplate)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'

def test_update_inspect_template_rest_required_fields(request_type=dlp.UpdateInspectTemplateRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_inspect_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_inspect_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.InspectTemplate()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.InspectTemplate.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_inspect_template(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_inspect_template_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_inspect_template._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_inspect_template_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_update_inspect_template') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_update_inspect_template') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.UpdateInspectTemplateRequest.pb(dlp.UpdateInspectTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.InspectTemplate.to_json(dlp.InspectTemplate())
        request = dlp.UpdateInspectTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.InspectTemplate()
        client.update_inspect_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_inspect_template_rest_bad_request(transport: str='rest', request_type=dlp.UpdateInspectTemplateRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'organizations/sample1/inspectTemplates/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_inspect_template(request)

def test_update_inspect_template_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.InspectTemplate()
        sample_request = {'name': 'organizations/sample1/inspectTemplates/sample2'}
        mock_args = dict(name='name_value', inspect_template=dlp.InspectTemplate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.InspectTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_inspect_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=organizations/*/inspectTemplates/*}' % client.transport._host, args[1])

def test_update_inspect_template_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_inspect_template(dlp.UpdateInspectTemplateRequest(), name='name_value', inspect_template=dlp.InspectTemplate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_inspect_template_rest_error():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.GetInspectTemplateRequest, dict])
def test_get_inspect_template_rest(request_type):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'organizations/sample1/inspectTemplates/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.InspectTemplate(name='name_value', display_name='display_name_value', description='description_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.InspectTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_inspect_template(request)
    assert isinstance(response, dlp.InspectTemplate)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'

def test_get_inspect_template_rest_required_fields(request_type=dlp.GetInspectTemplateRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_inspect_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_inspect_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.InspectTemplate()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.InspectTemplate.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_inspect_template(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_inspect_template_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_inspect_template._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_inspect_template_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_get_inspect_template') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_get_inspect_template') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.GetInspectTemplateRequest.pb(dlp.GetInspectTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.InspectTemplate.to_json(dlp.InspectTemplate())
        request = dlp.GetInspectTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.InspectTemplate()
        client.get_inspect_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_inspect_template_rest_bad_request(transport: str='rest', request_type=dlp.GetInspectTemplateRequest):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'organizations/sample1/inspectTemplates/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_inspect_template(request)

def test_get_inspect_template_rest_flattened():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.InspectTemplate()
        sample_request = {'name': 'organizations/sample1/inspectTemplates/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.InspectTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_inspect_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=organizations/*/inspectTemplates/*}' % client.transport._host, args[1])

def test_get_inspect_template_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_inspect_template(dlp.GetInspectTemplateRequest(), name='name_value')

def test_get_inspect_template_rest_error():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.ListInspectTemplatesRequest, dict])
def test_list_inspect_templates_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'organizations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.ListInspectTemplatesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.ListInspectTemplatesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_inspect_templates(request)
    assert isinstance(response, pagers.ListInspectTemplatesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_inspect_templates_rest_required_fields(request_type=dlp.ListInspectTemplatesRequest):
    if False:
        return 10
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_inspect_templates._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_inspect_templates._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('location_id', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.ListInspectTemplatesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.ListInspectTemplatesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_inspect_templates(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_inspect_templates_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_inspect_templates._get_unset_required_fields({})
    assert set(unset_fields) == set(('locationId', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_inspect_templates_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_list_inspect_templates') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_list_inspect_templates') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.ListInspectTemplatesRequest.pb(dlp.ListInspectTemplatesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.ListInspectTemplatesResponse.to_json(dlp.ListInspectTemplatesResponse())
        request = dlp.ListInspectTemplatesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.ListInspectTemplatesResponse()
        client.list_inspect_templates(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_inspect_templates_rest_bad_request(transport: str='rest', request_type=dlp.ListInspectTemplatesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'organizations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_inspect_templates(request)

def test_list_inspect_templates_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.ListInspectTemplatesResponse()
        sample_request = {'parent': 'organizations/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.ListInspectTemplatesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_inspect_templates(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=organizations/*}/inspectTemplates' % client.transport._host, args[1])

def test_list_inspect_templates_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_inspect_templates(dlp.ListInspectTemplatesRequest(), parent='parent_value')

def test_list_inspect_templates_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (dlp.ListInspectTemplatesResponse(inspect_templates=[dlp.InspectTemplate(), dlp.InspectTemplate(), dlp.InspectTemplate()], next_page_token='abc'), dlp.ListInspectTemplatesResponse(inspect_templates=[], next_page_token='def'), dlp.ListInspectTemplatesResponse(inspect_templates=[dlp.InspectTemplate()], next_page_token='ghi'), dlp.ListInspectTemplatesResponse(inspect_templates=[dlp.InspectTemplate(), dlp.InspectTemplate()]))
        response = response + response
        response = tuple((dlp.ListInspectTemplatesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'organizations/sample1'}
        pager = client.list_inspect_templates(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, dlp.InspectTemplate) for i in results))
        pages = list(client.list_inspect_templates(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [dlp.DeleteInspectTemplateRequest, dict])
def test_delete_inspect_template_rest(request_type):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'organizations/sample1/inspectTemplates/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_inspect_template(request)
    assert response is None

def test_delete_inspect_template_rest_required_fields(request_type=dlp.DeleteInspectTemplateRequest):
    if False:
        return 10
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_inspect_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_inspect_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_inspect_template(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_inspect_template_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_inspect_template._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_inspect_template_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_delete_inspect_template') as pre:
        pre.assert_not_called()
        pb_message = dlp.DeleteInspectTemplateRequest.pb(dlp.DeleteInspectTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = dlp.DeleteInspectTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_inspect_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_inspect_template_rest_bad_request(transport: str='rest', request_type=dlp.DeleteInspectTemplateRequest):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'organizations/sample1/inspectTemplates/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_inspect_template(request)

def test_delete_inspect_template_rest_flattened():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'organizations/sample1/inspectTemplates/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_inspect_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=organizations/*/inspectTemplates/*}' % client.transport._host, args[1])

def test_delete_inspect_template_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_inspect_template(dlp.DeleteInspectTemplateRequest(), name='name_value')

def test_delete_inspect_template_rest_error():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.CreateDeidentifyTemplateRequest, dict])
def test_create_deidentify_template_rest(request_type):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'organizations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.DeidentifyTemplate(name='name_value', display_name='display_name_value', description='description_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.DeidentifyTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_deidentify_template(request)
    assert isinstance(response, dlp.DeidentifyTemplate)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'

def test_create_deidentify_template_rest_required_fields(request_type=dlp.CreateDeidentifyTemplateRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_deidentify_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_deidentify_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.DeidentifyTemplate()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.DeidentifyTemplate.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_deidentify_template(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_deidentify_template_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_deidentify_template._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'deidentifyTemplate'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_deidentify_template_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_create_deidentify_template') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_create_deidentify_template') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.CreateDeidentifyTemplateRequest.pb(dlp.CreateDeidentifyTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.DeidentifyTemplate.to_json(dlp.DeidentifyTemplate())
        request = dlp.CreateDeidentifyTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.DeidentifyTemplate()
        client.create_deidentify_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_deidentify_template_rest_bad_request(transport: str='rest', request_type=dlp.CreateDeidentifyTemplateRequest):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'organizations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_deidentify_template(request)

def test_create_deidentify_template_rest_flattened():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.DeidentifyTemplate()
        sample_request = {'parent': 'organizations/sample1'}
        mock_args = dict(parent='parent_value', deidentify_template=dlp.DeidentifyTemplate(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.DeidentifyTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_deidentify_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=organizations/*}/deidentifyTemplates' % client.transport._host, args[1])

def test_create_deidentify_template_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_deidentify_template(dlp.CreateDeidentifyTemplateRequest(), parent='parent_value', deidentify_template=dlp.DeidentifyTemplate(name='name_value'))

def test_create_deidentify_template_rest_error():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.UpdateDeidentifyTemplateRequest, dict])
def test_update_deidentify_template_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'organizations/sample1/deidentifyTemplates/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.DeidentifyTemplate(name='name_value', display_name='display_name_value', description='description_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.DeidentifyTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_deidentify_template(request)
    assert isinstance(response, dlp.DeidentifyTemplate)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'

def test_update_deidentify_template_rest_required_fields(request_type=dlp.UpdateDeidentifyTemplateRequest):
    if False:
        return 10
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_deidentify_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_deidentify_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.DeidentifyTemplate()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.DeidentifyTemplate.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_deidentify_template(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_deidentify_template_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_deidentify_template._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_deidentify_template_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_update_deidentify_template') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_update_deidentify_template') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.UpdateDeidentifyTemplateRequest.pb(dlp.UpdateDeidentifyTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.DeidentifyTemplate.to_json(dlp.DeidentifyTemplate())
        request = dlp.UpdateDeidentifyTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.DeidentifyTemplate()
        client.update_deidentify_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_deidentify_template_rest_bad_request(transport: str='rest', request_type=dlp.UpdateDeidentifyTemplateRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'organizations/sample1/deidentifyTemplates/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_deidentify_template(request)

def test_update_deidentify_template_rest_flattened():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.DeidentifyTemplate()
        sample_request = {'name': 'organizations/sample1/deidentifyTemplates/sample2'}
        mock_args = dict(name='name_value', deidentify_template=dlp.DeidentifyTemplate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.DeidentifyTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_deidentify_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=organizations/*/deidentifyTemplates/*}' % client.transport._host, args[1])

def test_update_deidentify_template_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_deidentify_template(dlp.UpdateDeidentifyTemplateRequest(), name='name_value', deidentify_template=dlp.DeidentifyTemplate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_deidentify_template_rest_error():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.GetDeidentifyTemplateRequest, dict])
def test_get_deidentify_template_rest(request_type):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'organizations/sample1/deidentifyTemplates/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.DeidentifyTemplate(name='name_value', display_name='display_name_value', description='description_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.DeidentifyTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_deidentify_template(request)
    assert isinstance(response, dlp.DeidentifyTemplate)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'

def test_get_deidentify_template_rest_required_fields(request_type=dlp.GetDeidentifyTemplateRequest):
    if False:
        print('Hello World!')
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_deidentify_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_deidentify_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.DeidentifyTemplate()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.DeidentifyTemplate.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_deidentify_template(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_deidentify_template_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_deidentify_template._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_deidentify_template_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_get_deidentify_template') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_get_deidentify_template') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.GetDeidentifyTemplateRequest.pb(dlp.GetDeidentifyTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.DeidentifyTemplate.to_json(dlp.DeidentifyTemplate())
        request = dlp.GetDeidentifyTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.DeidentifyTemplate()
        client.get_deidentify_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_deidentify_template_rest_bad_request(transport: str='rest', request_type=dlp.GetDeidentifyTemplateRequest):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'organizations/sample1/deidentifyTemplates/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_deidentify_template(request)

def test_get_deidentify_template_rest_flattened():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.DeidentifyTemplate()
        sample_request = {'name': 'organizations/sample1/deidentifyTemplates/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.DeidentifyTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_deidentify_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=organizations/*/deidentifyTemplates/*}' % client.transport._host, args[1])

def test_get_deidentify_template_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_deidentify_template(dlp.GetDeidentifyTemplateRequest(), name='name_value')

def test_get_deidentify_template_rest_error():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.ListDeidentifyTemplatesRequest, dict])
def test_list_deidentify_templates_rest(request_type):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'organizations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.ListDeidentifyTemplatesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.ListDeidentifyTemplatesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_deidentify_templates(request)
    assert isinstance(response, pagers.ListDeidentifyTemplatesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_deidentify_templates_rest_required_fields(request_type=dlp.ListDeidentifyTemplatesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_deidentify_templates._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_deidentify_templates._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('location_id', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.ListDeidentifyTemplatesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.ListDeidentifyTemplatesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_deidentify_templates(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_deidentify_templates_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_deidentify_templates._get_unset_required_fields({})
    assert set(unset_fields) == set(('locationId', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_deidentify_templates_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_list_deidentify_templates') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_list_deidentify_templates') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.ListDeidentifyTemplatesRequest.pb(dlp.ListDeidentifyTemplatesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.ListDeidentifyTemplatesResponse.to_json(dlp.ListDeidentifyTemplatesResponse())
        request = dlp.ListDeidentifyTemplatesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.ListDeidentifyTemplatesResponse()
        client.list_deidentify_templates(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_deidentify_templates_rest_bad_request(transport: str='rest', request_type=dlp.ListDeidentifyTemplatesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'organizations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_deidentify_templates(request)

def test_list_deidentify_templates_rest_flattened():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.ListDeidentifyTemplatesResponse()
        sample_request = {'parent': 'organizations/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.ListDeidentifyTemplatesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_deidentify_templates(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=organizations/*}/deidentifyTemplates' % client.transport._host, args[1])

def test_list_deidentify_templates_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_deidentify_templates(dlp.ListDeidentifyTemplatesRequest(), parent='parent_value')

def test_list_deidentify_templates_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[dlp.DeidentifyTemplate(), dlp.DeidentifyTemplate(), dlp.DeidentifyTemplate()], next_page_token='abc'), dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[], next_page_token='def'), dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[dlp.DeidentifyTemplate()], next_page_token='ghi'), dlp.ListDeidentifyTemplatesResponse(deidentify_templates=[dlp.DeidentifyTemplate(), dlp.DeidentifyTemplate()]))
        response = response + response
        response = tuple((dlp.ListDeidentifyTemplatesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'organizations/sample1'}
        pager = client.list_deidentify_templates(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, dlp.DeidentifyTemplate) for i in results))
        pages = list(client.list_deidentify_templates(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [dlp.DeleteDeidentifyTemplateRequest, dict])
def test_delete_deidentify_template_rest(request_type):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'organizations/sample1/deidentifyTemplates/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_deidentify_template(request)
    assert response is None

def test_delete_deidentify_template_rest_required_fields(request_type=dlp.DeleteDeidentifyTemplateRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_deidentify_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_deidentify_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_deidentify_template(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_deidentify_template_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_deidentify_template._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_deidentify_template_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_delete_deidentify_template') as pre:
        pre.assert_not_called()
        pb_message = dlp.DeleteDeidentifyTemplateRequest.pb(dlp.DeleteDeidentifyTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = dlp.DeleteDeidentifyTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_deidentify_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_deidentify_template_rest_bad_request(transport: str='rest', request_type=dlp.DeleteDeidentifyTemplateRequest):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'organizations/sample1/deidentifyTemplates/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_deidentify_template(request)

def test_delete_deidentify_template_rest_flattened():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'organizations/sample1/deidentifyTemplates/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_deidentify_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=organizations/*/deidentifyTemplates/*}' % client.transport._host, args[1])

def test_delete_deidentify_template_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_deidentify_template(dlp.DeleteDeidentifyTemplateRequest(), name='name_value')

def test_delete_deidentify_template_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.CreateJobTriggerRequest, dict])
def test_create_job_trigger_rest(request_type):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.JobTrigger(name='name_value', display_name='display_name_value', description='description_value', status=dlp.JobTrigger.Status.HEALTHY)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.JobTrigger.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_job_trigger(request)
    assert isinstance(response, dlp.JobTrigger)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.status == dlp.JobTrigger.Status.HEALTHY

def test_create_job_trigger_rest_required_fields(request_type=dlp.CreateJobTriggerRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_job_trigger._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_job_trigger._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.JobTrigger()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.JobTrigger.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_job_trigger(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_job_trigger_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_job_trigger._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'jobTrigger'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_job_trigger_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_create_job_trigger') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_create_job_trigger') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.CreateJobTriggerRequest.pb(dlp.CreateJobTriggerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.JobTrigger.to_json(dlp.JobTrigger())
        request = dlp.CreateJobTriggerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.JobTrigger()
        client.create_job_trigger(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_job_trigger_rest_bad_request(transport: str='rest', request_type=dlp.CreateJobTriggerRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_job_trigger(request)

def test_create_job_trigger_rest_flattened():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.JobTrigger()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value', job_trigger=dlp.JobTrigger(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.JobTrigger.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_job_trigger(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*}/jobTriggers' % client.transport._host, args[1])

def test_create_job_trigger_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_job_trigger(dlp.CreateJobTriggerRequest(), parent='parent_value', job_trigger=dlp.JobTrigger(name='name_value'))

def test_create_job_trigger_rest_error():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.UpdateJobTriggerRequest, dict])
def test_update_job_trigger_rest(request_type):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/jobTriggers/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.JobTrigger(name='name_value', display_name='display_name_value', description='description_value', status=dlp.JobTrigger.Status.HEALTHY)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.JobTrigger.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_job_trigger(request)
    assert isinstance(response, dlp.JobTrigger)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.status == dlp.JobTrigger.Status.HEALTHY

def test_update_job_trigger_rest_required_fields(request_type=dlp.UpdateJobTriggerRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_job_trigger._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_job_trigger._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.JobTrigger()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.JobTrigger.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_job_trigger(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_job_trigger_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_job_trigger._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_job_trigger_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_update_job_trigger') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_update_job_trigger') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.UpdateJobTriggerRequest.pb(dlp.UpdateJobTriggerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.JobTrigger.to_json(dlp.JobTrigger())
        request = dlp.UpdateJobTriggerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.JobTrigger()
        client.update_job_trigger(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_job_trigger_rest_bad_request(transport: str='rest', request_type=dlp.UpdateJobTriggerRequest):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/jobTriggers/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_job_trigger(request)

def test_update_job_trigger_rest_flattened():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.JobTrigger()
        sample_request = {'name': 'projects/sample1/jobTriggers/sample2'}
        mock_args = dict(name='name_value', job_trigger=dlp.JobTrigger(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.JobTrigger.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_job_trigger(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/jobTriggers/*}' % client.transport._host, args[1])

def test_update_job_trigger_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_job_trigger(dlp.UpdateJobTriggerRequest(), name='name_value', job_trigger=dlp.JobTrigger(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_job_trigger_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.HybridInspectJobTriggerRequest, dict])
def test_hybrid_inspect_job_trigger_rest(request_type):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/jobTriggers/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.HybridInspectResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.HybridInspectResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.hybrid_inspect_job_trigger(request)
    assert isinstance(response, dlp.HybridInspectResponse)

def test_hybrid_inspect_job_trigger_rest_required_fields(request_type=dlp.HybridInspectJobTriggerRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).hybrid_inspect_job_trigger._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).hybrid_inspect_job_trigger._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.HybridInspectResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.HybridInspectResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.hybrid_inspect_job_trigger(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_hybrid_inspect_job_trigger_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.hybrid_inspect_job_trigger._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_hybrid_inspect_job_trigger_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_hybrid_inspect_job_trigger') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_hybrid_inspect_job_trigger') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.HybridInspectJobTriggerRequest.pb(dlp.HybridInspectJobTriggerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.HybridInspectResponse.to_json(dlp.HybridInspectResponse())
        request = dlp.HybridInspectJobTriggerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.HybridInspectResponse()
        client.hybrid_inspect_job_trigger(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_hybrid_inspect_job_trigger_rest_bad_request(transport: str='rest', request_type=dlp.HybridInspectJobTriggerRequest):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/jobTriggers/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.hybrid_inspect_job_trigger(request)

def test_hybrid_inspect_job_trigger_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.HybridInspectResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/jobTriggers/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.HybridInspectResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.hybrid_inspect_job_trigger(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/jobTriggers/*}:hybridInspect' % client.transport._host, args[1])

def test_hybrid_inspect_job_trigger_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.hybrid_inspect_job_trigger(dlp.HybridInspectJobTriggerRequest(), name='name_value')

def test_hybrid_inspect_job_trigger_rest_error():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.GetJobTriggerRequest, dict])
def test_get_job_trigger_rest(request_type):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/jobTriggers/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.JobTrigger(name='name_value', display_name='display_name_value', description='description_value', status=dlp.JobTrigger.Status.HEALTHY)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.JobTrigger.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_job_trigger(request)
    assert isinstance(response, dlp.JobTrigger)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.status == dlp.JobTrigger.Status.HEALTHY

def test_get_job_trigger_rest_required_fields(request_type=dlp.GetJobTriggerRequest):
    if False:
        return 10
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_job_trigger._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_job_trigger._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.JobTrigger()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.JobTrigger.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_job_trigger(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_job_trigger_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_job_trigger._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_job_trigger_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_get_job_trigger') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_get_job_trigger') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.GetJobTriggerRequest.pb(dlp.GetJobTriggerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.JobTrigger.to_json(dlp.JobTrigger())
        request = dlp.GetJobTriggerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.JobTrigger()
        client.get_job_trigger(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_job_trigger_rest_bad_request(transport: str='rest', request_type=dlp.GetJobTriggerRequest):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/jobTriggers/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_job_trigger(request)

def test_get_job_trigger_rest_flattened():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.JobTrigger()
        sample_request = {'name': 'projects/sample1/jobTriggers/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.JobTrigger.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_job_trigger(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/jobTriggers/*}' % client.transport._host, args[1])

def test_get_job_trigger_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_job_trigger(dlp.GetJobTriggerRequest(), name='name_value')

def test_get_job_trigger_rest_error():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.ListJobTriggersRequest, dict])
def test_list_job_triggers_rest(request_type):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.ListJobTriggersResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.ListJobTriggersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_job_triggers(request)
    assert isinstance(response, pagers.ListJobTriggersPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_job_triggers_rest_required_fields(request_type=dlp.ListJobTriggersRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_job_triggers._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_job_triggers._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'location_id', 'order_by', 'page_size', 'page_token', 'type_'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.ListJobTriggersResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.ListJobTriggersResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_job_triggers(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_job_triggers_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_job_triggers._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'locationId', 'orderBy', 'pageSize', 'pageToken', 'type')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_job_triggers_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_list_job_triggers') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_list_job_triggers') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.ListJobTriggersRequest.pb(dlp.ListJobTriggersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.ListJobTriggersResponse.to_json(dlp.ListJobTriggersResponse())
        request = dlp.ListJobTriggersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.ListJobTriggersResponse()
        client.list_job_triggers(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_job_triggers_rest_bad_request(transport: str='rest', request_type=dlp.ListJobTriggersRequest):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_job_triggers(request)

def test_list_job_triggers_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.ListJobTriggersResponse()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.ListJobTriggersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_job_triggers(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*}/jobTriggers' % client.transport._host, args[1])

def test_list_job_triggers_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_job_triggers(dlp.ListJobTriggersRequest(), parent='parent_value')

def test_list_job_triggers_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (dlp.ListJobTriggersResponse(job_triggers=[dlp.JobTrigger(), dlp.JobTrigger(), dlp.JobTrigger()], next_page_token='abc'), dlp.ListJobTriggersResponse(job_triggers=[], next_page_token='def'), dlp.ListJobTriggersResponse(job_triggers=[dlp.JobTrigger()], next_page_token='ghi'), dlp.ListJobTriggersResponse(job_triggers=[dlp.JobTrigger(), dlp.JobTrigger()]))
        response = response + response
        response = tuple((dlp.ListJobTriggersResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1'}
        pager = client.list_job_triggers(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, dlp.JobTrigger) for i in results))
        pages = list(client.list_job_triggers(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [dlp.DeleteJobTriggerRequest, dict])
def test_delete_job_trigger_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/jobTriggers/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_job_trigger(request)
    assert response is None

def test_delete_job_trigger_rest_required_fields(request_type=dlp.DeleteJobTriggerRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_job_trigger._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_job_trigger._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_job_trigger(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_job_trigger_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_job_trigger._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_job_trigger_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_delete_job_trigger') as pre:
        pre.assert_not_called()
        pb_message = dlp.DeleteJobTriggerRequest.pb(dlp.DeleteJobTriggerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = dlp.DeleteJobTriggerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_job_trigger(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_job_trigger_rest_bad_request(transport: str='rest', request_type=dlp.DeleteJobTriggerRequest):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/jobTriggers/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_job_trigger(request)

def test_delete_job_trigger_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/jobTriggers/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_job_trigger(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/jobTriggers/*}' % client.transport._host, args[1])

def test_delete_job_trigger_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_job_trigger(dlp.DeleteJobTriggerRequest(), name='name_value')

def test_delete_job_trigger_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.ActivateJobTriggerRequest, dict])
def test_activate_job_trigger_rest(request_type):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/jobTriggers/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.DlpJob(name='name_value', type_=dlp.DlpJobType.INSPECT_JOB, state=dlp.DlpJob.JobState.PENDING, job_trigger_name='job_trigger_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.DlpJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.activate_job_trigger(request)
    assert isinstance(response, dlp.DlpJob)
    assert response.name == 'name_value'
    assert response.type_ == dlp.DlpJobType.INSPECT_JOB
    assert response.state == dlp.DlpJob.JobState.PENDING
    assert response.job_trigger_name == 'job_trigger_name_value'

def test_activate_job_trigger_rest_required_fields(request_type=dlp.ActivateJobTriggerRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).activate_job_trigger._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).activate_job_trigger._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.DlpJob()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.DlpJob.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.activate_job_trigger(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_activate_job_trigger_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.activate_job_trigger._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_activate_job_trigger_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_activate_job_trigger') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_activate_job_trigger') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.ActivateJobTriggerRequest.pb(dlp.ActivateJobTriggerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.DlpJob.to_json(dlp.DlpJob())
        request = dlp.ActivateJobTriggerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.DlpJob()
        client.activate_job_trigger(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_activate_job_trigger_rest_bad_request(transport: str='rest', request_type=dlp.ActivateJobTriggerRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/jobTriggers/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.activate_job_trigger(request)

def test_activate_job_trigger_rest_error():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.CreateDiscoveryConfigRequest, dict])
def test_create_discovery_config_rest(request_type):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.DiscoveryConfig(name='name_value', display_name='display_name_value', inspect_templates=['inspect_templates_value'], status=dlp.DiscoveryConfig.Status.RUNNING)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.DiscoveryConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_discovery_config(request)
    assert isinstance(response, dlp.DiscoveryConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.inspect_templates == ['inspect_templates_value']
    assert response.status == dlp.DiscoveryConfig.Status.RUNNING

def test_create_discovery_config_rest_required_fields(request_type=dlp.CreateDiscoveryConfigRequest):
    if False:
        print('Hello World!')
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_discovery_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_discovery_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.DiscoveryConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.DiscoveryConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_discovery_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_discovery_config_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_discovery_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'discoveryConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_discovery_config_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_create_discovery_config') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_create_discovery_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.CreateDiscoveryConfigRequest.pb(dlp.CreateDiscoveryConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.DiscoveryConfig.to_json(dlp.DiscoveryConfig())
        request = dlp.CreateDiscoveryConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.DiscoveryConfig()
        client.create_discovery_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_discovery_config_rest_bad_request(transport: str='rest', request_type=dlp.CreateDiscoveryConfigRequest):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_discovery_config(request)

def test_create_discovery_config_rest_flattened():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.DiscoveryConfig()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', discovery_config=dlp.DiscoveryConfig(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.DiscoveryConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_discovery_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/discoveryConfigs' % client.transport._host, args[1])

def test_create_discovery_config_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_discovery_config(dlp.CreateDiscoveryConfigRequest(), parent='parent_value', discovery_config=dlp.DiscoveryConfig(name='name_value'))

def test_create_discovery_config_rest_error():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.UpdateDiscoveryConfigRequest, dict])
def test_update_discovery_config_rest(request_type):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/discoveryConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.DiscoveryConfig(name='name_value', display_name='display_name_value', inspect_templates=['inspect_templates_value'], status=dlp.DiscoveryConfig.Status.RUNNING)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.DiscoveryConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_discovery_config(request)
    assert isinstance(response, dlp.DiscoveryConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.inspect_templates == ['inspect_templates_value']
    assert response.status == dlp.DiscoveryConfig.Status.RUNNING

def test_update_discovery_config_rest_required_fields(request_type=dlp.UpdateDiscoveryConfigRequest):
    if False:
        print('Hello World!')
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_discovery_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_discovery_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.DiscoveryConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.DiscoveryConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_discovery_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_discovery_config_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_discovery_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name', 'discoveryConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_discovery_config_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_update_discovery_config') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_update_discovery_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.UpdateDiscoveryConfigRequest.pb(dlp.UpdateDiscoveryConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.DiscoveryConfig.to_json(dlp.DiscoveryConfig())
        request = dlp.UpdateDiscoveryConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.DiscoveryConfig()
        client.update_discovery_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_discovery_config_rest_bad_request(transport: str='rest', request_type=dlp.UpdateDiscoveryConfigRequest):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/discoveryConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_discovery_config(request)

def test_update_discovery_config_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.DiscoveryConfig()
        sample_request = {'name': 'projects/sample1/locations/sample2/discoveryConfigs/sample3'}
        mock_args = dict(name='name_value', discovery_config=dlp.DiscoveryConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.DiscoveryConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_discovery_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/discoveryConfigs/*}' % client.transport._host, args[1])

def test_update_discovery_config_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_discovery_config(dlp.UpdateDiscoveryConfigRequest(), name='name_value', discovery_config=dlp.DiscoveryConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_discovery_config_rest_error():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.GetDiscoveryConfigRequest, dict])
def test_get_discovery_config_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/discoveryConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.DiscoveryConfig(name='name_value', display_name='display_name_value', inspect_templates=['inspect_templates_value'], status=dlp.DiscoveryConfig.Status.RUNNING)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.DiscoveryConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_discovery_config(request)
    assert isinstance(response, dlp.DiscoveryConfig)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.inspect_templates == ['inspect_templates_value']
    assert response.status == dlp.DiscoveryConfig.Status.RUNNING

def test_get_discovery_config_rest_required_fields(request_type=dlp.GetDiscoveryConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_discovery_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_discovery_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.DiscoveryConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.DiscoveryConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_discovery_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_discovery_config_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_discovery_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_discovery_config_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_get_discovery_config') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_get_discovery_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.GetDiscoveryConfigRequest.pb(dlp.GetDiscoveryConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.DiscoveryConfig.to_json(dlp.DiscoveryConfig())
        request = dlp.GetDiscoveryConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.DiscoveryConfig()
        client.get_discovery_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_discovery_config_rest_bad_request(transport: str='rest', request_type=dlp.GetDiscoveryConfigRequest):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/discoveryConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_discovery_config(request)

def test_get_discovery_config_rest_flattened():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.DiscoveryConfig()
        sample_request = {'name': 'projects/sample1/locations/sample2/discoveryConfigs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.DiscoveryConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_discovery_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/discoveryConfigs/*}' % client.transport._host, args[1])

def test_get_discovery_config_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_discovery_config(dlp.GetDiscoveryConfigRequest(), name='name_value')

def test_get_discovery_config_rest_error():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.ListDiscoveryConfigsRequest, dict])
def test_list_discovery_configs_rest(request_type):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.ListDiscoveryConfigsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.ListDiscoveryConfigsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_discovery_configs(request)
    assert isinstance(response, pagers.ListDiscoveryConfigsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_discovery_configs_rest_required_fields(request_type=dlp.ListDiscoveryConfigsRequest):
    if False:
        return 10
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_discovery_configs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_discovery_configs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.ListDiscoveryConfigsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.ListDiscoveryConfigsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_discovery_configs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_discovery_configs_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_discovery_configs._get_unset_required_fields({})
    assert set(unset_fields) == set(('orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_discovery_configs_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_list_discovery_configs') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_list_discovery_configs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.ListDiscoveryConfigsRequest.pb(dlp.ListDiscoveryConfigsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.ListDiscoveryConfigsResponse.to_json(dlp.ListDiscoveryConfigsResponse())
        request = dlp.ListDiscoveryConfigsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.ListDiscoveryConfigsResponse()
        client.list_discovery_configs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_discovery_configs_rest_bad_request(transport: str='rest', request_type=dlp.ListDiscoveryConfigsRequest):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_discovery_configs(request)

def test_list_discovery_configs_rest_flattened():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.ListDiscoveryConfigsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.ListDiscoveryConfigsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_discovery_configs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*}/discoveryConfigs' % client.transport._host, args[1])

def test_list_discovery_configs_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_discovery_configs(dlp.ListDiscoveryConfigsRequest(), parent='parent_value')

def test_list_discovery_configs_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (dlp.ListDiscoveryConfigsResponse(discovery_configs=[dlp.DiscoveryConfig(), dlp.DiscoveryConfig(), dlp.DiscoveryConfig()], next_page_token='abc'), dlp.ListDiscoveryConfigsResponse(discovery_configs=[], next_page_token='def'), dlp.ListDiscoveryConfigsResponse(discovery_configs=[dlp.DiscoveryConfig()], next_page_token='ghi'), dlp.ListDiscoveryConfigsResponse(discovery_configs=[dlp.DiscoveryConfig(), dlp.DiscoveryConfig()]))
        response = response + response
        response = tuple((dlp.ListDiscoveryConfigsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_discovery_configs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, dlp.DiscoveryConfig) for i in results))
        pages = list(client.list_discovery_configs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [dlp.DeleteDiscoveryConfigRequest, dict])
def test_delete_discovery_config_rest(request_type):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/discoveryConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_discovery_config(request)
    assert response is None

def test_delete_discovery_config_rest_required_fields(request_type=dlp.DeleteDiscoveryConfigRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_discovery_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_discovery_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_discovery_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_discovery_config_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_discovery_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_discovery_config_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_delete_discovery_config') as pre:
        pre.assert_not_called()
        pb_message = dlp.DeleteDiscoveryConfigRequest.pb(dlp.DeleteDiscoveryConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = dlp.DeleteDiscoveryConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_discovery_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_discovery_config_rest_bad_request(transport: str='rest', request_type=dlp.DeleteDiscoveryConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/discoveryConfigs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_discovery_config(request)

def test_delete_discovery_config_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/discoveryConfigs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_discovery_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/discoveryConfigs/*}' % client.transport._host, args[1])

def test_delete_discovery_config_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_discovery_config(dlp.DeleteDiscoveryConfigRequest(), name='name_value')

def test_delete_discovery_config_rest_error():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.CreateDlpJobRequest, dict])
def test_create_dlp_job_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.DlpJob(name='name_value', type_=dlp.DlpJobType.INSPECT_JOB, state=dlp.DlpJob.JobState.PENDING, job_trigger_name='job_trigger_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.DlpJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_dlp_job(request)
    assert isinstance(response, dlp.DlpJob)
    assert response.name == 'name_value'
    assert response.type_ == dlp.DlpJobType.INSPECT_JOB
    assert response.state == dlp.DlpJob.JobState.PENDING
    assert response.job_trigger_name == 'job_trigger_name_value'

def test_create_dlp_job_rest_required_fields(request_type=dlp.CreateDlpJobRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_dlp_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_dlp_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.DlpJob()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.DlpJob.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_dlp_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_dlp_job_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_dlp_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_dlp_job_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_create_dlp_job') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_create_dlp_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.CreateDlpJobRequest.pb(dlp.CreateDlpJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.DlpJob.to_json(dlp.DlpJob())
        request = dlp.CreateDlpJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.DlpJob()
        client.create_dlp_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_dlp_job_rest_bad_request(transport: str='rest', request_type=dlp.CreateDlpJobRequest):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_dlp_job(request)

def test_create_dlp_job_rest_flattened():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.DlpJob()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.DlpJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_dlp_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*}/dlpJobs' % client.transport._host, args[1])

def test_create_dlp_job_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_dlp_job(dlp.CreateDlpJobRequest(), parent='parent_value', inspect_job=dlp.InspectJobConfig(storage_config=storage.StorageConfig(datastore_options=storage.DatastoreOptions(partition_id=storage.PartitionId(project_id='project_id_value')))), risk_job=dlp.RiskAnalysisJobConfig(privacy_metric=dlp.PrivacyMetric(numerical_stats_config=dlp.PrivacyMetric.NumericalStatsConfig(field=storage.FieldId(name='name_value')))))

def test_create_dlp_job_rest_error():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.ListDlpJobsRequest, dict])
def test_list_dlp_jobs_rest(request_type):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.ListDlpJobsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.ListDlpJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_dlp_jobs(request)
    assert isinstance(response, pagers.ListDlpJobsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_dlp_jobs_rest_required_fields(request_type=dlp.ListDlpJobsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_dlp_jobs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_dlp_jobs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'location_id', 'order_by', 'page_size', 'page_token', 'type_'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.ListDlpJobsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.ListDlpJobsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_dlp_jobs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_dlp_jobs_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_dlp_jobs._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'locationId', 'orderBy', 'pageSize', 'pageToken', 'type')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_dlp_jobs_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_list_dlp_jobs') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_list_dlp_jobs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.ListDlpJobsRequest.pb(dlp.ListDlpJobsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.ListDlpJobsResponse.to_json(dlp.ListDlpJobsResponse())
        request = dlp.ListDlpJobsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.ListDlpJobsResponse()
        client.list_dlp_jobs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_dlp_jobs_rest_bad_request(transport: str='rest', request_type=dlp.ListDlpJobsRequest):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_dlp_jobs(request)

def test_list_dlp_jobs_rest_flattened():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.ListDlpJobsResponse()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.ListDlpJobsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_dlp_jobs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*}/dlpJobs' % client.transport._host, args[1])

def test_list_dlp_jobs_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_dlp_jobs(dlp.ListDlpJobsRequest(), parent='parent_value')

def test_list_dlp_jobs_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (dlp.ListDlpJobsResponse(jobs=[dlp.DlpJob(), dlp.DlpJob(), dlp.DlpJob()], next_page_token='abc'), dlp.ListDlpJobsResponse(jobs=[], next_page_token='def'), dlp.ListDlpJobsResponse(jobs=[dlp.DlpJob()], next_page_token='ghi'), dlp.ListDlpJobsResponse(jobs=[dlp.DlpJob(), dlp.DlpJob()]))
        response = response + response
        response = tuple((dlp.ListDlpJobsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1'}
        pager = client.list_dlp_jobs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, dlp.DlpJob) for i in results))
        pages = list(client.list_dlp_jobs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [dlp.GetDlpJobRequest, dict])
def test_get_dlp_job_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/dlpJobs/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.DlpJob(name='name_value', type_=dlp.DlpJobType.INSPECT_JOB, state=dlp.DlpJob.JobState.PENDING, job_trigger_name='job_trigger_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.DlpJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_dlp_job(request)
    assert isinstance(response, dlp.DlpJob)
    assert response.name == 'name_value'
    assert response.type_ == dlp.DlpJobType.INSPECT_JOB
    assert response.state == dlp.DlpJob.JobState.PENDING
    assert response.job_trigger_name == 'job_trigger_name_value'

def test_get_dlp_job_rest_required_fields(request_type=dlp.GetDlpJobRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_dlp_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_dlp_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.DlpJob()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.DlpJob.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_dlp_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_dlp_job_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_dlp_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_dlp_job_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_get_dlp_job') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_get_dlp_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.GetDlpJobRequest.pb(dlp.GetDlpJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.DlpJob.to_json(dlp.DlpJob())
        request = dlp.GetDlpJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.DlpJob()
        client.get_dlp_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_dlp_job_rest_bad_request(transport: str='rest', request_type=dlp.GetDlpJobRequest):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/dlpJobs/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_dlp_job(request)

def test_get_dlp_job_rest_flattened():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.DlpJob()
        sample_request = {'name': 'projects/sample1/dlpJobs/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.DlpJob.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_dlp_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/dlpJobs/*}' % client.transport._host, args[1])

def test_get_dlp_job_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_dlp_job(dlp.GetDlpJobRequest(), name='name_value')

def test_get_dlp_job_rest_error():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.DeleteDlpJobRequest, dict])
def test_delete_dlp_job_rest(request_type):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/dlpJobs/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_dlp_job(request)
    assert response is None

def test_delete_dlp_job_rest_required_fields(request_type=dlp.DeleteDlpJobRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_dlp_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_dlp_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_dlp_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_dlp_job_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_dlp_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_dlp_job_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_delete_dlp_job') as pre:
        pre.assert_not_called()
        pb_message = dlp.DeleteDlpJobRequest.pb(dlp.DeleteDlpJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = dlp.DeleteDlpJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_dlp_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_dlp_job_rest_bad_request(transport: str='rest', request_type=dlp.DeleteDlpJobRequest):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/dlpJobs/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_dlp_job(request)

def test_delete_dlp_job_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/dlpJobs/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_dlp_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/dlpJobs/*}' % client.transport._host, args[1])

def test_delete_dlp_job_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_dlp_job(dlp.DeleteDlpJobRequest(), name='name_value')

def test_delete_dlp_job_rest_error():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.CancelDlpJobRequest, dict])
def test_cancel_dlp_job_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/dlpJobs/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.cancel_dlp_job(request)
    assert response is None

def test_cancel_dlp_job_rest_required_fields(request_type=dlp.CancelDlpJobRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).cancel_dlp_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).cancel_dlp_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.cancel_dlp_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_cancel_dlp_job_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.cancel_dlp_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_cancel_dlp_job_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_cancel_dlp_job') as pre:
        pre.assert_not_called()
        pb_message = dlp.CancelDlpJobRequest.pb(dlp.CancelDlpJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = dlp.CancelDlpJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.cancel_dlp_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_cancel_dlp_job_rest_bad_request(transport: str='rest', request_type=dlp.CancelDlpJobRequest):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/dlpJobs/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.cancel_dlp_job(request)

def test_cancel_dlp_job_rest_error():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.CreateStoredInfoTypeRequest, dict])
def test_create_stored_info_type_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'organizations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.StoredInfoType(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.StoredInfoType.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_stored_info_type(request)
    assert isinstance(response, dlp.StoredInfoType)
    assert response.name == 'name_value'

def test_create_stored_info_type_rest_required_fields(request_type=dlp.CreateStoredInfoTypeRequest):
    if False:
        return 10
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_stored_info_type._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_stored_info_type._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.StoredInfoType()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.StoredInfoType.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_stored_info_type(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_stored_info_type_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_stored_info_type._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'config'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_stored_info_type_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_create_stored_info_type') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_create_stored_info_type') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.CreateStoredInfoTypeRequest.pb(dlp.CreateStoredInfoTypeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.StoredInfoType.to_json(dlp.StoredInfoType())
        request = dlp.CreateStoredInfoTypeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.StoredInfoType()
        client.create_stored_info_type(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_stored_info_type_rest_bad_request(transport: str='rest', request_type=dlp.CreateStoredInfoTypeRequest):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'organizations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_stored_info_type(request)

def test_create_stored_info_type_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.StoredInfoType()
        sample_request = {'parent': 'organizations/sample1'}
        mock_args = dict(parent='parent_value', config=dlp.StoredInfoTypeConfig(display_name='display_name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.StoredInfoType.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_stored_info_type(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=organizations/*}/storedInfoTypes' % client.transport._host, args[1])

def test_create_stored_info_type_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_stored_info_type(dlp.CreateStoredInfoTypeRequest(), parent='parent_value', config=dlp.StoredInfoTypeConfig(display_name='display_name_value'))

def test_create_stored_info_type_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.UpdateStoredInfoTypeRequest, dict])
def test_update_stored_info_type_rest(request_type):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'organizations/sample1/storedInfoTypes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.StoredInfoType(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.StoredInfoType.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_stored_info_type(request)
    assert isinstance(response, dlp.StoredInfoType)
    assert response.name == 'name_value'

def test_update_stored_info_type_rest_required_fields(request_type=dlp.UpdateStoredInfoTypeRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_stored_info_type._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_stored_info_type._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.StoredInfoType()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.StoredInfoType.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_stored_info_type(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_stored_info_type_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_stored_info_type._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_stored_info_type_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_update_stored_info_type') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_update_stored_info_type') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.UpdateStoredInfoTypeRequest.pb(dlp.UpdateStoredInfoTypeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.StoredInfoType.to_json(dlp.StoredInfoType())
        request = dlp.UpdateStoredInfoTypeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.StoredInfoType()
        client.update_stored_info_type(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_stored_info_type_rest_bad_request(transport: str='rest', request_type=dlp.UpdateStoredInfoTypeRequest):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'organizations/sample1/storedInfoTypes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_stored_info_type(request)

def test_update_stored_info_type_rest_flattened():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.StoredInfoType()
        sample_request = {'name': 'organizations/sample1/storedInfoTypes/sample2'}
        mock_args = dict(name='name_value', config=dlp.StoredInfoTypeConfig(display_name='display_name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.StoredInfoType.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_stored_info_type(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=organizations/*/storedInfoTypes/*}' % client.transport._host, args[1])

def test_update_stored_info_type_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_stored_info_type(dlp.UpdateStoredInfoTypeRequest(), name='name_value', config=dlp.StoredInfoTypeConfig(display_name='display_name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_stored_info_type_rest_error():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.GetStoredInfoTypeRequest, dict])
def test_get_stored_info_type_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'organizations/sample1/storedInfoTypes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.StoredInfoType(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.StoredInfoType.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_stored_info_type(request)
    assert isinstance(response, dlp.StoredInfoType)
    assert response.name == 'name_value'

def test_get_stored_info_type_rest_required_fields(request_type=dlp.GetStoredInfoTypeRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_stored_info_type._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_stored_info_type._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.StoredInfoType()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.StoredInfoType.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_stored_info_type(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_stored_info_type_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_stored_info_type._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_stored_info_type_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_get_stored_info_type') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_get_stored_info_type') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.GetStoredInfoTypeRequest.pb(dlp.GetStoredInfoTypeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.StoredInfoType.to_json(dlp.StoredInfoType())
        request = dlp.GetStoredInfoTypeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.StoredInfoType()
        client.get_stored_info_type(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_stored_info_type_rest_bad_request(transport: str='rest', request_type=dlp.GetStoredInfoTypeRequest):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'organizations/sample1/storedInfoTypes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_stored_info_type(request)

def test_get_stored_info_type_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.StoredInfoType()
        sample_request = {'name': 'organizations/sample1/storedInfoTypes/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.StoredInfoType.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_stored_info_type(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=organizations/*/storedInfoTypes/*}' % client.transport._host, args[1])

def test_get_stored_info_type_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_stored_info_type(dlp.GetStoredInfoTypeRequest(), name='name_value')

def test_get_stored_info_type_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.ListStoredInfoTypesRequest, dict])
def test_list_stored_info_types_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'organizations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.ListStoredInfoTypesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.ListStoredInfoTypesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_stored_info_types(request)
    assert isinstance(response, pagers.ListStoredInfoTypesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_stored_info_types_rest_required_fields(request_type=dlp.ListStoredInfoTypesRequest):
    if False:
        return 10
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_stored_info_types._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_stored_info_types._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('location_id', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.ListStoredInfoTypesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.ListStoredInfoTypesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_stored_info_types(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_stored_info_types_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_stored_info_types._get_unset_required_fields({})
    assert set(unset_fields) == set(('locationId', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_stored_info_types_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_list_stored_info_types') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_list_stored_info_types') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.ListStoredInfoTypesRequest.pb(dlp.ListStoredInfoTypesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.ListStoredInfoTypesResponse.to_json(dlp.ListStoredInfoTypesResponse())
        request = dlp.ListStoredInfoTypesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.ListStoredInfoTypesResponse()
        client.list_stored_info_types(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_stored_info_types_rest_bad_request(transport: str='rest', request_type=dlp.ListStoredInfoTypesRequest):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'organizations/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_stored_info_types(request)

def test_list_stored_info_types_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.ListStoredInfoTypesResponse()
        sample_request = {'parent': 'organizations/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.ListStoredInfoTypesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_stored_info_types(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=organizations/*}/storedInfoTypes' % client.transport._host, args[1])

def test_list_stored_info_types_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_stored_info_types(dlp.ListStoredInfoTypesRequest(), parent='parent_value')

def test_list_stored_info_types_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (dlp.ListStoredInfoTypesResponse(stored_info_types=[dlp.StoredInfoType(), dlp.StoredInfoType(), dlp.StoredInfoType()], next_page_token='abc'), dlp.ListStoredInfoTypesResponse(stored_info_types=[], next_page_token='def'), dlp.ListStoredInfoTypesResponse(stored_info_types=[dlp.StoredInfoType()], next_page_token='ghi'), dlp.ListStoredInfoTypesResponse(stored_info_types=[dlp.StoredInfoType(), dlp.StoredInfoType()]))
        response = response + response
        response = tuple((dlp.ListStoredInfoTypesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'organizations/sample1'}
        pager = client.list_stored_info_types(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, dlp.StoredInfoType) for i in results))
        pages = list(client.list_stored_info_types(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [dlp.DeleteStoredInfoTypeRequest, dict])
def test_delete_stored_info_type_rest(request_type):
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'organizations/sample1/storedInfoTypes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_stored_info_type(request)
    assert response is None

def test_delete_stored_info_type_rest_required_fields(request_type=dlp.DeleteStoredInfoTypeRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_stored_info_type._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_stored_info_type._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_stored_info_type(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_stored_info_type_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_stored_info_type._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_stored_info_type_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_delete_stored_info_type') as pre:
        pre.assert_not_called()
        pb_message = dlp.DeleteStoredInfoTypeRequest.pb(dlp.DeleteStoredInfoTypeRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = dlp.DeleteStoredInfoTypeRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_stored_info_type(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_stored_info_type_rest_bad_request(transport: str='rest', request_type=dlp.DeleteStoredInfoTypeRequest):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'organizations/sample1/storedInfoTypes/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_stored_info_type(request)

def test_delete_stored_info_type_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'organizations/sample1/storedInfoTypes/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_stored_info_type(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=organizations/*/storedInfoTypes/*}' % client.transport._host, args[1])

def test_delete_stored_info_type_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_stored_info_type(dlp.DeleteStoredInfoTypeRequest(), name='name_value')

def test_delete_stored_info_type_rest_error():
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.HybridInspectDlpJobRequest, dict])
def test_hybrid_inspect_dlp_job_rest(request_type):
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/dlpJobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.HybridInspectResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.HybridInspectResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.hybrid_inspect_dlp_job(request)
    assert isinstance(response, dlp.HybridInspectResponse)

def test_hybrid_inspect_dlp_job_rest_required_fields(request_type=dlp.HybridInspectDlpJobRequest):
    if False:
        return 10
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).hybrid_inspect_dlp_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).hybrid_inspect_dlp_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = dlp.HybridInspectResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = dlp.HybridInspectResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.hybrid_inspect_dlp_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_hybrid_inspect_dlp_job_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.hybrid_inspect_dlp_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_hybrid_inspect_dlp_job_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'post_hybrid_inspect_dlp_job') as post, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_hybrid_inspect_dlp_job') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = dlp.HybridInspectDlpJobRequest.pb(dlp.HybridInspectDlpJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = dlp.HybridInspectResponse.to_json(dlp.HybridInspectResponse())
        request = dlp.HybridInspectDlpJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = dlp.HybridInspectResponse()
        client.hybrid_inspect_dlp_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_hybrid_inspect_dlp_job_rest_bad_request(transport: str='rest', request_type=dlp.HybridInspectDlpJobRequest):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/dlpJobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.hybrid_inspect_dlp_job(request)

def test_hybrid_inspect_dlp_job_rest_flattened():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = dlp.HybridInspectResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/dlpJobs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = dlp.HybridInspectResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.hybrid_inspect_dlp_job(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/dlpJobs/*}:hybridInspect' % client.transport._host, args[1])

def test_hybrid_inspect_dlp_job_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.hybrid_inspect_dlp_job(dlp.HybridInspectDlpJobRequest(), name='name_value')

def test_hybrid_inspect_dlp_job_rest_error():
    if False:
        return 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [dlp.FinishDlpJobRequest, dict])
def test_finish_dlp_job_rest(request_type):
    if False:
        print('Hello World!')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/dlpJobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.finish_dlp_job(request)
    assert response is None

def test_finish_dlp_job_rest_required_fields(request_type=dlp.FinishDlpJobRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.DlpServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).finish_dlp_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).finish_dlp_job._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.finish_dlp_job(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_finish_dlp_job_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.finish_dlp_job._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_finish_dlp_job_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.DlpServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.DlpServiceRestInterceptor())
    client = DlpServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.DlpServiceRestInterceptor, 'pre_finish_dlp_job') as pre:
        pre.assert_not_called()
        pb_message = dlp.FinishDlpJobRequest.pb(dlp.FinishDlpJobRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = dlp.FinishDlpJobRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.finish_dlp_job(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_finish_dlp_job_rest_bad_request(transport: str='rest', request_type=dlp.FinishDlpJobRequest):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/dlpJobs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.finish_dlp_job(request)

def test_finish_dlp_job_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.DlpServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.DlpServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DlpServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.DlpServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = DlpServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = DlpServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.DlpServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = DlpServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.DlpServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = DlpServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        print('Hello World!')
    transport = transports.DlpServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.DlpServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.DlpServiceGrpcTransport, transports.DlpServiceGrpcAsyncIOTransport, transports.DlpServiceRestTransport])
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
        print('Hello World!')
    transport = DlpServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.DlpServiceGrpcTransport)

def test_dlp_service_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.DlpServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_dlp_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.dlp_v2.services.dlp_service.transports.DlpServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.DlpServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('inspect_content', 'redact_image', 'deidentify_content', 'reidentify_content', 'list_info_types', 'create_inspect_template', 'update_inspect_template', 'get_inspect_template', 'list_inspect_templates', 'delete_inspect_template', 'create_deidentify_template', 'update_deidentify_template', 'get_deidentify_template', 'list_deidentify_templates', 'delete_deidentify_template', 'create_job_trigger', 'update_job_trigger', 'hybrid_inspect_job_trigger', 'get_job_trigger', 'list_job_triggers', 'delete_job_trigger', 'activate_job_trigger', 'create_discovery_config', 'update_discovery_config', 'get_discovery_config', 'list_discovery_configs', 'delete_discovery_config', 'create_dlp_job', 'list_dlp_jobs', 'get_dlp_job', 'delete_dlp_job', 'cancel_dlp_job', 'create_stored_info_type', 'update_stored_info_type', 'get_stored_info_type', 'list_stored_info_types', 'delete_stored_info_type', 'hybrid_inspect_dlp_job', 'finish_dlp_job')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_dlp_service_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.dlp_v2.services.dlp_service.transports.DlpServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.DlpServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_dlp_service_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.dlp_v2.services.dlp_service.transports.DlpServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.DlpServiceTransport()
        adc.assert_called_once()

def test_dlp_service_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        DlpServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.DlpServiceGrpcTransport, transports.DlpServiceGrpcAsyncIOTransport])
def test_dlp_service_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.DlpServiceGrpcTransport, transports.DlpServiceGrpcAsyncIOTransport, transports.DlpServiceRestTransport])
def test_dlp_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.DlpServiceGrpcTransport, grpc_helpers), (transports.DlpServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_dlp_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('dlp.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='dlp.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.DlpServiceGrpcTransport, transports.DlpServiceGrpcAsyncIOTransport])
def test_dlp_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_dlp_service_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.DlpServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_dlp_service_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dlp.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('dlp.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dlp.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_dlp_service_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dlp.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('dlp.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dlp.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_dlp_service_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = DlpServiceClient(credentials=creds1, transport=transport_name)
    client2 = DlpServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.inspect_content._session
    session2 = client2.transport.inspect_content._session
    assert session1 != session2
    session1 = client1.transport.redact_image._session
    session2 = client2.transport.redact_image._session
    assert session1 != session2
    session1 = client1.transport.deidentify_content._session
    session2 = client2.transport.deidentify_content._session
    assert session1 != session2
    session1 = client1.transport.reidentify_content._session
    session2 = client2.transport.reidentify_content._session
    assert session1 != session2
    session1 = client1.transport.list_info_types._session
    session2 = client2.transport.list_info_types._session
    assert session1 != session2
    session1 = client1.transport.create_inspect_template._session
    session2 = client2.transport.create_inspect_template._session
    assert session1 != session2
    session1 = client1.transport.update_inspect_template._session
    session2 = client2.transport.update_inspect_template._session
    assert session1 != session2
    session1 = client1.transport.get_inspect_template._session
    session2 = client2.transport.get_inspect_template._session
    assert session1 != session2
    session1 = client1.transport.list_inspect_templates._session
    session2 = client2.transport.list_inspect_templates._session
    assert session1 != session2
    session1 = client1.transport.delete_inspect_template._session
    session2 = client2.transport.delete_inspect_template._session
    assert session1 != session2
    session1 = client1.transport.create_deidentify_template._session
    session2 = client2.transport.create_deidentify_template._session
    assert session1 != session2
    session1 = client1.transport.update_deidentify_template._session
    session2 = client2.transport.update_deidentify_template._session
    assert session1 != session2
    session1 = client1.transport.get_deidentify_template._session
    session2 = client2.transport.get_deidentify_template._session
    assert session1 != session2
    session1 = client1.transport.list_deidentify_templates._session
    session2 = client2.transport.list_deidentify_templates._session
    assert session1 != session2
    session1 = client1.transport.delete_deidentify_template._session
    session2 = client2.transport.delete_deidentify_template._session
    assert session1 != session2
    session1 = client1.transport.create_job_trigger._session
    session2 = client2.transport.create_job_trigger._session
    assert session1 != session2
    session1 = client1.transport.update_job_trigger._session
    session2 = client2.transport.update_job_trigger._session
    assert session1 != session2
    session1 = client1.transport.hybrid_inspect_job_trigger._session
    session2 = client2.transport.hybrid_inspect_job_trigger._session
    assert session1 != session2
    session1 = client1.transport.get_job_trigger._session
    session2 = client2.transport.get_job_trigger._session
    assert session1 != session2
    session1 = client1.transport.list_job_triggers._session
    session2 = client2.transport.list_job_triggers._session
    assert session1 != session2
    session1 = client1.transport.delete_job_trigger._session
    session2 = client2.transport.delete_job_trigger._session
    assert session1 != session2
    session1 = client1.transport.activate_job_trigger._session
    session2 = client2.transport.activate_job_trigger._session
    assert session1 != session2
    session1 = client1.transport.create_discovery_config._session
    session2 = client2.transport.create_discovery_config._session
    assert session1 != session2
    session1 = client1.transport.update_discovery_config._session
    session2 = client2.transport.update_discovery_config._session
    assert session1 != session2
    session1 = client1.transport.get_discovery_config._session
    session2 = client2.transport.get_discovery_config._session
    assert session1 != session2
    session1 = client1.transport.list_discovery_configs._session
    session2 = client2.transport.list_discovery_configs._session
    assert session1 != session2
    session1 = client1.transport.delete_discovery_config._session
    session2 = client2.transport.delete_discovery_config._session
    assert session1 != session2
    session1 = client1.transport.create_dlp_job._session
    session2 = client2.transport.create_dlp_job._session
    assert session1 != session2
    session1 = client1.transport.list_dlp_jobs._session
    session2 = client2.transport.list_dlp_jobs._session
    assert session1 != session2
    session1 = client1.transport.get_dlp_job._session
    session2 = client2.transport.get_dlp_job._session
    assert session1 != session2
    session1 = client1.transport.delete_dlp_job._session
    session2 = client2.transport.delete_dlp_job._session
    assert session1 != session2
    session1 = client1.transport.cancel_dlp_job._session
    session2 = client2.transport.cancel_dlp_job._session
    assert session1 != session2
    session1 = client1.transport.create_stored_info_type._session
    session2 = client2.transport.create_stored_info_type._session
    assert session1 != session2
    session1 = client1.transport.update_stored_info_type._session
    session2 = client2.transport.update_stored_info_type._session
    assert session1 != session2
    session1 = client1.transport.get_stored_info_type._session
    session2 = client2.transport.get_stored_info_type._session
    assert session1 != session2
    session1 = client1.transport.list_stored_info_types._session
    session2 = client2.transport.list_stored_info_types._session
    assert session1 != session2
    session1 = client1.transport.delete_stored_info_type._session
    session2 = client2.transport.delete_stored_info_type._session
    assert session1 != session2
    session1 = client1.transport.hybrid_inspect_dlp_job._session
    session2 = client2.transport.hybrid_inspect_dlp_job._session
    assert session1 != session2
    session1 = client1.transport.finish_dlp_job._session
    session2 = client2.transport.finish_dlp_job._session
    assert session1 != session2

def test_dlp_service_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.DlpServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_dlp_service_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.DlpServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.DlpServiceGrpcTransport, transports.DlpServiceGrpcAsyncIOTransport])
def test_dlp_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.DlpServiceGrpcTransport, transports.DlpServiceGrpcAsyncIOTransport])
def test_dlp_service_transport_channel_mtls_with_adc(transport_class):
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

def test_deidentify_template_path():
    if False:
        while True:
            i = 10
    organization = 'squid'
    deidentify_template = 'clam'
    expected = 'organizations/{organization}/deidentifyTemplates/{deidentify_template}'.format(organization=organization, deidentify_template=deidentify_template)
    actual = DlpServiceClient.deidentify_template_path(organization, deidentify_template)
    assert expected == actual

def test_parse_deidentify_template_path():
    if False:
        return 10
    expected = {'organization': 'whelk', 'deidentify_template': 'octopus'}
    path = DlpServiceClient.deidentify_template_path(**expected)
    actual = DlpServiceClient.parse_deidentify_template_path(path)
    assert expected == actual

def test_discovery_config_path():
    if False:
        print('Hello World!')
    project = 'oyster'
    location = 'nudibranch'
    discovery_config = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/discoveryConfigs/{discovery_config}'.format(project=project, location=location, discovery_config=discovery_config)
    actual = DlpServiceClient.discovery_config_path(project, location, discovery_config)
    assert expected == actual

def test_parse_discovery_config_path():
    if False:
        return 10
    expected = {'project': 'mussel', 'location': 'winkle', 'discovery_config': 'nautilus'}
    path = DlpServiceClient.discovery_config_path(**expected)
    actual = DlpServiceClient.parse_discovery_config_path(path)
    assert expected == actual

def test_dlp_content_path():
    if False:
        i = 10
        return i + 15
    project = 'scallop'
    expected = 'projects/{project}/dlpContent'.format(project=project)
    actual = DlpServiceClient.dlp_content_path(project)
    assert expected == actual

def test_parse_dlp_content_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'abalone'}
    path = DlpServiceClient.dlp_content_path(**expected)
    actual = DlpServiceClient.parse_dlp_content_path(path)
    assert expected == actual

def test_dlp_job_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    dlp_job = 'clam'
    expected = 'projects/{project}/dlpJobs/{dlp_job}'.format(project=project, dlp_job=dlp_job)
    actual = DlpServiceClient.dlp_job_path(project, dlp_job)
    assert expected == actual

def test_parse_dlp_job_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'whelk', 'dlp_job': 'octopus'}
    path = DlpServiceClient.dlp_job_path(**expected)
    actual = DlpServiceClient.parse_dlp_job_path(path)
    assert expected == actual

def test_finding_path():
    if False:
        while True:
            i = 10
    project = 'oyster'
    location = 'nudibranch'
    finding = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/findings/{finding}'.format(project=project, location=location, finding=finding)
    actual = DlpServiceClient.finding_path(project, location, finding)
    assert expected == actual

def test_parse_finding_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'mussel', 'location': 'winkle', 'finding': 'nautilus'}
    path = DlpServiceClient.finding_path(**expected)
    actual = DlpServiceClient.parse_finding_path(path)
    assert expected == actual

def test_inspect_template_path():
    if False:
        i = 10
        return i + 15
    organization = 'scallop'
    inspect_template = 'abalone'
    expected = 'organizations/{organization}/inspectTemplates/{inspect_template}'.format(organization=organization, inspect_template=inspect_template)
    actual = DlpServiceClient.inspect_template_path(organization, inspect_template)
    assert expected == actual

def test_parse_inspect_template_path():
    if False:
        return 10
    expected = {'organization': 'squid', 'inspect_template': 'clam'}
    path = DlpServiceClient.inspect_template_path(**expected)
    actual = DlpServiceClient.parse_inspect_template_path(path)
    assert expected == actual

def test_job_trigger_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    job_trigger = 'octopus'
    expected = 'projects/{project}/jobTriggers/{job_trigger}'.format(project=project, job_trigger=job_trigger)
    actual = DlpServiceClient.job_trigger_path(project, job_trigger)
    assert expected == actual

def test_parse_job_trigger_path():
    if False:
        print('Hello World!')
    expected = {'project': 'oyster', 'job_trigger': 'nudibranch'}
    path = DlpServiceClient.job_trigger_path(**expected)
    actual = DlpServiceClient.parse_job_trigger_path(path)
    assert expected == actual

def test_stored_info_type_path():
    if False:
        return 10
    organization = 'cuttlefish'
    stored_info_type = 'mussel'
    expected = 'organizations/{organization}/storedInfoTypes/{stored_info_type}'.format(organization=organization, stored_info_type=stored_info_type)
    actual = DlpServiceClient.stored_info_type_path(organization, stored_info_type)
    assert expected == actual

def test_parse_stored_info_type_path():
    if False:
        i = 10
        return i + 15
    expected = {'organization': 'winkle', 'stored_info_type': 'nautilus'}
    path = DlpServiceClient.stored_info_type_path(**expected)
    actual = DlpServiceClient.parse_stored_info_type_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'scallop'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = DlpServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'abalone'}
    path = DlpServiceClient.common_billing_account_path(**expected)
    actual = DlpServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'squid'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = DlpServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'clam'}
    path = DlpServiceClient.common_folder_path(**expected)
    actual = DlpServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'whelk'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = DlpServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'octopus'}
    path = DlpServiceClient.common_organization_path(**expected)
    actual = DlpServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'oyster'
    expected = 'projects/{project}'.format(project=project)
    actual = DlpServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nudibranch'}
    path = DlpServiceClient.common_project_path(**expected)
    actual = DlpServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'cuttlefish'
    location = 'mussel'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = DlpServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'winkle', 'location': 'nautilus'}
    path = DlpServiceClient.common_location_path(**expected)
    actual = DlpServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.DlpServiceTransport, '_prep_wrapped_messages') as prep:
        client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.DlpServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = DlpServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = DlpServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
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
        client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = DlpServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(DlpServiceClient, transports.DlpServiceGrpcTransport), (DlpServiceAsyncClient, transports.DlpServiceGrpcAsyncIOTransport)])
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