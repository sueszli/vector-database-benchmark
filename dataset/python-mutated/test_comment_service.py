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
from google.protobuf import timestamp_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.support_v2.services.comment_service import CommentServiceAsyncClient, CommentServiceClient, pagers, transports
from google.cloud.support_v2.types import actor
from google.cloud.support_v2.types import comment
from google.cloud.support_v2.types import comment as gcs_comment
from google.cloud.support_v2.types import comment_service

def client_cert_source_callback():
    if False:
        while True:
            i = 10
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        print('Hello World!')
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        print('Hello World!')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert CommentServiceClient._get_default_mtls_endpoint(None) is None
    assert CommentServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert CommentServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert CommentServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert CommentServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert CommentServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(CommentServiceClient, 'grpc'), (CommentServiceAsyncClient, 'grpc_asyncio'), (CommentServiceClient, 'rest')])
def test_comment_service_client_from_service_account_info(client_class, transport_name):
    if False:
        i = 10
        return i + 15
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('cloudsupport.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudsupport.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.CommentServiceGrpcTransport, 'grpc'), (transports.CommentServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.CommentServiceRestTransport, 'rest')])
def test_comment_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(CommentServiceClient, 'grpc'), (CommentServiceAsyncClient, 'grpc_asyncio'), (CommentServiceClient, 'rest')])
def test_comment_service_client_from_service_account_file(client_class, transport_name):
    if False:
        while True:
            i = 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_file') as factory:
        factory.return_value = creds
        client = client_class.from_service_account_file('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        client = client_class.from_service_account_json('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('cloudsupport.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudsupport.googleapis.com')

def test_comment_service_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = CommentServiceClient.get_transport_class()
    available_transports = [transports.CommentServiceGrpcTransport, transports.CommentServiceRestTransport]
    assert transport in available_transports
    transport = CommentServiceClient.get_transport_class('grpc')
    assert transport == transports.CommentServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CommentServiceClient, transports.CommentServiceGrpcTransport, 'grpc'), (CommentServiceAsyncClient, transports.CommentServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (CommentServiceClient, transports.CommentServiceRestTransport, 'rest')])
@mock.patch.object(CommentServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CommentServiceClient))
@mock.patch.object(CommentServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CommentServiceAsyncClient))
def test_comment_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(CommentServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(CommentServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(CommentServiceClient, transports.CommentServiceGrpcTransport, 'grpc', 'true'), (CommentServiceAsyncClient, transports.CommentServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (CommentServiceClient, transports.CommentServiceGrpcTransport, 'grpc', 'false'), (CommentServiceAsyncClient, transports.CommentServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (CommentServiceClient, transports.CommentServiceRestTransport, 'rest', 'true'), (CommentServiceClient, transports.CommentServiceRestTransport, 'rest', 'false')])
@mock.patch.object(CommentServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CommentServiceClient))
@mock.patch.object(CommentServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CommentServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_comment_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [CommentServiceClient, CommentServiceAsyncClient])
@mock.patch.object(CommentServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CommentServiceClient))
@mock.patch.object(CommentServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CommentServiceAsyncClient))
def test_comment_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CommentServiceClient, transports.CommentServiceGrpcTransport, 'grpc'), (CommentServiceAsyncClient, transports.CommentServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (CommentServiceClient, transports.CommentServiceRestTransport, 'rest')])
def test_comment_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CommentServiceClient, transports.CommentServiceGrpcTransport, 'grpc', grpc_helpers), (CommentServiceAsyncClient, transports.CommentServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (CommentServiceClient, transports.CommentServiceRestTransport, 'rest', None)])
def test_comment_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_comment_service_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.support_v2.services.comment_service.transports.CommentServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = CommentServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CommentServiceClient, transports.CommentServiceGrpcTransport, 'grpc', grpc_helpers), (CommentServiceAsyncClient, transports.CommentServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_comment_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('cloudsupport.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='cloudsupport.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [comment_service.ListCommentsRequest, dict])
def test_list_comments(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_comments), '__call__') as call:
        call.return_value = comment_service.ListCommentsResponse(next_page_token='next_page_token_value')
        response = client.list_comments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == comment_service.ListCommentsRequest()
    assert isinstance(response, pagers.ListCommentsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_comments_empty_call():
    if False:
        return 10
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_comments), '__call__') as call:
        client.list_comments()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == comment_service.ListCommentsRequest()

@pytest.mark.asyncio
async def test_list_comments_async(transport: str='grpc_asyncio', request_type=comment_service.ListCommentsRequest):
    client = CommentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_comments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(comment_service.ListCommentsResponse(next_page_token='next_page_token_value'))
        response = await client.list_comments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == comment_service.ListCommentsRequest()
    assert isinstance(response, pagers.ListCommentsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_comments_async_from_dict():
    await test_list_comments_async(request_type=dict)

def test_list_comments_field_headers():
    if False:
        while True:
            i = 10
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = comment_service.ListCommentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_comments), '__call__') as call:
        call.return_value = comment_service.ListCommentsResponse()
        client.list_comments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_comments_field_headers_async():
    client = CommentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = comment_service.ListCommentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_comments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(comment_service.ListCommentsResponse())
        await client.list_comments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_comments_flattened():
    if False:
        i = 10
        return i + 15
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_comments), '__call__') as call:
        call.return_value = comment_service.ListCommentsResponse()
        client.list_comments(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_comments_flattened_error():
    if False:
        print('Hello World!')
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_comments(comment_service.ListCommentsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_comments_flattened_async():
    client = CommentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_comments), '__call__') as call:
        call.return_value = comment_service.ListCommentsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(comment_service.ListCommentsResponse())
        response = await client.list_comments(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_comments_flattened_error_async():
    client = CommentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_comments(comment_service.ListCommentsRequest(), parent='parent_value')

def test_list_comments_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_comments), '__call__') as call:
        call.side_effect = (comment_service.ListCommentsResponse(comments=[comment.Comment(), comment.Comment(), comment.Comment()], next_page_token='abc'), comment_service.ListCommentsResponse(comments=[], next_page_token='def'), comment_service.ListCommentsResponse(comments=[comment.Comment()], next_page_token='ghi'), comment_service.ListCommentsResponse(comments=[comment.Comment(), comment.Comment()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_comments(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, comment.Comment) for i in results))

def test_list_comments_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_comments), '__call__') as call:
        call.side_effect = (comment_service.ListCommentsResponse(comments=[comment.Comment(), comment.Comment(), comment.Comment()], next_page_token='abc'), comment_service.ListCommentsResponse(comments=[], next_page_token='def'), comment_service.ListCommentsResponse(comments=[comment.Comment()], next_page_token='ghi'), comment_service.ListCommentsResponse(comments=[comment.Comment(), comment.Comment()]), RuntimeError)
        pages = list(client.list_comments(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_comments_async_pager():
    client = CommentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_comments), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (comment_service.ListCommentsResponse(comments=[comment.Comment(), comment.Comment(), comment.Comment()], next_page_token='abc'), comment_service.ListCommentsResponse(comments=[], next_page_token='def'), comment_service.ListCommentsResponse(comments=[comment.Comment()], next_page_token='ghi'), comment_service.ListCommentsResponse(comments=[comment.Comment(), comment.Comment()]), RuntimeError)
        async_pager = await client.list_comments(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, comment.Comment) for i in responses))

@pytest.mark.asyncio
async def test_list_comments_async_pages():
    client = CommentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_comments), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (comment_service.ListCommentsResponse(comments=[comment.Comment(), comment.Comment(), comment.Comment()], next_page_token='abc'), comment_service.ListCommentsResponse(comments=[], next_page_token='def'), comment_service.ListCommentsResponse(comments=[comment.Comment()], next_page_token='ghi'), comment_service.ListCommentsResponse(comments=[comment.Comment(), comment.Comment()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_comments(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [comment_service.CreateCommentRequest, dict])
def test_create_comment(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_comment), '__call__') as call:
        call.return_value = gcs_comment.Comment(name='name_value', body='body_value', plain_text_body='plain_text_body_value')
        response = client.create_comment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == comment_service.CreateCommentRequest()
    assert isinstance(response, gcs_comment.Comment)
    assert response.name == 'name_value'
    assert response.body == 'body_value'
    assert response.plain_text_body == 'plain_text_body_value'

def test_create_comment_empty_call():
    if False:
        return 10
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_comment), '__call__') as call:
        client.create_comment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == comment_service.CreateCommentRequest()

@pytest.mark.asyncio
async def test_create_comment_async(transport: str='grpc_asyncio', request_type=comment_service.CreateCommentRequest):
    client = CommentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_comment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_comment.Comment(name='name_value', body='body_value', plain_text_body='plain_text_body_value'))
        response = await client.create_comment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == comment_service.CreateCommentRequest()
    assert isinstance(response, gcs_comment.Comment)
    assert response.name == 'name_value'
    assert response.body == 'body_value'
    assert response.plain_text_body == 'plain_text_body_value'

@pytest.mark.asyncio
async def test_create_comment_async_from_dict():
    await test_create_comment_async(request_type=dict)

def test_create_comment_field_headers():
    if False:
        i = 10
        return i + 15
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = comment_service.CreateCommentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_comment), '__call__') as call:
        call.return_value = gcs_comment.Comment()
        client.create_comment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_comment_field_headers_async():
    client = CommentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = comment_service.CreateCommentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_comment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_comment.Comment())
        await client.create_comment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_comment_flattened():
    if False:
        i = 10
        return i + 15
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_comment), '__call__') as call:
        call.return_value = gcs_comment.Comment()
        client.create_comment(parent='parent_value', comment=gcs_comment.Comment(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].comment
        mock_val = gcs_comment.Comment(name='name_value')
        assert arg == mock_val

def test_create_comment_flattened_error():
    if False:
        while True:
            i = 10
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_comment(comment_service.CreateCommentRequest(), parent='parent_value', comment=gcs_comment.Comment(name='name_value'))

@pytest.mark.asyncio
async def test_create_comment_flattened_async():
    client = CommentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_comment), '__call__') as call:
        call.return_value = gcs_comment.Comment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_comment.Comment())
        response = await client.create_comment(parent='parent_value', comment=gcs_comment.Comment(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].comment
        mock_val = gcs_comment.Comment(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_comment_flattened_error_async():
    client = CommentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_comment(comment_service.CreateCommentRequest(), parent='parent_value', comment=gcs_comment.Comment(name='name_value'))

@pytest.mark.parametrize('request_type', [comment_service.ListCommentsRequest, dict])
def test_list_comments_rest(request_type):
    if False:
        return 10
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/cases/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = comment_service.ListCommentsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = comment_service.ListCommentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_comments(request)
    assert isinstance(response, pagers.ListCommentsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_comments_rest_required_fields(request_type=comment_service.ListCommentsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CommentServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_comments._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_comments._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = comment_service.ListCommentsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = comment_service.ListCommentsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_comments(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_comments_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CommentServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_comments._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_comments_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CommentServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CommentServiceRestInterceptor())
    client = CommentServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CommentServiceRestInterceptor, 'post_list_comments') as post, mock.patch.object(transports.CommentServiceRestInterceptor, 'pre_list_comments') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = comment_service.ListCommentsRequest.pb(comment_service.ListCommentsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = comment_service.ListCommentsResponse.to_json(comment_service.ListCommentsResponse())
        request = comment_service.ListCommentsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = comment_service.ListCommentsResponse()
        client.list_comments(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_comments_rest_bad_request(transport: str='rest', request_type=comment_service.ListCommentsRequest):
    if False:
        print('Hello World!')
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/cases/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_comments(request)

def test_list_comments_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = comment_service.ListCommentsResponse()
        sample_request = {'parent': 'projects/sample1/cases/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = comment_service.ListCommentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_comments(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/cases/*}/comments' % client.transport._host, args[1])

def test_list_comments_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_comments(comment_service.ListCommentsRequest(), parent='parent_value')

def test_list_comments_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (comment_service.ListCommentsResponse(comments=[comment.Comment(), comment.Comment(), comment.Comment()], next_page_token='abc'), comment_service.ListCommentsResponse(comments=[], next_page_token='def'), comment_service.ListCommentsResponse(comments=[comment.Comment()], next_page_token='ghi'), comment_service.ListCommentsResponse(comments=[comment.Comment(), comment.Comment()]))
        response = response + response
        response = tuple((comment_service.ListCommentsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/cases/sample2'}
        pager = client.list_comments(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, comment.Comment) for i in results))
        pages = list(client.list_comments(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [comment_service.CreateCommentRequest, dict])
def test_create_comment_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/cases/sample2'}
    request_init['comment'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'creator': {'display_name': 'display_name_value', 'email': 'email_value', 'google_support': True}, 'body': 'body_value', 'plain_text_body': 'plain_text_body_value'}
    test_field = comment_service.CreateCommentRequest.meta.fields['comment']

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
    for (field, value) in request_init['comment'].items():
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
                for i in range(0, len(request_init['comment'][field])):
                    del request_init['comment'][field][i][subfield]
            else:
                del request_init['comment'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_comment.Comment(name='name_value', body='body_value', plain_text_body='plain_text_body_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_comment.Comment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_comment(request)
    assert isinstance(response, gcs_comment.Comment)
    assert response.name == 'name_value'
    assert response.body == 'body_value'
    assert response.plain_text_body == 'plain_text_body_value'

def test_create_comment_rest_required_fields(request_type=comment_service.CreateCommentRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CommentServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_comment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_comment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcs_comment.Comment()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcs_comment.Comment.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_comment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_comment_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CommentServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_comment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'comment'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_comment_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CommentServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CommentServiceRestInterceptor())
    client = CommentServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CommentServiceRestInterceptor, 'post_create_comment') as post, mock.patch.object(transports.CommentServiceRestInterceptor, 'pre_create_comment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = comment_service.CreateCommentRequest.pb(comment_service.CreateCommentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcs_comment.Comment.to_json(gcs_comment.Comment())
        request = comment_service.CreateCommentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcs_comment.Comment()
        client.create_comment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_comment_rest_bad_request(transport: str='rest', request_type=comment_service.CreateCommentRequest):
    if False:
        return 10
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/cases/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_comment(request)

def test_create_comment_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_comment.Comment()
        sample_request = {'parent': 'projects/sample1/cases/sample2'}
        mock_args = dict(parent='parent_value', comment=gcs_comment.Comment(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_comment.Comment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_comment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/cases/*}/comments' % client.transport._host, args[1])

def test_create_comment_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_comment(comment_service.CreateCommentRequest(), parent='parent_value', comment=gcs_comment.Comment(name='name_value'))

def test_create_comment_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        print('Hello World!')
    transport = transports.CommentServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.CommentServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CommentServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.CommentServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CommentServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CommentServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.CommentServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CommentServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.CommentServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = CommentServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.CommentServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.CommentServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.CommentServiceGrpcTransport, transports.CommentServiceGrpcAsyncIOTransport, transports.CommentServiceRestTransport])
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
        while True:
            i = 10
    transport = CommentServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.CommentServiceGrpcTransport)

def test_comment_service_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.CommentServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_comment_service_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.support_v2.services.comment_service.transports.CommentServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.CommentServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_comments', 'create_comment')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_comment_service_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.support_v2.services.comment_service.transports.CommentServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CommentServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_comment_service_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.support_v2.services.comment_service.transports.CommentServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CommentServiceTransport()
        adc.assert_called_once()

def test_comment_service_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        CommentServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.CommentServiceGrpcTransport, transports.CommentServiceGrpcAsyncIOTransport])
def test_comment_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.CommentServiceGrpcTransport, transports.CommentServiceGrpcAsyncIOTransport, transports.CommentServiceRestTransport])
def test_comment_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.CommentServiceGrpcTransport, grpc_helpers), (transports.CommentServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_comment_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('cloudsupport.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='cloudsupport.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.CommentServiceGrpcTransport, transports.CommentServiceGrpcAsyncIOTransport])
def test_comment_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_comment_service_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.CommentServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_comment_service_host_no_port(transport_name):
    if False:
        return 10
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudsupport.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('cloudsupport.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudsupport.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_comment_service_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudsupport.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('cloudsupport.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudsupport.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_comment_service_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = CommentServiceClient(credentials=creds1, transport=transport_name)
    client2 = CommentServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_comments._session
    session2 = client2.transport.list_comments._session
    assert session1 != session2
    session1 = client1.transport.create_comment._session
    session2 = client2.transport.create_comment._session
    assert session1 != session2

def test_comment_service_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CommentServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_comment_service_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CommentServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.CommentServiceGrpcTransport, transports.CommentServiceGrpcAsyncIOTransport])
def test_comment_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.CommentServiceGrpcTransport, transports.CommentServiceGrpcAsyncIOTransport])
def test_comment_service_transport_channel_mtls_with_adc(transport_class):
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

def test_case_path():
    if False:
        print('Hello World!')
    organization = 'squid'
    case = 'clam'
    expected = 'organizations/{organization}/cases/{case}'.format(organization=organization, case=case)
    actual = CommentServiceClient.case_path(organization, case)
    assert expected == actual

def test_parse_case_path():
    if False:
        i = 10
        return i + 15
    expected = {'organization': 'whelk', 'case': 'octopus'}
    path = CommentServiceClient.case_path(**expected)
    actual = CommentServiceClient.parse_case_path(path)
    assert expected == actual

def test_comment_path():
    if False:
        print('Hello World!')
    organization = 'oyster'
    case = 'nudibranch'
    comment = 'cuttlefish'
    expected = 'organizations/{organization}/cases/{case}/comments/{comment}'.format(organization=organization, case=case, comment=comment)
    actual = CommentServiceClient.comment_path(organization, case, comment)
    assert expected == actual

def test_parse_comment_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'mussel', 'case': 'winkle', 'comment': 'nautilus'}
    path = CommentServiceClient.comment_path(**expected)
    actual = CommentServiceClient.parse_comment_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'scallop'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = CommentServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'abalone'}
    path = CommentServiceClient.common_billing_account_path(**expected)
    actual = CommentServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'squid'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = CommentServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'clam'}
    path = CommentServiceClient.common_folder_path(**expected)
    actual = CommentServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'whelk'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = CommentServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'octopus'}
    path = CommentServiceClient.common_organization_path(**expected)
    actual = CommentServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'oyster'
    expected = 'projects/{project}'.format(project=project)
    actual = CommentServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'nudibranch'}
    path = CommentServiceClient.common_project_path(**expected)
    actual = CommentServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    location = 'mussel'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = CommentServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'winkle', 'location': 'nautilus'}
    path = CommentServiceClient.common_location_path(**expected)
    actual = CommentServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.CommentServiceTransport, '_prep_wrapped_messages') as prep:
        client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.CommentServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = CommentServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = CommentServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        for i in range(10):
            print('nop')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = CommentServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(CommentServiceClient, transports.CommentServiceGrpcTransport), (CommentServiceAsyncClient, transports.CommentServiceGrpcAsyncIOTransport)])
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