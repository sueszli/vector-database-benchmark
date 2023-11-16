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
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import empty_pb2
from google.protobuf import json_format
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.resourcemanager_v3.services.tag_bindings import TagBindingsAsyncClient, TagBindingsClient, pagers, transports
from google.cloud.resourcemanager_v3.types import tag_bindings

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
        i = 10
        return i + 15
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert TagBindingsClient._get_default_mtls_endpoint(None) is None
    assert TagBindingsClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert TagBindingsClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert TagBindingsClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert TagBindingsClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert TagBindingsClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(TagBindingsClient, 'grpc'), (TagBindingsAsyncClient, 'grpc_asyncio'), (TagBindingsClient, 'rest')])
def test_tag_bindings_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('cloudresourcemanager.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudresourcemanager.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.TagBindingsGrpcTransport, 'grpc'), (transports.TagBindingsGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.TagBindingsRestTransport, 'rest')])
def test_tag_bindings_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(TagBindingsClient, 'grpc'), (TagBindingsAsyncClient, 'grpc_asyncio'), (TagBindingsClient, 'rest')])
def test_tag_bindings_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('cloudresourcemanager.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudresourcemanager.googleapis.com')

def test_tag_bindings_client_get_transport_class():
    if False:
        return 10
    transport = TagBindingsClient.get_transport_class()
    available_transports = [transports.TagBindingsGrpcTransport, transports.TagBindingsRestTransport]
    assert transport in available_transports
    transport = TagBindingsClient.get_transport_class('grpc')
    assert transport == transports.TagBindingsGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(TagBindingsClient, transports.TagBindingsGrpcTransport, 'grpc'), (TagBindingsAsyncClient, transports.TagBindingsGrpcAsyncIOTransport, 'grpc_asyncio'), (TagBindingsClient, transports.TagBindingsRestTransport, 'rest')])
@mock.patch.object(TagBindingsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TagBindingsClient))
@mock.patch.object(TagBindingsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TagBindingsAsyncClient))
def test_tag_bindings_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(TagBindingsClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(TagBindingsClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(TagBindingsClient, transports.TagBindingsGrpcTransport, 'grpc', 'true'), (TagBindingsAsyncClient, transports.TagBindingsGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (TagBindingsClient, transports.TagBindingsGrpcTransport, 'grpc', 'false'), (TagBindingsAsyncClient, transports.TagBindingsGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (TagBindingsClient, transports.TagBindingsRestTransport, 'rest', 'true'), (TagBindingsClient, transports.TagBindingsRestTransport, 'rest', 'false')])
@mock.patch.object(TagBindingsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TagBindingsClient))
@mock.patch.object(TagBindingsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TagBindingsAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_tag_bindings_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [TagBindingsClient, TagBindingsAsyncClient])
@mock.patch.object(TagBindingsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TagBindingsClient))
@mock.patch.object(TagBindingsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TagBindingsAsyncClient))
def test_tag_bindings_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(TagBindingsClient, transports.TagBindingsGrpcTransport, 'grpc'), (TagBindingsAsyncClient, transports.TagBindingsGrpcAsyncIOTransport, 'grpc_asyncio'), (TagBindingsClient, transports.TagBindingsRestTransport, 'rest')])
def test_tag_bindings_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(TagBindingsClient, transports.TagBindingsGrpcTransport, 'grpc', grpc_helpers), (TagBindingsAsyncClient, transports.TagBindingsGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (TagBindingsClient, transports.TagBindingsRestTransport, 'rest', None)])
def test_tag_bindings_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_tag_bindings_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.resourcemanager_v3.services.tag_bindings.transports.TagBindingsGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = TagBindingsClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(TagBindingsClient, transports.TagBindingsGrpcTransport, 'grpc', grpc_helpers), (TagBindingsAsyncClient, transports.TagBindingsGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_tag_bindings_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('cloudresourcemanager.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), scopes=None, default_host='cloudresourcemanager.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [tag_bindings.ListTagBindingsRequest, dict])
def test_list_tag_bindings(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_tag_bindings), '__call__') as call:
        call.return_value = tag_bindings.ListTagBindingsResponse(next_page_token='next_page_token_value')
        response = client.list_tag_bindings(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_bindings.ListTagBindingsRequest()
    assert isinstance(response, pagers.ListTagBindingsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_tag_bindings_empty_call():
    if False:
        return 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_tag_bindings), '__call__') as call:
        client.list_tag_bindings()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_bindings.ListTagBindingsRequest()

@pytest.mark.asyncio
async def test_list_tag_bindings_async(transport: str='grpc_asyncio', request_type=tag_bindings.ListTagBindingsRequest):
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_tag_bindings), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tag_bindings.ListTagBindingsResponse(next_page_token='next_page_token_value'))
        response = await client.list_tag_bindings(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_bindings.ListTagBindingsRequest()
    assert isinstance(response, pagers.ListTagBindingsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_tag_bindings_async_from_dict():
    await test_list_tag_bindings_async(request_type=dict)

def test_list_tag_bindings_flattened():
    if False:
        print('Hello World!')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_tag_bindings), '__call__') as call:
        call.return_value = tag_bindings.ListTagBindingsResponse()
        client.list_tag_bindings(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_tag_bindings_flattened_error():
    if False:
        print('Hello World!')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_tag_bindings(tag_bindings.ListTagBindingsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_tag_bindings_flattened_async():
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_tag_bindings), '__call__') as call:
        call.return_value = tag_bindings.ListTagBindingsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tag_bindings.ListTagBindingsResponse())
        response = await client.list_tag_bindings(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_tag_bindings_flattened_error_async():
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_tag_bindings(tag_bindings.ListTagBindingsRequest(), parent='parent_value')

def test_list_tag_bindings_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_tag_bindings), '__call__') as call:
        call.side_effect = (tag_bindings.ListTagBindingsResponse(tag_bindings=[tag_bindings.TagBinding(), tag_bindings.TagBinding(), tag_bindings.TagBinding()], next_page_token='abc'), tag_bindings.ListTagBindingsResponse(tag_bindings=[], next_page_token='def'), tag_bindings.ListTagBindingsResponse(tag_bindings=[tag_bindings.TagBinding()], next_page_token='ghi'), tag_bindings.ListTagBindingsResponse(tag_bindings=[tag_bindings.TagBinding(), tag_bindings.TagBinding()]), RuntimeError)
        metadata = ()
        pager = client.list_tag_bindings(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tag_bindings.TagBinding) for i in results))

def test_list_tag_bindings_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_tag_bindings), '__call__') as call:
        call.side_effect = (tag_bindings.ListTagBindingsResponse(tag_bindings=[tag_bindings.TagBinding(), tag_bindings.TagBinding(), tag_bindings.TagBinding()], next_page_token='abc'), tag_bindings.ListTagBindingsResponse(tag_bindings=[], next_page_token='def'), tag_bindings.ListTagBindingsResponse(tag_bindings=[tag_bindings.TagBinding()], next_page_token='ghi'), tag_bindings.ListTagBindingsResponse(tag_bindings=[tag_bindings.TagBinding(), tag_bindings.TagBinding()]), RuntimeError)
        pages = list(client.list_tag_bindings(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_tag_bindings_async_pager():
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_tag_bindings), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (tag_bindings.ListTagBindingsResponse(tag_bindings=[tag_bindings.TagBinding(), tag_bindings.TagBinding(), tag_bindings.TagBinding()], next_page_token='abc'), tag_bindings.ListTagBindingsResponse(tag_bindings=[], next_page_token='def'), tag_bindings.ListTagBindingsResponse(tag_bindings=[tag_bindings.TagBinding()], next_page_token='ghi'), tag_bindings.ListTagBindingsResponse(tag_bindings=[tag_bindings.TagBinding(), tag_bindings.TagBinding()]), RuntimeError)
        async_pager = await client.list_tag_bindings(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, tag_bindings.TagBinding) for i in responses))

@pytest.mark.asyncio
async def test_list_tag_bindings_async_pages():
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_tag_bindings), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (tag_bindings.ListTagBindingsResponse(tag_bindings=[tag_bindings.TagBinding(), tag_bindings.TagBinding(), tag_bindings.TagBinding()], next_page_token='abc'), tag_bindings.ListTagBindingsResponse(tag_bindings=[], next_page_token='def'), tag_bindings.ListTagBindingsResponse(tag_bindings=[tag_bindings.TagBinding()], next_page_token='ghi'), tag_bindings.ListTagBindingsResponse(tag_bindings=[tag_bindings.TagBinding(), tag_bindings.TagBinding()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_tag_bindings(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [tag_bindings.CreateTagBindingRequest, dict])
def test_create_tag_binding(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_tag_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_tag_binding(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_bindings.CreateTagBindingRequest()
    assert isinstance(response, future.Future)

def test_create_tag_binding_empty_call():
    if False:
        return 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_tag_binding), '__call__') as call:
        client.create_tag_binding()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_bindings.CreateTagBindingRequest()

@pytest.mark.asyncio
async def test_create_tag_binding_async(transport: str='grpc_asyncio', request_type=tag_bindings.CreateTagBindingRequest):
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_tag_binding), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_tag_binding(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_bindings.CreateTagBindingRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_tag_binding_async_from_dict():
    await test_create_tag_binding_async(request_type=dict)

def test_create_tag_binding_flattened():
    if False:
        i = 10
        return i + 15
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_tag_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_tag_binding(tag_binding=tag_bindings.TagBinding(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].tag_binding
        mock_val = tag_bindings.TagBinding(name='name_value')
        assert arg == mock_val

def test_create_tag_binding_flattened_error():
    if False:
        i = 10
        return i + 15
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_tag_binding(tag_bindings.CreateTagBindingRequest(), tag_binding=tag_bindings.TagBinding(name='name_value'))

@pytest.mark.asyncio
async def test_create_tag_binding_flattened_async():
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_tag_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_tag_binding(tag_binding=tag_bindings.TagBinding(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].tag_binding
        mock_val = tag_bindings.TagBinding(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_tag_binding_flattened_error_async():
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_tag_binding(tag_bindings.CreateTagBindingRequest(), tag_binding=tag_bindings.TagBinding(name='name_value'))

@pytest.mark.parametrize('request_type', [tag_bindings.DeleteTagBindingRequest, dict])
def test_delete_tag_binding(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_tag_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_tag_binding(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_bindings.DeleteTagBindingRequest()
    assert isinstance(response, future.Future)

def test_delete_tag_binding_empty_call():
    if False:
        return 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_tag_binding), '__call__') as call:
        client.delete_tag_binding()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_bindings.DeleteTagBindingRequest()

@pytest.mark.asyncio
async def test_delete_tag_binding_async(transport: str='grpc_asyncio', request_type=tag_bindings.DeleteTagBindingRequest):
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_tag_binding), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_tag_binding(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_bindings.DeleteTagBindingRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_tag_binding_async_from_dict():
    await test_delete_tag_binding_async(request_type=dict)

def test_delete_tag_binding_field_headers():
    if False:
        return 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials())
    request = tag_bindings.DeleteTagBindingRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_tag_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_tag_binding(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_tag_binding_field_headers_async():
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tag_bindings.DeleteTagBindingRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_tag_binding), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_tag_binding(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_tag_binding_flattened():
    if False:
        while True:
            i = 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_tag_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_tag_binding(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_tag_binding_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_tag_binding(tag_bindings.DeleteTagBindingRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_tag_binding_flattened_async():
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_tag_binding), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_tag_binding(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_tag_binding_flattened_error_async():
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_tag_binding(tag_bindings.DeleteTagBindingRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [tag_bindings.ListEffectiveTagsRequest, dict])
def test_list_effective_tags(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_effective_tags), '__call__') as call:
        call.return_value = tag_bindings.ListEffectiveTagsResponse(next_page_token='next_page_token_value')
        response = client.list_effective_tags(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_bindings.ListEffectiveTagsRequest()
    assert isinstance(response, pagers.ListEffectiveTagsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_effective_tags_empty_call():
    if False:
        return 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_effective_tags), '__call__') as call:
        client.list_effective_tags()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_bindings.ListEffectiveTagsRequest()

@pytest.mark.asyncio
async def test_list_effective_tags_async(transport: str='grpc_asyncio', request_type=tag_bindings.ListEffectiveTagsRequest):
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_effective_tags), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tag_bindings.ListEffectiveTagsResponse(next_page_token='next_page_token_value'))
        response = await client.list_effective_tags(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_bindings.ListEffectiveTagsRequest()
    assert isinstance(response, pagers.ListEffectiveTagsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_effective_tags_async_from_dict():
    await test_list_effective_tags_async(request_type=dict)

def test_list_effective_tags_flattened():
    if False:
        print('Hello World!')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_effective_tags), '__call__') as call:
        call.return_value = tag_bindings.ListEffectiveTagsResponse()
        client.list_effective_tags(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_effective_tags_flattened_error():
    if False:
        return 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_effective_tags(tag_bindings.ListEffectiveTagsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_effective_tags_flattened_async():
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_effective_tags), '__call__') as call:
        call.return_value = tag_bindings.ListEffectiveTagsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tag_bindings.ListEffectiveTagsResponse())
        response = await client.list_effective_tags(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_effective_tags_flattened_error_async():
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_effective_tags(tag_bindings.ListEffectiveTagsRequest(), parent='parent_value')

def test_list_effective_tags_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_effective_tags), '__call__') as call:
        call.side_effect = (tag_bindings.ListEffectiveTagsResponse(effective_tags=[tag_bindings.EffectiveTag(), tag_bindings.EffectiveTag(), tag_bindings.EffectiveTag()], next_page_token='abc'), tag_bindings.ListEffectiveTagsResponse(effective_tags=[], next_page_token='def'), tag_bindings.ListEffectiveTagsResponse(effective_tags=[tag_bindings.EffectiveTag()], next_page_token='ghi'), tag_bindings.ListEffectiveTagsResponse(effective_tags=[tag_bindings.EffectiveTag(), tag_bindings.EffectiveTag()]), RuntimeError)
        metadata = ()
        pager = client.list_effective_tags(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tag_bindings.EffectiveTag) for i in results))

def test_list_effective_tags_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_effective_tags), '__call__') as call:
        call.side_effect = (tag_bindings.ListEffectiveTagsResponse(effective_tags=[tag_bindings.EffectiveTag(), tag_bindings.EffectiveTag(), tag_bindings.EffectiveTag()], next_page_token='abc'), tag_bindings.ListEffectiveTagsResponse(effective_tags=[], next_page_token='def'), tag_bindings.ListEffectiveTagsResponse(effective_tags=[tag_bindings.EffectiveTag()], next_page_token='ghi'), tag_bindings.ListEffectiveTagsResponse(effective_tags=[tag_bindings.EffectiveTag(), tag_bindings.EffectiveTag()]), RuntimeError)
        pages = list(client.list_effective_tags(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_effective_tags_async_pager():
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_effective_tags), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (tag_bindings.ListEffectiveTagsResponse(effective_tags=[tag_bindings.EffectiveTag(), tag_bindings.EffectiveTag(), tag_bindings.EffectiveTag()], next_page_token='abc'), tag_bindings.ListEffectiveTagsResponse(effective_tags=[], next_page_token='def'), tag_bindings.ListEffectiveTagsResponse(effective_tags=[tag_bindings.EffectiveTag()], next_page_token='ghi'), tag_bindings.ListEffectiveTagsResponse(effective_tags=[tag_bindings.EffectiveTag(), tag_bindings.EffectiveTag()]), RuntimeError)
        async_pager = await client.list_effective_tags(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, tag_bindings.EffectiveTag) for i in responses))

@pytest.mark.asyncio
async def test_list_effective_tags_async_pages():
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_effective_tags), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (tag_bindings.ListEffectiveTagsResponse(effective_tags=[tag_bindings.EffectiveTag(), tag_bindings.EffectiveTag(), tag_bindings.EffectiveTag()], next_page_token='abc'), tag_bindings.ListEffectiveTagsResponse(effective_tags=[], next_page_token='def'), tag_bindings.ListEffectiveTagsResponse(effective_tags=[tag_bindings.EffectiveTag()], next_page_token='ghi'), tag_bindings.ListEffectiveTagsResponse(effective_tags=[tag_bindings.EffectiveTag(), tag_bindings.EffectiveTag()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_effective_tags(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [tag_bindings.ListTagBindingsRequest, dict])
def test_list_tag_bindings_rest(request_type):
    if False:
        print('Hello World!')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tag_bindings.ListTagBindingsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = tag_bindings.ListTagBindingsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_tag_bindings(request)
    assert isinstance(response, pagers.ListTagBindingsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_tag_bindings_rest_required_fields(request_type=tag_bindings.ListTagBindingsRequest):
    if False:
        return 10
    transport_class = transports.TagBindingsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'parent' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_tag_bindings._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == request_init['parent']
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_tag_bindings._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token', 'parent'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tag_bindings.ListTagBindingsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tag_bindings.ListTagBindingsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_tag_bindings(request)
            expected_params = [('parent', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_tag_bindings_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.TagBindingsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_tag_bindings._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken', 'parent')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_tag_bindings_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.TagBindingsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TagBindingsRestInterceptor())
    client = TagBindingsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TagBindingsRestInterceptor, 'post_list_tag_bindings') as post, mock.patch.object(transports.TagBindingsRestInterceptor, 'pre_list_tag_bindings') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tag_bindings.ListTagBindingsRequest.pb(tag_bindings.ListTagBindingsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tag_bindings.ListTagBindingsResponse.to_json(tag_bindings.ListTagBindingsResponse())
        request = tag_bindings.ListTagBindingsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tag_bindings.ListTagBindingsResponse()
        client.list_tag_bindings(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_tag_bindings_rest_bad_request(transport: str='rest', request_type=tag_bindings.ListTagBindingsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_tag_bindings(request)

def test_list_tag_bindings_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tag_bindings.ListTagBindingsResponse()
        sample_request = {}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = tag_bindings.ListTagBindingsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_tag_bindings(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/tagBindings' % client.transport._host, args[1])

def test_list_tag_bindings_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_tag_bindings(tag_bindings.ListTagBindingsRequest(), parent='parent_value')

def test_list_tag_bindings_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (tag_bindings.ListTagBindingsResponse(tag_bindings=[tag_bindings.TagBinding(), tag_bindings.TagBinding(), tag_bindings.TagBinding()], next_page_token='abc'), tag_bindings.ListTagBindingsResponse(tag_bindings=[], next_page_token='def'), tag_bindings.ListTagBindingsResponse(tag_bindings=[tag_bindings.TagBinding()], next_page_token='ghi'), tag_bindings.ListTagBindingsResponse(tag_bindings=[tag_bindings.TagBinding(), tag_bindings.TagBinding()]))
        response = response + response
        response = tuple((tag_bindings.ListTagBindingsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {}
        pager = client.list_tag_bindings(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tag_bindings.TagBinding) for i in results))
        pages = list(client.list_tag_bindings(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [tag_bindings.CreateTagBindingRequest, dict])
def test_create_tag_binding_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request_init['tag_binding'] = {'name': 'name_value', 'parent': 'parent_value', 'tag_value': 'tag_value_value', 'tag_value_namespaced_name': 'tag_value_namespaced_name_value'}
    test_field = tag_bindings.CreateTagBindingRequest.meta.fields['tag_binding']

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
    for (field, value) in request_init['tag_binding'].items():
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
                for i in range(0, len(request_init['tag_binding'][field])):
                    del request_init['tag_binding'][field][i][subfield]
            else:
                del request_init['tag_binding'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_tag_binding(request)
    assert response.operation.name == 'operations/spam'

def test_create_tag_binding_rest_required_fields(request_type=tag_bindings.CreateTagBindingRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.TagBindingsRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_tag_binding._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_tag_binding._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('validate_only',))
    jsonified_request.update(unset_fields)
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_tag_binding(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_tag_binding_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.TagBindingsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_tag_binding._get_unset_required_fields({})
    assert set(unset_fields) == set(('validateOnly',)) & set(('tagBinding',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_tag_binding_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.TagBindingsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TagBindingsRestInterceptor())
    client = TagBindingsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.TagBindingsRestInterceptor, 'post_create_tag_binding') as post, mock.patch.object(transports.TagBindingsRestInterceptor, 'pre_create_tag_binding') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tag_bindings.CreateTagBindingRequest.pb(tag_bindings.CreateTagBindingRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = tag_bindings.CreateTagBindingRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_tag_binding(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_tag_binding_rest_bad_request(transport: str='rest', request_type=tag_bindings.CreateTagBindingRequest):
    if False:
        for i in range(10):
            print('nop')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_tag_binding(request)

def test_create_tag_binding_rest_flattened():
    if False:
        return 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {}
        mock_args = dict(tag_binding=tag_bindings.TagBinding(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_tag_binding(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/tagBindings' % client.transport._host, args[1])

def test_create_tag_binding_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_tag_binding(tag_bindings.CreateTagBindingRequest(), tag_binding=tag_bindings.TagBinding(name='name_value'))

def test_create_tag_binding_rest_error():
    if False:
        return 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [tag_bindings.DeleteTagBindingRequest, dict])
def test_delete_tag_binding_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'tagBindings/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_tag_binding(request)
    assert response.operation.name == 'operations/spam'

def test_delete_tag_binding_rest_required_fields(request_type=tag_bindings.DeleteTagBindingRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.TagBindingsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_tag_binding._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_tag_binding._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_tag_binding(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_tag_binding_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.TagBindingsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_tag_binding._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_tag_binding_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.TagBindingsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TagBindingsRestInterceptor())
    client = TagBindingsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.TagBindingsRestInterceptor, 'post_delete_tag_binding') as post, mock.patch.object(transports.TagBindingsRestInterceptor, 'pre_delete_tag_binding') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tag_bindings.DeleteTagBindingRequest.pb(tag_bindings.DeleteTagBindingRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = tag_bindings.DeleteTagBindingRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_tag_binding(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_tag_binding_rest_bad_request(transport: str='rest', request_type=tag_bindings.DeleteTagBindingRequest):
    if False:
        for i in range(10):
            print('nop')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'tagBindings/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_tag_binding(request)

def test_delete_tag_binding_rest_flattened():
    if False:
        return 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'tagBindings/sample1'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_tag_binding(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/{name=tagBindings/**}' % client.transport._host, args[1])

def test_delete_tag_binding_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_tag_binding(tag_bindings.DeleteTagBindingRequest(), name='name_value')

def test_delete_tag_binding_rest_error():
    if False:
        print('Hello World!')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [tag_bindings.ListEffectiveTagsRequest, dict])
def test_list_effective_tags_rest(request_type):
    if False:
        return 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tag_bindings.ListEffectiveTagsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = tag_bindings.ListEffectiveTagsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_effective_tags(request)
    assert isinstance(response, pagers.ListEffectiveTagsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_effective_tags_rest_required_fields(request_type=tag_bindings.ListEffectiveTagsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.TagBindingsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'parent' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_effective_tags._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == request_init['parent']
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_effective_tags._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token', 'parent'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tag_bindings.ListEffectiveTagsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tag_bindings.ListEffectiveTagsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_effective_tags(request)
            expected_params = [('parent', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_effective_tags_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.TagBindingsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_effective_tags._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken', 'parent')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_effective_tags_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.TagBindingsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TagBindingsRestInterceptor())
    client = TagBindingsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TagBindingsRestInterceptor, 'post_list_effective_tags') as post, mock.patch.object(transports.TagBindingsRestInterceptor, 'pre_list_effective_tags') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tag_bindings.ListEffectiveTagsRequest.pb(tag_bindings.ListEffectiveTagsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tag_bindings.ListEffectiveTagsResponse.to_json(tag_bindings.ListEffectiveTagsResponse())
        request = tag_bindings.ListEffectiveTagsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tag_bindings.ListEffectiveTagsResponse()
        client.list_effective_tags(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_effective_tags_rest_bad_request(transport: str='rest', request_type=tag_bindings.ListEffectiveTagsRequest):
    if False:
        return 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_effective_tags(request)

def test_list_effective_tags_rest_flattened():
    if False:
        while True:
            i = 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tag_bindings.ListEffectiveTagsResponse()
        sample_request = {}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = tag_bindings.ListEffectiveTagsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_effective_tags(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/effectiveTags' % client.transport._host, args[1])

def test_list_effective_tags_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_effective_tags(tag_bindings.ListEffectiveTagsRequest(), parent='parent_value')

def test_list_effective_tags_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (tag_bindings.ListEffectiveTagsResponse(effective_tags=[tag_bindings.EffectiveTag(), tag_bindings.EffectiveTag(), tag_bindings.EffectiveTag()], next_page_token='abc'), tag_bindings.ListEffectiveTagsResponse(effective_tags=[], next_page_token='def'), tag_bindings.ListEffectiveTagsResponse(effective_tags=[tag_bindings.EffectiveTag()], next_page_token='ghi'), tag_bindings.ListEffectiveTagsResponse(effective_tags=[tag_bindings.EffectiveTag(), tag_bindings.EffectiveTag()]))
        response = response + response
        response = tuple((tag_bindings.ListEffectiveTagsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {}
        pager = client.list_effective_tags(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tag_bindings.EffectiveTag) for i in results))
        pages = list(client.list_effective_tags(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.TagBindingsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.TagBindingsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TagBindingsClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.TagBindingsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = TagBindingsClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = TagBindingsClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.TagBindingsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TagBindingsClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.TagBindingsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = TagBindingsClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.TagBindingsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.TagBindingsGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.TagBindingsGrpcTransport, transports.TagBindingsGrpcAsyncIOTransport, transports.TagBindingsRestTransport])
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
        while True:
            i = 10
    transport = TagBindingsClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.TagBindingsGrpcTransport)

def test_tag_bindings_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.TagBindingsTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_tag_bindings_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.resourcemanager_v3.services.tag_bindings.transports.TagBindingsTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.TagBindingsTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_tag_bindings', 'create_tag_binding', 'delete_tag_binding', 'list_effective_tags', 'get_operation')
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

def test_tag_bindings_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.resourcemanager_v3.services.tag_bindings.transports.TagBindingsTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.TagBindingsTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), quota_project_id='octopus')

def test_tag_bindings_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.resourcemanager_v3.services.tag_bindings.transports.TagBindingsTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.TagBindingsTransport()
        adc.assert_called_once()

def test_tag_bindings_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        TagBindingsClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.TagBindingsGrpcTransport, transports.TagBindingsGrpcAsyncIOTransport])
def test_tag_bindings_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.TagBindingsGrpcTransport, transports.TagBindingsGrpcAsyncIOTransport, transports.TagBindingsRestTransport])
def test_tag_bindings_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.TagBindingsGrpcTransport, grpc_helpers), (transports.TagBindingsGrpcAsyncIOTransport, grpc_helpers_async)])
def test_tag_bindings_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('cloudresourcemanager.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), scopes=['1', '2'], default_host='cloudresourcemanager.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.TagBindingsGrpcTransport, transports.TagBindingsGrpcAsyncIOTransport])
def test_tag_bindings_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_tag_bindings_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.TagBindingsRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_tag_bindings_rest_lro_client():
    if False:
        print('Hello World!')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_tag_bindings_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudresourcemanager.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('cloudresourcemanager.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudresourcemanager.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_tag_bindings_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudresourcemanager.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('cloudresourcemanager.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudresourcemanager.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_tag_bindings_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = TagBindingsClient(credentials=creds1, transport=transport_name)
    client2 = TagBindingsClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_tag_bindings._session
    session2 = client2.transport.list_tag_bindings._session
    assert session1 != session2
    session1 = client1.transport.create_tag_binding._session
    session2 = client2.transport.create_tag_binding._session
    assert session1 != session2
    session1 = client1.transport.delete_tag_binding._session
    session2 = client2.transport.delete_tag_binding._session
    assert session1 != session2
    session1 = client1.transport.list_effective_tags._session
    session2 = client2.transport.list_effective_tags._session
    assert session1 != session2

def test_tag_bindings_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.TagBindingsGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_tag_bindings_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.TagBindingsGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.TagBindingsGrpcTransport, transports.TagBindingsGrpcAsyncIOTransport])
def test_tag_bindings_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.TagBindingsGrpcTransport, transports.TagBindingsGrpcAsyncIOTransport])
def test_tag_bindings_transport_channel_mtls_with_adc(transport_class):
    if False:
        i = 10
        return i + 15
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

def test_tag_bindings_grpc_lro_client():
    if False:
        while True:
            i = 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_tag_bindings_grpc_lro_async_client():
    if False:
        print('Hello World!')
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_tag_binding_path():
    if False:
        i = 10
        return i + 15
    tag_binding = 'squid'
    expected = 'tagBindings/{tag_binding}'.format(tag_binding=tag_binding)
    actual = TagBindingsClient.tag_binding_path(tag_binding)
    assert expected == actual

def test_parse_tag_binding_path():
    if False:
        return 10
    expected = {'tag_binding': 'clam'}
    path = TagBindingsClient.tag_binding_path(**expected)
    actual = TagBindingsClient.parse_tag_binding_path(path)
    assert expected == actual

def test_tag_key_path():
    if False:
        return 10
    tag_key = 'whelk'
    expected = 'tagKeys/{tag_key}'.format(tag_key=tag_key)
    actual = TagBindingsClient.tag_key_path(tag_key)
    assert expected == actual

def test_parse_tag_key_path():
    if False:
        print('Hello World!')
    expected = {'tag_key': 'octopus'}
    path = TagBindingsClient.tag_key_path(**expected)
    actual = TagBindingsClient.parse_tag_key_path(path)
    assert expected == actual

def test_tag_value_path():
    if False:
        print('Hello World!')
    tag_value = 'oyster'
    expected = 'tagValues/{tag_value}'.format(tag_value=tag_value)
    actual = TagBindingsClient.tag_value_path(tag_value)
    assert expected == actual

def test_parse_tag_value_path():
    if False:
        return 10
    expected = {'tag_value': 'nudibranch'}
    path = TagBindingsClient.tag_value_path(**expected)
    actual = TagBindingsClient.parse_tag_value_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = TagBindingsClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'mussel'}
    path = TagBindingsClient.common_billing_account_path(**expected)
    actual = TagBindingsClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = TagBindingsClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'nautilus'}
    path = TagBindingsClient.common_folder_path(**expected)
    actual = TagBindingsClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = TagBindingsClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'abalone'}
    path = TagBindingsClient.common_organization_path(**expected)
    actual = TagBindingsClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = TagBindingsClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'clam'}
    path = TagBindingsClient.common_project_path(**expected)
    actual = TagBindingsClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = TagBindingsClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = TagBindingsClient.common_location_path(**expected)
    actual = TagBindingsClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.TagBindingsTransport, '_prep_wrapped_messages') as prep:
        client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.TagBindingsTransport, '_prep_wrapped_messages') as prep:
        transport_class = TagBindingsClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        print('Hello World!')
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_get_operation(transport: str='grpc'):
    if False:
        return 10
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = TagBindingsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        while True:
            i = 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = TagBindingsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(TagBindingsClient, transports.TagBindingsGrpcTransport), (TagBindingsAsyncClient, transports.TagBindingsGrpcAsyncIOTransport)])
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