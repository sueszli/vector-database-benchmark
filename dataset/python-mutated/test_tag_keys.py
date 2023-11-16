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
from google.iam.v1 import iam_policy_pb2
from google.iam.v1 import options_pb2
from google.iam.v1 import policy_pb2
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.type import expr_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.resourcemanager_v3.services.tag_keys import TagKeysAsyncClient, TagKeysClient, pagers, transports
from google.cloud.resourcemanager_v3.types import tag_keys

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
    assert TagKeysClient._get_default_mtls_endpoint(None) is None
    assert TagKeysClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert TagKeysClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert TagKeysClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert TagKeysClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert TagKeysClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(TagKeysClient, 'grpc'), (TagKeysAsyncClient, 'grpc_asyncio'), (TagKeysClient, 'rest')])
def test_tag_keys_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('cloudresourcemanager.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudresourcemanager.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.TagKeysGrpcTransport, 'grpc'), (transports.TagKeysGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.TagKeysRestTransport, 'rest')])
def test_tag_keys_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(TagKeysClient, 'grpc'), (TagKeysAsyncClient, 'grpc_asyncio'), (TagKeysClient, 'rest')])
def test_tag_keys_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('cloudresourcemanager.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudresourcemanager.googleapis.com')

def test_tag_keys_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = TagKeysClient.get_transport_class()
    available_transports = [transports.TagKeysGrpcTransport, transports.TagKeysRestTransport]
    assert transport in available_transports
    transport = TagKeysClient.get_transport_class('grpc')
    assert transport == transports.TagKeysGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(TagKeysClient, transports.TagKeysGrpcTransport, 'grpc'), (TagKeysAsyncClient, transports.TagKeysGrpcAsyncIOTransport, 'grpc_asyncio'), (TagKeysClient, transports.TagKeysRestTransport, 'rest')])
@mock.patch.object(TagKeysClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TagKeysClient))
@mock.patch.object(TagKeysAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TagKeysAsyncClient))
def test_tag_keys_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(TagKeysClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(TagKeysClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(TagKeysClient, transports.TagKeysGrpcTransport, 'grpc', 'true'), (TagKeysAsyncClient, transports.TagKeysGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (TagKeysClient, transports.TagKeysGrpcTransport, 'grpc', 'false'), (TagKeysAsyncClient, transports.TagKeysGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (TagKeysClient, transports.TagKeysRestTransport, 'rest', 'true'), (TagKeysClient, transports.TagKeysRestTransport, 'rest', 'false')])
@mock.patch.object(TagKeysClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TagKeysClient))
@mock.patch.object(TagKeysAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TagKeysAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_tag_keys_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [TagKeysClient, TagKeysAsyncClient])
@mock.patch.object(TagKeysClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TagKeysClient))
@mock.patch.object(TagKeysAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TagKeysAsyncClient))
def test_tag_keys_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(TagKeysClient, transports.TagKeysGrpcTransport, 'grpc'), (TagKeysAsyncClient, transports.TagKeysGrpcAsyncIOTransport, 'grpc_asyncio'), (TagKeysClient, transports.TagKeysRestTransport, 'rest')])
def test_tag_keys_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(TagKeysClient, transports.TagKeysGrpcTransport, 'grpc', grpc_helpers), (TagKeysAsyncClient, transports.TagKeysGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (TagKeysClient, transports.TagKeysRestTransport, 'rest', None)])
def test_tag_keys_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_tag_keys_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.resourcemanager_v3.services.tag_keys.transports.TagKeysGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = TagKeysClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(TagKeysClient, transports.TagKeysGrpcTransport, 'grpc', grpc_helpers), (TagKeysAsyncClient, transports.TagKeysGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_tag_keys_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('cloudresourcemanager.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), scopes=None, default_host='cloudresourcemanager.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [tag_keys.ListTagKeysRequest, dict])
def test_list_tag_keys(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_tag_keys), '__call__') as call:
        call.return_value = tag_keys.ListTagKeysResponse(next_page_token='next_page_token_value')
        response = client.list_tag_keys(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_keys.ListTagKeysRequest()
    assert isinstance(response, pagers.ListTagKeysPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_tag_keys_empty_call():
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_tag_keys), '__call__') as call:
        client.list_tag_keys()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_keys.ListTagKeysRequest()

@pytest.mark.asyncio
async def test_list_tag_keys_async(transport: str='grpc_asyncio', request_type=tag_keys.ListTagKeysRequest):
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_tag_keys), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tag_keys.ListTagKeysResponse(next_page_token='next_page_token_value'))
        response = await client.list_tag_keys(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_keys.ListTagKeysRequest()
    assert isinstance(response, pagers.ListTagKeysAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_tag_keys_async_from_dict():
    await test_list_tag_keys_async(request_type=dict)

def test_list_tag_keys_flattened():
    if False:
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_tag_keys), '__call__') as call:
        call.return_value = tag_keys.ListTagKeysResponse()
        client.list_tag_keys(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_tag_keys_flattened_error():
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_tag_keys(tag_keys.ListTagKeysRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_tag_keys_flattened_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_tag_keys), '__call__') as call:
        call.return_value = tag_keys.ListTagKeysResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tag_keys.ListTagKeysResponse())
        response = await client.list_tag_keys(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_tag_keys_flattened_error_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_tag_keys(tag_keys.ListTagKeysRequest(), parent='parent_value')

def test_list_tag_keys_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_tag_keys), '__call__') as call:
        call.side_effect = (tag_keys.ListTagKeysResponse(tag_keys=[tag_keys.TagKey(), tag_keys.TagKey(), tag_keys.TagKey()], next_page_token='abc'), tag_keys.ListTagKeysResponse(tag_keys=[], next_page_token='def'), tag_keys.ListTagKeysResponse(tag_keys=[tag_keys.TagKey()], next_page_token='ghi'), tag_keys.ListTagKeysResponse(tag_keys=[tag_keys.TagKey(), tag_keys.TagKey()]), RuntimeError)
        metadata = ()
        pager = client.list_tag_keys(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tag_keys.TagKey) for i in results))

def test_list_tag_keys_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_tag_keys), '__call__') as call:
        call.side_effect = (tag_keys.ListTagKeysResponse(tag_keys=[tag_keys.TagKey(), tag_keys.TagKey(), tag_keys.TagKey()], next_page_token='abc'), tag_keys.ListTagKeysResponse(tag_keys=[], next_page_token='def'), tag_keys.ListTagKeysResponse(tag_keys=[tag_keys.TagKey()], next_page_token='ghi'), tag_keys.ListTagKeysResponse(tag_keys=[tag_keys.TagKey(), tag_keys.TagKey()]), RuntimeError)
        pages = list(client.list_tag_keys(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_tag_keys_async_pager():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_tag_keys), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (tag_keys.ListTagKeysResponse(tag_keys=[tag_keys.TagKey(), tag_keys.TagKey(), tag_keys.TagKey()], next_page_token='abc'), tag_keys.ListTagKeysResponse(tag_keys=[], next_page_token='def'), tag_keys.ListTagKeysResponse(tag_keys=[tag_keys.TagKey()], next_page_token='ghi'), tag_keys.ListTagKeysResponse(tag_keys=[tag_keys.TagKey(), tag_keys.TagKey()]), RuntimeError)
        async_pager = await client.list_tag_keys(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, tag_keys.TagKey) for i in responses))

@pytest.mark.asyncio
async def test_list_tag_keys_async_pages():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_tag_keys), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (tag_keys.ListTagKeysResponse(tag_keys=[tag_keys.TagKey(), tag_keys.TagKey(), tag_keys.TagKey()], next_page_token='abc'), tag_keys.ListTagKeysResponse(tag_keys=[], next_page_token='def'), tag_keys.ListTagKeysResponse(tag_keys=[tag_keys.TagKey()], next_page_token='ghi'), tag_keys.ListTagKeysResponse(tag_keys=[tag_keys.TagKey(), tag_keys.TagKey()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_tag_keys(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [tag_keys.GetTagKeyRequest, dict])
def test_get_tag_key(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_tag_key), '__call__') as call:
        call.return_value = tag_keys.TagKey(name='name_value', parent='parent_value', short_name='short_name_value', namespaced_name='namespaced_name_value', description='description_value', etag='etag_value', purpose=tag_keys.Purpose.GCE_FIREWALL)
        response = client.get_tag_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_keys.GetTagKeyRequest()
    assert isinstance(response, tag_keys.TagKey)
    assert response.name == 'name_value'
    assert response.parent == 'parent_value'
    assert response.short_name == 'short_name_value'
    assert response.namespaced_name == 'namespaced_name_value'
    assert response.description == 'description_value'
    assert response.etag == 'etag_value'
    assert response.purpose == tag_keys.Purpose.GCE_FIREWALL

def test_get_tag_key_empty_call():
    if False:
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_tag_key), '__call__') as call:
        client.get_tag_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_keys.GetTagKeyRequest()

@pytest.mark.asyncio
async def test_get_tag_key_async(transport: str='grpc_asyncio', request_type=tag_keys.GetTagKeyRequest):
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_tag_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tag_keys.TagKey(name='name_value', parent='parent_value', short_name='short_name_value', namespaced_name='namespaced_name_value', description='description_value', etag='etag_value', purpose=tag_keys.Purpose.GCE_FIREWALL))
        response = await client.get_tag_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_keys.GetTagKeyRequest()
    assert isinstance(response, tag_keys.TagKey)
    assert response.name == 'name_value'
    assert response.parent == 'parent_value'
    assert response.short_name == 'short_name_value'
    assert response.namespaced_name == 'namespaced_name_value'
    assert response.description == 'description_value'
    assert response.etag == 'etag_value'
    assert response.purpose == tag_keys.Purpose.GCE_FIREWALL

@pytest.mark.asyncio
async def test_get_tag_key_async_from_dict():
    await test_get_tag_key_async(request_type=dict)

def test_get_tag_key_field_headers():
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    request = tag_keys.GetTagKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_tag_key), '__call__') as call:
        call.return_value = tag_keys.TagKey()
        client.get_tag_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_tag_key_field_headers_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tag_keys.GetTagKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_tag_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tag_keys.TagKey())
        await client.get_tag_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_tag_key_flattened():
    if False:
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_tag_key), '__call__') as call:
        call.return_value = tag_keys.TagKey()
        client.get_tag_key(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_tag_key_flattened_error():
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_tag_key(tag_keys.GetTagKeyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_tag_key_flattened_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_tag_key), '__call__') as call:
        call.return_value = tag_keys.TagKey()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tag_keys.TagKey())
        response = await client.get_tag_key(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_tag_key_flattened_error_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_tag_key(tag_keys.GetTagKeyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [tag_keys.GetNamespacedTagKeyRequest, dict])
def test_get_namespaced_tag_key(request_type, transport: str='grpc'):
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_namespaced_tag_key), '__call__') as call:
        call.return_value = tag_keys.TagKey(name='name_value', parent='parent_value', short_name='short_name_value', namespaced_name='namespaced_name_value', description='description_value', etag='etag_value', purpose=tag_keys.Purpose.GCE_FIREWALL)
        response = client.get_namespaced_tag_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_keys.GetNamespacedTagKeyRequest()
    assert isinstance(response, tag_keys.TagKey)
    assert response.name == 'name_value'
    assert response.parent == 'parent_value'
    assert response.short_name == 'short_name_value'
    assert response.namespaced_name == 'namespaced_name_value'
    assert response.description == 'description_value'
    assert response.etag == 'etag_value'
    assert response.purpose == tag_keys.Purpose.GCE_FIREWALL

def test_get_namespaced_tag_key_empty_call():
    if False:
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_namespaced_tag_key), '__call__') as call:
        client.get_namespaced_tag_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_keys.GetNamespacedTagKeyRequest()

@pytest.mark.asyncio
async def test_get_namespaced_tag_key_async(transport: str='grpc_asyncio', request_type=tag_keys.GetNamespacedTagKeyRequest):
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_namespaced_tag_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tag_keys.TagKey(name='name_value', parent='parent_value', short_name='short_name_value', namespaced_name='namespaced_name_value', description='description_value', etag='etag_value', purpose=tag_keys.Purpose.GCE_FIREWALL))
        response = await client.get_namespaced_tag_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_keys.GetNamespacedTagKeyRequest()
    assert isinstance(response, tag_keys.TagKey)
    assert response.name == 'name_value'
    assert response.parent == 'parent_value'
    assert response.short_name == 'short_name_value'
    assert response.namespaced_name == 'namespaced_name_value'
    assert response.description == 'description_value'
    assert response.etag == 'etag_value'
    assert response.purpose == tag_keys.Purpose.GCE_FIREWALL

@pytest.mark.asyncio
async def test_get_namespaced_tag_key_async_from_dict():
    await test_get_namespaced_tag_key_async(request_type=dict)

def test_get_namespaced_tag_key_flattened():
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_namespaced_tag_key), '__call__') as call:
        call.return_value = tag_keys.TagKey()
        client.get_namespaced_tag_key(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_namespaced_tag_key_flattened_error():
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_namespaced_tag_key(tag_keys.GetNamespacedTagKeyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_namespaced_tag_key_flattened_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_namespaced_tag_key), '__call__') as call:
        call.return_value = tag_keys.TagKey()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(tag_keys.TagKey())
        response = await client.get_namespaced_tag_key(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_namespaced_tag_key_flattened_error_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_namespaced_tag_key(tag_keys.GetNamespacedTagKeyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [tag_keys.CreateTagKeyRequest, dict])
def test_create_tag_key(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_tag_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_tag_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_keys.CreateTagKeyRequest()
    assert isinstance(response, future.Future)

def test_create_tag_key_empty_call():
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_tag_key), '__call__') as call:
        client.create_tag_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_keys.CreateTagKeyRequest()

@pytest.mark.asyncio
async def test_create_tag_key_async(transport: str='grpc_asyncio', request_type=tag_keys.CreateTagKeyRequest):
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_tag_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_tag_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_keys.CreateTagKeyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_tag_key_async_from_dict():
    await test_create_tag_key_async(request_type=dict)

def test_create_tag_key_flattened():
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_tag_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_tag_key(tag_key=tag_keys.TagKey(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].tag_key
        mock_val = tag_keys.TagKey(name='name_value')
        assert arg == mock_val

def test_create_tag_key_flattened_error():
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_tag_key(tag_keys.CreateTagKeyRequest(), tag_key=tag_keys.TagKey(name='name_value'))

@pytest.mark.asyncio
async def test_create_tag_key_flattened_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_tag_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_tag_key(tag_key=tag_keys.TagKey(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].tag_key
        mock_val = tag_keys.TagKey(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_tag_key_flattened_error_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_tag_key(tag_keys.CreateTagKeyRequest(), tag_key=tag_keys.TagKey(name='name_value'))

@pytest.mark.parametrize('request_type', [tag_keys.UpdateTagKeyRequest, dict])
def test_update_tag_key(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_tag_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_tag_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_keys.UpdateTagKeyRequest()
    assert isinstance(response, future.Future)

def test_update_tag_key_empty_call():
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_tag_key), '__call__') as call:
        client.update_tag_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_keys.UpdateTagKeyRequest()

@pytest.mark.asyncio
async def test_update_tag_key_async(transport: str='grpc_asyncio', request_type=tag_keys.UpdateTagKeyRequest):
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_tag_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_tag_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_keys.UpdateTagKeyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_tag_key_async_from_dict():
    await test_update_tag_key_async(request_type=dict)

def test_update_tag_key_field_headers():
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    request = tag_keys.UpdateTagKeyRequest()
    request.tag_key.name = 'name_value'
    with mock.patch.object(type(client.transport.update_tag_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_tag_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'tag_key.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_tag_key_field_headers_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tag_keys.UpdateTagKeyRequest()
    request.tag_key.name = 'name_value'
    with mock.patch.object(type(client.transport.update_tag_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_tag_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'tag_key.name=name_value') in kw['metadata']

def test_update_tag_key_flattened():
    if False:
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_tag_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_tag_key(tag_key=tag_keys.TagKey(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].tag_key
        mock_val = tag_keys.TagKey(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_tag_key_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_tag_key(tag_keys.UpdateTagKeyRequest(), tag_key=tag_keys.TagKey(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_tag_key_flattened_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_tag_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_tag_key(tag_key=tag_keys.TagKey(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].tag_key
        mock_val = tag_keys.TagKey(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_tag_key_flattened_error_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_tag_key(tag_keys.UpdateTagKeyRequest(), tag_key=tag_keys.TagKey(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [tag_keys.DeleteTagKeyRequest, dict])
def test_delete_tag_key(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_tag_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_tag_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_keys.DeleteTagKeyRequest()
    assert isinstance(response, future.Future)

def test_delete_tag_key_empty_call():
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_tag_key), '__call__') as call:
        client.delete_tag_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_keys.DeleteTagKeyRequest()

@pytest.mark.asyncio
async def test_delete_tag_key_async(transport: str='grpc_asyncio', request_type=tag_keys.DeleteTagKeyRequest):
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_tag_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_tag_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == tag_keys.DeleteTagKeyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_tag_key_async_from_dict():
    await test_delete_tag_key_async(request_type=dict)

def test_delete_tag_key_field_headers():
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    request = tag_keys.DeleteTagKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_tag_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_tag_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_tag_key_field_headers_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = tag_keys.DeleteTagKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_tag_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_tag_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_tag_key_flattened():
    if False:
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_tag_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_tag_key(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_tag_key_flattened_error():
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_tag_key(tag_keys.DeleteTagKeyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_tag_key_flattened_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_tag_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_tag_key(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_tag_key_flattened_error_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_tag_key(tag_keys.DeleteTagKeyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy(request_type, transport: str='grpc'):
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_empty_call():
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        client.get_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()

@pytest.mark.asyncio
async def test_get_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.GetIamPolicyRequest):
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_get_iam_policy_async_from_dict():
    await test_get_iam_policy_async(request_type=dict)

def test_get_iam_policy_field_headers():
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_iam_policy_field_headers_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_get_iam_policy_from_dict_foreign():
    if False:
        for i in range(10):
            print('nop')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_get_iam_policy_flattened():
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.get_iam_policy(resource='resource_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

def test_get_iam_policy_flattened_error():
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_iam_policy(iam_policy_pb2.GetIamPolicyRequest(), resource='resource_value')

@pytest.mark.asyncio
async def test_get_iam_policy_flattened_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(resource='resource_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_iam_policy_flattened_error_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_iam_policy(iam_policy_pb2.GetIamPolicyRequest(), resource='resource_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_empty_call():
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        client.set_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()

@pytest.mark.asyncio
async def test_set_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.SetIamPolicyRequest):
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.set_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_set_iam_policy_async_from_dict():
    await test_set_iam_policy_async(request_type=dict)

def test_set_iam_policy_field_headers():
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_set_iam_policy_field_headers_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.set_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_set_iam_policy_from_dict_foreign():
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774), 'update_mask': field_mask_pb2.FieldMask(paths=['paths_value'])})
        call.assert_called()

def test_set_iam_policy_flattened():
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.set_iam_policy(resource='resource_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

def test_set_iam_policy_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.set_iam_policy(iam_policy_pb2.SetIamPolicyRequest(), resource='resource_value')

@pytest.mark.asyncio
async def test_set_iam_policy_flattened_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(resource='resource_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_set_iam_policy_flattened_error_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.set_iam_policy(iam_policy_pb2.SetIamPolicyRequest(), resource='resource_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions(request_type, transport: str='grpc'):
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response = client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_empty_call():
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        client.test_iam_permissions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()

@pytest.mark.asyncio
async def test_test_iam_permissions_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value']))
        response = await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

@pytest.mark.asyncio
async def test_test_iam_permissions_async_from_dict():
    await test_test_iam_permissions_async(request_type=dict)

def test_test_iam_permissions_field_headers():
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

@pytest.mark.asyncio
async def test_test_iam_permissions_field_headers_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource_value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource_value') in kw['metadata']

def test_test_iam_permissions_from_dict_foreign():
    if False:
        for i in range(10):
            print('nop')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_test_iam_permissions_flattened():
    if False:
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(resource='resource_value', permissions=['permissions_value'])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val
        arg = args[0].permissions
        mock_val = ['permissions_value']
        assert arg == mock_val

def test_test_iam_permissions_flattened_error():
    if False:
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.test_iam_permissions(iam_policy_pb2.TestIamPermissionsRequest(), resource='resource_value', permissions=['permissions_value'])

@pytest.mark.asyncio
async def test_test_iam_permissions_flattened_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(resource='resource_value', permissions=['permissions_value'])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val
        arg = args[0].permissions
        mock_val = ['permissions_value']
        assert arg == mock_val

@pytest.mark.asyncio
async def test_test_iam_permissions_flattened_error_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.test_iam_permissions(iam_policy_pb2.TestIamPermissionsRequest(), resource='resource_value', permissions=['permissions_value'])

@pytest.mark.parametrize('request_type', [tag_keys.ListTagKeysRequest, dict])
def test_list_tag_keys_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tag_keys.ListTagKeysResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = tag_keys.ListTagKeysResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_tag_keys(request)
    assert isinstance(response, pagers.ListTagKeysPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_tag_keys_rest_required_fields(request_type=tag_keys.ListTagKeysRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.TagKeysRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'parent' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_tag_keys._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == request_init['parent']
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_tag_keys._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token', 'parent'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tag_keys.ListTagKeysResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tag_keys.ListTagKeysResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_tag_keys(request)
            expected_params = [('parent', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_tag_keys_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.TagKeysRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_tag_keys._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken', 'parent')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_tag_keys_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.TagKeysRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TagKeysRestInterceptor())
    client = TagKeysClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TagKeysRestInterceptor, 'post_list_tag_keys') as post, mock.patch.object(transports.TagKeysRestInterceptor, 'pre_list_tag_keys') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tag_keys.ListTagKeysRequest.pb(tag_keys.ListTagKeysRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tag_keys.ListTagKeysResponse.to_json(tag_keys.ListTagKeysResponse())
        request = tag_keys.ListTagKeysRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tag_keys.ListTagKeysResponse()
        client.list_tag_keys(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_tag_keys_rest_bad_request(transport: str='rest', request_type=tag_keys.ListTagKeysRequest):
    if False:
        for i in range(10):
            print('nop')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_tag_keys(request)

def test_list_tag_keys_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tag_keys.ListTagKeysResponse()
        sample_request = {}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = tag_keys.ListTagKeysResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_tag_keys(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/tagKeys' % client.transport._host, args[1])

def test_list_tag_keys_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_tag_keys(tag_keys.ListTagKeysRequest(), parent='parent_value')

def test_list_tag_keys_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (tag_keys.ListTagKeysResponse(tag_keys=[tag_keys.TagKey(), tag_keys.TagKey(), tag_keys.TagKey()], next_page_token='abc'), tag_keys.ListTagKeysResponse(tag_keys=[], next_page_token='def'), tag_keys.ListTagKeysResponse(tag_keys=[tag_keys.TagKey()], next_page_token='ghi'), tag_keys.ListTagKeysResponse(tag_keys=[tag_keys.TagKey(), tag_keys.TagKey()]))
        response = response + response
        response = tuple((tag_keys.ListTagKeysResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {}
        pager = client.list_tag_keys(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tag_keys.TagKey) for i in results))
        pages = list(client.list_tag_keys(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [tag_keys.GetTagKeyRequest, dict])
def test_get_tag_key_rest(request_type):
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'tagKeys/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tag_keys.TagKey(name='name_value', parent='parent_value', short_name='short_name_value', namespaced_name='namespaced_name_value', description='description_value', etag='etag_value', purpose=tag_keys.Purpose.GCE_FIREWALL)
        response_value = Response()
        response_value.status_code = 200
        return_value = tag_keys.TagKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_tag_key(request)
    assert isinstance(response, tag_keys.TagKey)
    assert response.name == 'name_value'
    assert response.parent == 'parent_value'
    assert response.short_name == 'short_name_value'
    assert response.namespaced_name == 'namespaced_name_value'
    assert response.description == 'description_value'
    assert response.etag == 'etag_value'
    assert response.purpose == tag_keys.Purpose.GCE_FIREWALL

def test_get_tag_key_rest_required_fields(request_type=tag_keys.GetTagKeyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.TagKeysRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_tag_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_tag_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tag_keys.TagKey()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tag_keys.TagKey.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_tag_key(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_tag_key_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.TagKeysRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_tag_key._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_tag_key_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.TagKeysRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TagKeysRestInterceptor())
    client = TagKeysClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TagKeysRestInterceptor, 'post_get_tag_key') as post, mock.patch.object(transports.TagKeysRestInterceptor, 'pre_get_tag_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tag_keys.GetTagKeyRequest.pb(tag_keys.GetTagKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tag_keys.TagKey.to_json(tag_keys.TagKey())
        request = tag_keys.GetTagKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tag_keys.TagKey()
        client.get_tag_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_tag_key_rest_bad_request(transport: str='rest', request_type=tag_keys.GetTagKeyRequest):
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'tagKeys/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_tag_key(request)

def test_get_tag_key_rest_flattened():
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tag_keys.TagKey()
        sample_request = {'name': 'tagKeys/sample1'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = tag_keys.TagKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_tag_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/{name=tagKeys/*}' % client.transport._host, args[1])

def test_get_tag_key_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_tag_key(tag_keys.GetTagKeyRequest(), name='name_value')

def test_get_tag_key_rest_error():
    if False:
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [tag_keys.GetNamespacedTagKeyRequest, dict])
def test_get_namespaced_tag_key_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tag_keys.TagKey(name='name_value', parent='parent_value', short_name='short_name_value', namespaced_name='namespaced_name_value', description='description_value', etag='etag_value', purpose=tag_keys.Purpose.GCE_FIREWALL)
        response_value = Response()
        response_value.status_code = 200
        return_value = tag_keys.TagKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_namespaced_tag_key(request)
    assert isinstance(response, tag_keys.TagKey)
    assert response.name == 'name_value'
    assert response.parent == 'parent_value'
    assert response.short_name == 'short_name_value'
    assert response.namespaced_name == 'namespaced_name_value'
    assert response.description == 'description_value'
    assert response.etag == 'etag_value'
    assert response.purpose == tag_keys.Purpose.GCE_FIREWALL

def test_get_namespaced_tag_key_rest_required_fields(request_type=tag_keys.GetNamespacedTagKeyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.TagKeysRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'name' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_namespaced_tag_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == request_init['name']
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_namespaced_tag_key._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('name',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = tag_keys.TagKey()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = tag_keys.TagKey.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_namespaced_tag_key(request)
            expected_params = [('name', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_namespaced_tag_key_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.TagKeysRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_namespaced_tag_key._get_unset_required_fields({})
    assert set(unset_fields) == set(('name',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_namespaced_tag_key_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.TagKeysRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TagKeysRestInterceptor())
    client = TagKeysClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TagKeysRestInterceptor, 'post_get_namespaced_tag_key') as post, mock.patch.object(transports.TagKeysRestInterceptor, 'pre_get_namespaced_tag_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tag_keys.GetNamespacedTagKeyRequest.pb(tag_keys.GetNamespacedTagKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = tag_keys.TagKey.to_json(tag_keys.TagKey())
        request = tag_keys.GetNamespacedTagKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = tag_keys.TagKey()
        client.get_namespaced_tag_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_namespaced_tag_key_rest_bad_request(transport: str='rest', request_type=tag_keys.GetNamespacedTagKeyRequest):
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_namespaced_tag_key(request)

def test_get_namespaced_tag_key_rest_flattened():
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = tag_keys.TagKey()
        sample_request = {}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = tag_keys.TagKey.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_namespaced_tag_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/tagKeys/namespaced' % client.transport._host, args[1])

def test_get_namespaced_tag_key_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_namespaced_tag_key(tag_keys.GetNamespacedTagKeyRequest(), name='name_value')

def test_get_namespaced_tag_key_rest_error():
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [tag_keys.CreateTagKeyRequest, dict])
def test_create_tag_key_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request_init['tag_key'] = {'name': 'name_value', 'parent': 'parent_value', 'short_name': 'short_name_value', 'namespaced_name': 'namespaced_name_value', 'description': 'description_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'etag': 'etag_value', 'purpose': 1, 'purpose_data': {}}
    test_field = tag_keys.CreateTagKeyRequest.meta.fields['tag_key']

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
    for (field, value) in request_init['tag_key'].items():
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
                for i in range(0, len(request_init['tag_key'][field])):
                    del request_init['tag_key'][field][i][subfield]
            else:
                del request_init['tag_key'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_tag_key(request)
    assert response.operation.name == 'operations/spam'

def test_create_tag_key_rest_required_fields(request_type=tag_keys.CreateTagKeyRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.TagKeysRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_tag_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_tag_key._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('validate_only',))
    jsonified_request.update(unset_fields)
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_tag_key(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_tag_key_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.TagKeysRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_tag_key._get_unset_required_fields({})
    assert set(unset_fields) == set(('validateOnly',)) & set(('tagKey',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_tag_key_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.TagKeysRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TagKeysRestInterceptor())
    client = TagKeysClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.TagKeysRestInterceptor, 'post_create_tag_key') as post, mock.patch.object(transports.TagKeysRestInterceptor, 'pre_create_tag_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tag_keys.CreateTagKeyRequest.pb(tag_keys.CreateTagKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = tag_keys.CreateTagKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_tag_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_tag_key_rest_bad_request(transport: str='rest', request_type=tag_keys.CreateTagKeyRequest):
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_tag_key(request)

def test_create_tag_key_rest_flattened():
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {}
        mock_args = dict(tag_key=tag_keys.TagKey(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_tag_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/tagKeys' % client.transport._host, args[1])

def test_create_tag_key_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_tag_key(tag_keys.CreateTagKeyRequest(), tag_key=tag_keys.TagKey(name='name_value'))

def test_create_tag_key_rest_error():
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [tag_keys.UpdateTagKeyRequest, dict])
def test_update_tag_key_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'tag_key': {'name': 'tagKeys/sample1'}}
    request_init['tag_key'] = {'name': 'tagKeys/sample1', 'parent': 'parent_value', 'short_name': 'short_name_value', 'namespaced_name': 'namespaced_name_value', 'description': 'description_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'etag': 'etag_value', 'purpose': 1, 'purpose_data': {}}
    test_field = tag_keys.UpdateTagKeyRequest.meta.fields['tag_key']

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
    for (field, value) in request_init['tag_key'].items():
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
                for i in range(0, len(request_init['tag_key'][field])):
                    del request_init['tag_key'][field][i][subfield]
            else:
                del request_init['tag_key'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_tag_key(request)
    assert response.operation.name == 'operations/spam'

def test_update_tag_key_rest_required_fields(request_type=tag_keys.UpdateTagKeyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.TagKeysRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_tag_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_tag_key._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_tag_key(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_tag_key_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.TagKeysRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_tag_key._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask', 'validateOnly')) & set(('tagKey',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_tag_key_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.TagKeysRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TagKeysRestInterceptor())
    client = TagKeysClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.TagKeysRestInterceptor, 'post_update_tag_key') as post, mock.patch.object(transports.TagKeysRestInterceptor, 'pre_update_tag_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tag_keys.UpdateTagKeyRequest.pb(tag_keys.UpdateTagKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = tag_keys.UpdateTagKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_tag_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_tag_key_rest_bad_request(transport: str='rest', request_type=tag_keys.UpdateTagKeyRequest):
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'tag_key': {'name': 'tagKeys/sample1'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_tag_key(request)

def test_update_tag_key_rest_flattened():
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'tag_key': {'name': 'tagKeys/sample1'}}
        mock_args = dict(tag_key=tag_keys.TagKey(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_tag_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/{tag_key.name=tagKeys/*}' % client.transport._host, args[1])

def test_update_tag_key_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_tag_key(tag_keys.UpdateTagKeyRequest(), tag_key=tag_keys.TagKey(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_tag_key_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [tag_keys.DeleteTagKeyRequest, dict])
def test_delete_tag_key_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'tagKeys/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_tag_key(request)
    assert response.operation.name == 'operations/spam'

def test_delete_tag_key_rest_required_fields(request_type=tag_keys.DeleteTagKeyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.TagKeysRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_tag_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_tag_key._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('etag', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_tag_key(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_tag_key_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.TagKeysRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_tag_key._get_unset_required_fields({})
    assert set(unset_fields) == set(('etag', 'validateOnly')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_tag_key_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.TagKeysRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TagKeysRestInterceptor())
    client = TagKeysClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.TagKeysRestInterceptor, 'post_delete_tag_key') as post, mock.patch.object(transports.TagKeysRestInterceptor, 'pre_delete_tag_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = tag_keys.DeleteTagKeyRequest.pb(tag_keys.DeleteTagKeyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = tag_keys.DeleteTagKeyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_tag_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_tag_key_rest_bad_request(transport: str='rest', request_type=tag_keys.DeleteTagKeyRequest):
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'tagKeys/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_tag_key(request)

def test_delete_tag_key_rest_flattened():
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'tagKeys/sample1'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_tag_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/{name=tagKeys/*}' % client.transport._host, args[1])

def test_delete_tag_key_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_tag_key(tag_keys.DeleteTagKeyRequest(), name='name_value')

def test_delete_tag_key_rest_error():
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'tagKeys/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_rest_required_fields(request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.TagKeysRestTransport
    request_init = {}
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = policy_pb2.Policy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_iam_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_iam_policy_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.TagKeysRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_iam_policy_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.TagKeysRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TagKeysRestInterceptor())
    client = TagKeysClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TagKeysRestInterceptor, 'post_get_iam_policy') as post, mock.patch.object(transports.TagKeysRestInterceptor, 'pre_get_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.GetIamPolicyRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(policy_pb2.Policy())
        request = iam_policy_pb2.GetIamPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = policy_pb2.Policy()
        client.get_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'tagKeys/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

def test_get_iam_policy_rest_flattened():
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        sample_request = {'resource': 'tagKeys/sample1'}
        mock_args = dict(resource='resource_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_iam_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/{resource=tagKeys/*}:getIamPolicy' % client.transport._host, args[1])

def test_get_iam_policy_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_iam_policy(iam_policy_pb2.GetIamPolicyRequest(), resource='resource_value')

def test_get_iam_policy_rest_error():
    if False:
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'tagKeys/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_rest_required_fields(request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.TagKeysRestTransport
    request_init = {}
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = policy_pb2.Policy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.set_iam_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_iam_policy_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TagKeysRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'policy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_iam_policy_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TagKeysRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TagKeysRestInterceptor())
    client = TagKeysClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TagKeysRestInterceptor, 'post_set_iam_policy') as post, mock.patch.object(transports.TagKeysRestInterceptor, 'pre_set_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.SetIamPolicyRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(policy_pb2.Policy())
        request = iam_policy_pb2.SetIamPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = policy_pb2.Policy()
        client.set_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'tagKeys/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

def test_set_iam_policy_rest_flattened():
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        sample_request = {'resource': 'tagKeys/sample1'}
        mock_args = dict(resource='resource_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_iam_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/{resource=tagKeys/*}:setIamPolicy' % client.transport._host, args[1])

def test_set_iam_policy_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_iam_policy(iam_policy_pb2.SetIamPolicyRequest(), resource='resource_value')

def test_set_iam_policy_rest_error():
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'tagKeys/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.test_iam_permissions(request)
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_rest_required_fields(request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        return 10
    transport_class = transports.TagKeysRestTransport
    request_init = {}
    request_init['resource'] = ''
    request_init['permissions'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    jsonified_request['permissions'] = 'permissions_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    assert 'permissions' in jsonified_request
    assert jsonified_request['permissions'] == 'permissions_value'
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = iam_policy_pb2.TestIamPermissionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.test_iam_permissions(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_test_iam_permissions_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.TagKeysRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.test_iam_permissions._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'permissions'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_test_iam_permissions_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.TagKeysRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TagKeysRestInterceptor())
    client = TagKeysClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TagKeysRestInterceptor, 'post_test_iam_permissions') as post, mock.patch.object(transports.TagKeysRestInterceptor, 'pre_test_iam_permissions') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.TestIamPermissionsRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(iam_policy_pb2.TestIamPermissionsResponse())
        request = iam_policy_pb2.TestIamPermissionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_test_iam_permissions_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'tagKeys/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

def test_test_iam_permissions_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse()
        sample_request = {'resource': 'tagKeys/sample1'}
        mock_args = dict(resource='resource_value', permissions=['permissions_value'])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.test_iam_permissions(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3/{resource=tagKeys/*}:testIamPermissions' % client.transport._host, args[1])

def test_test_iam_permissions_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.test_iam_permissions(iam_policy_pb2.TestIamPermissionsRequest(), resource='resource_value', permissions=['permissions_value'])

def test_test_iam_permissions_rest_error():
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.TagKeysGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.TagKeysGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TagKeysClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.TagKeysGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = TagKeysClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = TagKeysClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.TagKeysGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TagKeysClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.TagKeysGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = TagKeysClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.TagKeysGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.TagKeysGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.TagKeysGrpcTransport, transports.TagKeysGrpcAsyncIOTransport, transports.TagKeysRestTransport])
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
    transport = TagKeysClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.TagKeysGrpcTransport)

def test_tag_keys_base_transport_error():
    if False:
        return 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.TagKeysTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_tag_keys_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.resourcemanager_v3.services.tag_keys.transports.TagKeysTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.TagKeysTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_tag_keys', 'get_tag_key', 'get_namespaced_tag_key', 'create_tag_key', 'update_tag_key', 'delete_tag_key', 'get_iam_policy', 'set_iam_policy', 'test_iam_permissions', 'get_operation')
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

def test_tag_keys_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.resourcemanager_v3.services.tag_keys.transports.TagKeysTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.TagKeysTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), quota_project_id='octopus')

def test_tag_keys_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.resourcemanager_v3.services.tag_keys.transports.TagKeysTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.TagKeysTransport()
        adc.assert_called_once()

def test_tag_keys_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        TagKeysClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.TagKeysGrpcTransport, transports.TagKeysGrpcAsyncIOTransport])
def test_tag_keys_transport_auth_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.TagKeysGrpcTransport, transports.TagKeysGrpcAsyncIOTransport, transports.TagKeysRestTransport])
def test_tag_keys_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.TagKeysGrpcTransport, grpc_helpers), (transports.TagKeysGrpcAsyncIOTransport, grpc_helpers_async)])
def test_tag_keys_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('cloudresourcemanager.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only'), scopes=['1', '2'], default_host='cloudresourcemanager.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.TagKeysGrpcTransport, transports.TagKeysGrpcAsyncIOTransport])
def test_tag_keys_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_tag_keys_http_transport_client_cert_source_for_mtls():
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.TagKeysRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_tag_keys_rest_lro_client():
    if False:
        while True:
            i = 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_tag_keys_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudresourcemanager.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('cloudresourcemanager.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudresourcemanager.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_tag_keys_host_with_port(transport_name):
    if False:
        print('Hello World!')
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudresourcemanager.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('cloudresourcemanager.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudresourcemanager.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_tag_keys_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = TagKeysClient(credentials=creds1, transport=transport_name)
    client2 = TagKeysClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_tag_keys._session
    session2 = client2.transport.list_tag_keys._session
    assert session1 != session2
    session1 = client1.transport.get_tag_key._session
    session2 = client2.transport.get_tag_key._session
    assert session1 != session2
    session1 = client1.transport.get_namespaced_tag_key._session
    session2 = client2.transport.get_namespaced_tag_key._session
    assert session1 != session2
    session1 = client1.transport.create_tag_key._session
    session2 = client2.transport.create_tag_key._session
    assert session1 != session2
    session1 = client1.transport.update_tag_key._session
    session2 = client2.transport.update_tag_key._session
    assert session1 != session2
    session1 = client1.transport.delete_tag_key._session
    session2 = client2.transport.delete_tag_key._session
    assert session1 != session2
    session1 = client1.transport.get_iam_policy._session
    session2 = client2.transport.get_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.set_iam_policy._session
    session2 = client2.transport.set_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.test_iam_permissions._session
    session2 = client2.transport.test_iam_permissions._session
    assert session1 != session2

def test_tag_keys_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.TagKeysGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_tag_keys_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.TagKeysGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.TagKeysGrpcTransport, transports.TagKeysGrpcAsyncIOTransport])
def test_tag_keys_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.TagKeysGrpcTransport, transports.TagKeysGrpcAsyncIOTransport])
def test_tag_keys_transport_channel_mtls_with_adc(transport_class):
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

def test_tag_keys_grpc_lro_client():
    if False:
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_tag_keys_grpc_lro_async_client():
    if False:
        i = 10
        return i + 15
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_tag_key_path():
    if False:
        print('Hello World!')
    tag_key = 'squid'
    expected = 'tagKeys/{tag_key}'.format(tag_key=tag_key)
    actual = TagKeysClient.tag_key_path(tag_key)
    assert expected == actual

def test_parse_tag_key_path():
    if False:
        return 10
    expected = {'tag_key': 'clam'}
    path = TagKeysClient.tag_key_path(**expected)
    actual = TagKeysClient.parse_tag_key_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = TagKeysClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'octopus'}
    path = TagKeysClient.common_billing_account_path(**expected)
    actual = TagKeysClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = TagKeysClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'nudibranch'}
    path = TagKeysClient.common_folder_path(**expected)
    actual = TagKeysClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = TagKeysClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'mussel'}
    path = TagKeysClient.common_organization_path(**expected)
    actual = TagKeysClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = TagKeysClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nautilus'}
    path = TagKeysClient.common_project_path(**expected)
    actual = TagKeysClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = TagKeysClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'squid', 'location': 'clam'}
    path = TagKeysClient.common_location_path(**expected)
    actual = TagKeysClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.TagKeysTransport, '_prep_wrapped_messages') as prep:
        client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.TagKeysTransport, '_prep_wrapped_messages') as prep:
        transport_class = TagKeysClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = TagKeysAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = TagKeysClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(TagKeysClient, transports.TagKeysGrpcTransport), (TagKeysAsyncClient, transports.TagKeysGrpcAsyncIOTransport)])
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