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
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import struct_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.dialogflowcx_v3beta1.services.transition_route_groups import TransitionRouteGroupsAsyncClient, TransitionRouteGroupsClient, pagers, transports
from google.cloud.dialogflowcx_v3beta1.types import advanced_settings, fulfillment, gcs, page, response_message
from google.cloud.dialogflowcx_v3beta1.types import transition_route_group as gcdc_transition_route_group
from google.cloud.dialogflowcx_v3beta1.types import transition_route_group

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
        print('Hello World!')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert TransitionRouteGroupsClient._get_default_mtls_endpoint(None) is None
    assert TransitionRouteGroupsClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert TransitionRouteGroupsClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert TransitionRouteGroupsClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert TransitionRouteGroupsClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert TransitionRouteGroupsClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(TransitionRouteGroupsClient, 'grpc'), (TransitionRouteGroupsAsyncClient, 'grpc_asyncio'), (TransitionRouteGroupsClient, 'rest')])
def test_transition_route_groups_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.TransitionRouteGroupsGrpcTransport, 'grpc'), (transports.TransitionRouteGroupsGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.TransitionRouteGroupsRestTransport, 'rest')])
def test_transition_route_groups_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(TransitionRouteGroupsClient, 'grpc'), (TransitionRouteGroupsAsyncClient, 'grpc_asyncio'), (TransitionRouteGroupsClient, 'rest')])
def test_transition_route_groups_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

def test_transition_route_groups_client_get_transport_class():
    if False:
        return 10
    transport = TransitionRouteGroupsClient.get_transport_class()
    available_transports = [transports.TransitionRouteGroupsGrpcTransport, transports.TransitionRouteGroupsRestTransport]
    assert transport in available_transports
    transport = TransitionRouteGroupsClient.get_transport_class('grpc')
    assert transport == transports.TransitionRouteGroupsGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(TransitionRouteGroupsClient, transports.TransitionRouteGroupsGrpcTransport, 'grpc'), (TransitionRouteGroupsAsyncClient, transports.TransitionRouteGroupsGrpcAsyncIOTransport, 'grpc_asyncio'), (TransitionRouteGroupsClient, transports.TransitionRouteGroupsRestTransport, 'rest')])
@mock.patch.object(TransitionRouteGroupsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TransitionRouteGroupsClient))
@mock.patch.object(TransitionRouteGroupsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TransitionRouteGroupsAsyncClient))
def test_transition_route_groups_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(TransitionRouteGroupsClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(TransitionRouteGroupsClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(TransitionRouteGroupsClient, transports.TransitionRouteGroupsGrpcTransport, 'grpc', 'true'), (TransitionRouteGroupsAsyncClient, transports.TransitionRouteGroupsGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (TransitionRouteGroupsClient, transports.TransitionRouteGroupsGrpcTransport, 'grpc', 'false'), (TransitionRouteGroupsAsyncClient, transports.TransitionRouteGroupsGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (TransitionRouteGroupsClient, transports.TransitionRouteGroupsRestTransport, 'rest', 'true'), (TransitionRouteGroupsClient, transports.TransitionRouteGroupsRestTransport, 'rest', 'false')])
@mock.patch.object(TransitionRouteGroupsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TransitionRouteGroupsClient))
@mock.patch.object(TransitionRouteGroupsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TransitionRouteGroupsAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_transition_route_groups_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [TransitionRouteGroupsClient, TransitionRouteGroupsAsyncClient])
@mock.patch.object(TransitionRouteGroupsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TransitionRouteGroupsClient))
@mock.patch.object(TransitionRouteGroupsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(TransitionRouteGroupsAsyncClient))
def test_transition_route_groups_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(TransitionRouteGroupsClient, transports.TransitionRouteGroupsGrpcTransport, 'grpc'), (TransitionRouteGroupsAsyncClient, transports.TransitionRouteGroupsGrpcAsyncIOTransport, 'grpc_asyncio'), (TransitionRouteGroupsClient, transports.TransitionRouteGroupsRestTransport, 'rest')])
def test_transition_route_groups_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(TransitionRouteGroupsClient, transports.TransitionRouteGroupsGrpcTransport, 'grpc', grpc_helpers), (TransitionRouteGroupsAsyncClient, transports.TransitionRouteGroupsGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (TransitionRouteGroupsClient, transports.TransitionRouteGroupsRestTransport, 'rest', None)])
def test_transition_route_groups_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_transition_route_groups_client_client_options_from_dict():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.dialogflowcx_v3beta1.services.transition_route_groups.transports.TransitionRouteGroupsGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = TransitionRouteGroupsClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(TransitionRouteGroupsClient, transports.TransitionRouteGroupsGrpcTransport, 'grpc', grpc_helpers), (TransitionRouteGroupsAsyncClient, transports.TransitionRouteGroupsGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_transition_route_groups_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('dialogflow.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), scopes=None, default_host='dialogflow.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [transition_route_group.ListTransitionRouteGroupsRequest, dict])
def test_list_transition_route_groups(request_type, transport: str='grpc'):
    if False:
        return 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_transition_route_groups), '__call__') as call:
        call.return_value = transition_route_group.ListTransitionRouteGroupsResponse(next_page_token='next_page_token_value')
        response = client.list_transition_route_groups(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transition_route_group.ListTransitionRouteGroupsRequest()
    assert isinstance(response, pagers.ListTransitionRouteGroupsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_transition_route_groups_empty_call():
    if False:
        return 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_transition_route_groups), '__call__') as call:
        client.list_transition_route_groups()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transition_route_group.ListTransitionRouteGroupsRequest()

@pytest.mark.asyncio
async def test_list_transition_route_groups_async(transport: str='grpc_asyncio', request_type=transition_route_group.ListTransitionRouteGroupsRequest):
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_transition_route_groups), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transition_route_group.ListTransitionRouteGroupsResponse(next_page_token='next_page_token_value'))
        response = await client.list_transition_route_groups(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transition_route_group.ListTransitionRouteGroupsRequest()
    assert isinstance(response, pagers.ListTransitionRouteGroupsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_transition_route_groups_async_from_dict():
    await test_list_transition_route_groups_async(request_type=dict)

def test_list_transition_route_groups_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    request = transition_route_group.ListTransitionRouteGroupsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_transition_route_groups), '__call__') as call:
        call.return_value = transition_route_group.ListTransitionRouteGroupsResponse()
        client.list_transition_route_groups(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_transition_route_groups_field_headers_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = transition_route_group.ListTransitionRouteGroupsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_transition_route_groups), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transition_route_group.ListTransitionRouteGroupsResponse())
        await client.list_transition_route_groups(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_transition_route_groups_flattened():
    if False:
        while True:
            i = 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_transition_route_groups), '__call__') as call:
        call.return_value = transition_route_group.ListTransitionRouteGroupsResponse()
        client.list_transition_route_groups(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_transition_route_groups_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_transition_route_groups(transition_route_group.ListTransitionRouteGroupsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_transition_route_groups_flattened_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_transition_route_groups), '__call__') as call:
        call.return_value = transition_route_group.ListTransitionRouteGroupsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transition_route_group.ListTransitionRouteGroupsResponse())
        response = await client.list_transition_route_groups(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_transition_route_groups_flattened_error_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_transition_route_groups(transition_route_group.ListTransitionRouteGroupsRequest(), parent='parent_value')

def test_list_transition_route_groups_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_transition_route_groups), '__call__') as call:
        call.side_effect = (transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[transition_route_group.TransitionRouteGroup(), transition_route_group.TransitionRouteGroup(), transition_route_group.TransitionRouteGroup()], next_page_token='abc'), transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[], next_page_token='def'), transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[transition_route_group.TransitionRouteGroup()], next_page_token='ghi'), transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[transition_route_group.TransitionRouteGroup(), transition_route_group.TransitionRouteGroup()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_transition_route_groups(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, transition_route_group.TransitionRouteGroup) for i in results))

def test_list_transition_route_groups_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_transition_route_groups), '__call__') as call:
        call.side_effect = (transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[transition_route_group.TransitionRouteGroup(), transition_route_group.TransitionRouteGroup(), transition_route_group.TransitionRouteGroup()], next_page_token='abc'), transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[], next_page_token='def'), transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[transition_route_group.TransitionRouteGroup()], next_page_token='ghi'), transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[transition_route_group.TransitionRouteGroup(), transition_route_group.TransitionRouteGroup()]), RuntimeError)
        pages = list(client.list_transition_route_groups(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_transition_route_groups_async_pager():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_transition_route_groups), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[transition_route_group.TransitionRouteGroup(), transition_route_group.TransitionRouteGroup(), transition_route_group.TransitionRouteGroup()], next_page_token='abc'), transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[], next_page_token='def'), transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[transition_route_group.TransitionRouteGroup()], next_page_token='ghi'), transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[transition_route_group.TransitionRouteGroup(), transition_route_group.TransitionRouteGroup()]), RuntimeError)
        async_pager = await client.list_transition_route_groups(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, transition_route_group.TransitionRouteGroup) for i in responses))

@pytest.mark.asyncio
async def test_list_transition_route_groups_async_pages():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_transition_route_groups), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[transition_route_group.TransitionRouteGroup(), transition_route_group.TransitionRouteGroup(), transition_route_group.TransitionRouteGroup()], next_page_token='abc'), transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[], next_page_token='def'), transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[transition_route_group.TransitionRouteGroup()], next_page_token='ghi'), transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[transition_route_group.TransitionRouteGroup(), transition_route_group.TransitionRouteGroup()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_transition_route_groups(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [transition_route_group.GetTransitionRouteGroupRequest, dict])
def test_get_transition_route_group(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_transition_route_group), '__call__') as call:
        call.return_value = transition_route_group.TransitionRouteGroup(name='name_value', display_name='display_name_value')
        response = client.get_transition_route_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transition_route_group.GetTransitionRouteGroupRequest()
    assert isinstance(response, transition_route_group.TransitionRouteGroup)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_get_transition_route_group_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_transition_route_group), '__call__') as call:
        client.get_transition_route_group()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transition_route_group.GetTransitionRouteGroupRequest()

@pytest.mark.asyncio
async def test_get_transition_route_group_async(transport: str='grpc_asyncio', request_type=transition_route_group.GetTransitionRouteGroupRequest):
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_transition_route_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transition_route_group.TransitionRouteGroup(name='name_value', display_name='display_name_value'))
        response = await client.get_transition_route_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transition_route_group.GetTransitionRouteGroupRequest()
    assert isinstance(response, transition_route_group.TransitionRouteGroup)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

@pytest.mark.asyncio
async def test_get_transition_route_group_async_from_dict():
    await test_get_transition_route_group_async(request_type=dict)

def test_get_transition_route_group_field_headers():
    if False:
        print('Hello World!')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    request = transition_route_group.GetTransitionRouteGroupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_transition_route_group), '__call__') as call:
        call.return_value = transition_route_group.TransitionRouteGroup()
        client.get_transition_route_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_transition_route_group_field_headers_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = transition_route_group.GetTransitionRouteGroupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_transition_route_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transition_route_group.TransitionRouteGroup())
        await client.get_transition_route_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_transition_route_group_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_transition_route_group), '__call__') as call:
        call.return_value = transition_route_group.TransitionRouteGroup()
        client.get_transition_route_group(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_transition_route_group_flattened_error():
    if False:
        while True:
            i = 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_transition_route_group(transition_route_group.GetTransitionRouteGroupRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_transition_route_group_flattened_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_transition_route_group), '__call__') as call:
        call.return_value = transition_route_group.TransitionRouteGroup()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(transition_route_group.TransitionRouteGroup())
        response = await client.get_transition_route_group(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_transition_route_group_flattened_error_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_transition_route_group(transition_route_group.GetTransitionRouteGroupRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gcdc_transition_route_group.CreateTransitionRouteGroupRequest, dict])
def test_create_transition_route_group(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_transition_route_group), '__call__') as call:
        call.return_value = gcdc_transition_route_group.TransitionRouteGroup(name='name_value', display_name='display_name_value')
        response = client.create_transition_route_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcdc_transition_route_group.CreateTransitionRouteGroupRequest()
    assert isinstance(response, gcdc_transition_route_group.TransitionRouteGroup)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_create_transition_route_group_empty_call():
    if False:
        while True:
            i = 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_transition_route_group), '__call__') as call:
        client.create_transition_route_group()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcdc_transition_route_group.CreateTransitionRouteGroupRequest()

@pytest.mark.asyncio
async def test_create_transition_route_group_async(transport: str='grpc_asyncio', request_type=gcdc_transition_route_group.CreateTransitionRouteGroupRequest):
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_transition_route_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcdc_transition_route_group.TransitionRouteGroup(name='name_value', display_name='display_name_value'))
        response = await client.create_transition_route_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcdc_transition_route_group.CreateTransitionRouteGroupRequest()
    assert isinstance(response, gcdc_transition_route_group.TransitionRouteGroup)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

@pytest.mark.asyncio
async def test_create_transition_route_group_async_from_dict():
    await test_create_transition_route_group_async(request_type=dict)

def test_create_transition_route_group_field_headers():
    if False:
        print('Hello World!')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcdc_transition_route_group.CreateTransitionRouteGroupRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_transition_route_group), '__call__') as call:
        call.return_value = gcdc_transition_route_group.TransitionRouteGroup()
        client.create_transition_route_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_transition_route_group_field_headers_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcdc_transition_route_group.CreateTransitionRouteGroupRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_transition_route_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcdc_transition_route_group.TransitionRouteGroup())
        await client.create_transition_route_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_transition_route_group_flattened():
    if False:
        return 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_transition_route_group), '__call__') as call:
        call.return_value = gcdc_transition_route_group.TransitionRouteGroup()
        client.create_transition_route_group(parent='parent_value', transition_route_group=gcdc_transition_route_group.TransitionRouteGroup(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].transition_route_group
        mock_val = gcdc_transition_route_group.TransitionRouteGroup(name='name_value')
        assert arg == mock_val

def test_create_transition_route_group_flattened_error():
    if False:
        print('Hello World!')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_transition_route_group(gcdc_transition_route_group.CreateTransitionRouteGroupRequest(), parent='parent_value', transition_route_group=gcdc_transition_route_group.TransitionRouteGroup(name='name_value'))

@pytest.mark.asyncio
async def test_create_transition_route_group_flattened_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_transition_route_group), '__call__') as call:
        call.return_value = gcdc_transition_route_group.TransitionRouteGroup()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcdc_transition_route_group.TransitionRouteGroup())
        response = await client.create_transition_route_group(parent='parent_value', transition_route_group=gcdc_transition_route_group.TransitionRouteGroup(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].transition_route_group
        mock_val = gcdc_transition_route_group.TransitionRouteGroup(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_transition_route_group_flattened_error_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_transition_route_group(gcdc_transition_route_group.CreateTransitionRouteGroupRequest(), parent='parent_value', transition_route_group=gcdc_transition_route_group.TransitionRouteGroup(name='name_value'))

@pytest.mark.parametrize('request_type', [gcdc_transition_route_group.UpdateTransitionRouteGroupRequest, dict])
def test_update_transition_route_group(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_transition_route_group), '__call__') as call:
        call.return_value = gcdc_transition_route_group.TransitionRouteGroup(name='name_value', display_name='display_name_value')
        response = client.update_transition_route_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcdc_transition_route_group.UpdateTransitionRouteGroupRequest()
    assert isinstance(response, gcdc_transition_route_group.TransitionRouteGroup)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_update_transition_route_group_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_transition_route_group), '__call__') as call:
        client.update_transition_route_group()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcdc_transition_route_group.UpdateTransitionRouteGroupRequest()

@pytest.mark.asyncio
async def test_update_transition_route_group_async(transport: str='grpc_asyncio', request_type=gcdc_transition_route_group.UpdateTransitionRouteGroupRequest):
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_transition_route_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcdc_transition_route_group.TransitionRouteGroup(name='name_value', display_name='display_name_value'))
        response = await client.update_transition_route_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcdc_transition_route_group.UpdateTransitionRouteGroupRequest()
    assert isinstance(response, gcdc_transition_route_group.TransitionRouteGroup)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

@pytest.mark.asyncio
async def test_update_transition_route_group_async_from_dict():
    await test_update_transition_route_group_async(request_type=dict)

def test_update_transition_route_group_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcdc_transition_route_group.UpdateTransitionRouteGroupRequest()
    request.transition_route_group.name = 'name_value'
    with mock.patch.object(type(client.transport.update_transition_route_group), '__call__') as call:
        call.return_value = gcdc_transition_route_group.TransitionRouteGroup()
        client.update_transition_route_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'transition_route_group.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_transition_route_group_field_headers_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcdc_transition_route_group.UpdateTransitionRouteGroupRequest()
    request.transition_route_group.name = 'name_value'
    with mock.patch.object(type(client.transport.update_transition_route_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcdc_transition_route_group.TransitionRouteGroup())
        await client.update_transition_route_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'transition_route_group.name=name_value') in kw['metadata']

def test_update_transition_route_group_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_transition_route_group), '__call__') as call:
        call.return_value = gcdc_transition_route_group.TransitionRouteGroup()
        client.update_transition_route_group(transition_route_group=gcdc_transition_route_group.TransitionRouteGroup(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].transition_route_group
        mock_val = gcdc_transition_route_group.TransitionRouteGroup(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_transition_route_group_flattened_error():
    if False:
        print('Hello World!')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_transition_route_group(gcdc_transition_route_group.UpdateTransitionRouteGroupRequest(), transition_route_group=gcdc_transition_route_group.TransitionRouteGroup(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_transition_route_group_flattened_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_transition_route_group), '__call__') as call:
        call.return_value = gcdc_transition_route_group.TransitionRouteGroup()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcdc_transition_route_group.TransitionRouteGroup())
        response = await client.update_transition_route_group(transition_route_group=gcdc_transition_route_group.TransitionRouteGroup(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].transition_route_group
        mock_val = gcdc_transition_route_group.TransitionRouteGroup(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_transition_route_group_flattened_error_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_transition_route_group(gcdc_transition_route_group.UpdateTransitionRouteGroupRequest(), transition_route_group=gcdc_transition_route_group.TransitionRouteGroup(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [transition_route_group.DeleteTransitionRouteGroupRequest, dict])
def test_delete_transition_route_group(request_type, transport: str='grpc'):
    if False:
        return 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_transition_route_group), '__call__') as call:
        call.return_value = None
        response = client.delete_transition_route_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transition_route_group.DeleteTransitionRouteGroupRequest()
    assert response is None

def test_delete_transition_route_group_empty_call():
    if False:
        i = 10
        return i + 15
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_transition_route_group), '__call__') as call:
        client.delete_transition_route_group()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transition_route_group.DeleteTransitionRouteGroupRequest()

@pytest.mark.asyncio
async def test_delete_transition_route_group_async(transport: str='grpc_asyncio', request_type=transition_route_group.DeleteTransitionRouteGroupRequest):
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_transition_route_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_transition_route_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == transition_route_group.DeleteTransitionRouteGroupRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_transition_route_group_async_from_dict():
    await test_delete_transition_route_group_async(request_type=dict)

def test_delete_transition_route_group_field_headers():
    if False:
        print('Hello World!')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    request = transition_route_group.DeleteTransitionRouteGroupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_transition_route_group), '__call__') as call:
        call.return_value = None
        client.delete_transition_route_group(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_transition_route_group_field_headers_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = transition_route_group.DeleteTransitionRouteGroupRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_transition_route_group), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_transition_route_group(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_transition_route_group_flattened():
    if False:
        i = 10
        return i + 15
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_transition_route_group), '__call__') as call:
        call.return_value = None
        client.delete_transition_route_group(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_transition_route_group_flattened_error():
    if False:
        return 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_transition_route_group(transition_route_group.DeleteTransitionRouteGroupRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_transition_route_group_flattened_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_transition_route_group), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_transition_route_group(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_transition_route_group_flattened_error_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_transition_route_group(transition_route_group.DeleteTransitionRouteGroupRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [transition_route_group.ListTransitionRouteGroupsRequest, dict])
def test_list_transition_route_groups_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/agents/sample3/flows/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transition_route_group.ListTransitionRouteGroupsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = transition_route_group.ListTransitionRouteGroupsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_transition_route_groups(request)
    assert isinstance(response, pagers.ListTransitionRouteGroupsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_transition_route_groups_rest_required_fields(request_type=transition_route_group.ListTransitionRouteGroupsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.TransitionRouteGroupsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_transition_route_groups._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_transition_route_groups._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('language_code', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = transition_route_group.ListTransitionRouteGroupsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = transition_route_group.ListTransitionRouteGroupsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_transition_route_groups(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_transition_route_groups_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.TransitionRouteGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_transition_route_groups._get_unset_required_fields({})
    assert set(unset_fields) == set(('languageCode', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_transition_route_groups_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.TransitionRouteGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TransitionRouteGroupsRestInterceptor())
    client = TransitionRouteGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TransitionRouteGroupsRestInterceptor, 'post_list_transition_route_groups') as post, mock.patch.object(transports.TransitionRouteGroupsRestInterceptor, 'pre_list_transition_route_groups') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = transition_route_group.ListTransitionRouteGroupsRequest.pb(transition_route_group.ListTransitionRouteGroupsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = transition_route_group.ListTransitionRouteGroupsResponse.to_json(transition_route_group.ListTransitionRouteGroupsResponse())
        request = transition_route_group.ListTransitionRouteGroupsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = transition_route_group.ListTransitionRouteGroupsResponse()
        client.list_transition_route_groups(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_transition_route_groups_rest_bad_request(transport: str='rest', request_type=transition_route_group.ListTransitionRouteGroupsRequest):
    if False:
        i = 10
        return i + 15
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/agents/sample3/flows/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_transition_route_groups(request)

def test_list_transition_route_groups_rest_flattened():
    if False:
        return 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transition_route_group.ListTransitionRouteGroupsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/agents/sample3/flows/sample4'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = transition_route_group.ListTransitionRouteGroupsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_transition_route_groups(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{parent=projects/*/locations/*/agents/*/flows/*}/transitionRouteGroups' % client.transport._host, args[1])

def test_list_transition_route_groups_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_transition_route_groups(transition_route_group.ListTransitionRouteGroupsRequest(), parent='parent_value')

def test_list_transition_route_groups_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[transition_route_group.TransitionRouteGroup(), transition_route_group.TransitionRouteGroup(), transition_route_group.TransitionRouteGroup()], next_page_token='abc'), transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[], next_page_token='def'), transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[transition_route_group.TransitionRouteGroup()], next_page_token='ghi'), transition_route_group.ListTransitionRouteGroupsResponse(transition_route_groups=[transition_route_group.TransitionRouteGroup(), transition_route_group.TransitionRouteGroup()]))
        response = response + response
        response = tuple((transition_route_group.ListTransitionRouteGroupsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/agents/sample3/flows/sample4'}
        pager = client.list_transition_route_groups(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, transition_route_group.TransitionRouteGroup) for i in results))
        pages = list(client.list_transition_route_groups(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [transition_route_group.GetTransitionRouteGroupRequest, dict])
def test_get_transition_route_group_rest(request_type):
    if False:
        print('Hello World!')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/agents/sample3/flows/sample4/transitionRouteGroups/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transition_route_group.TransitionRouteGroup(name='name_value', display_name='display_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = transition_route_group.TransitionRouteGroup.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_transition_route_group(request)
    assert isinstance(response, transition_route_group.TransitionRouteGroup)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_get_transition_route_group_rest_required_fields(request_type=transition_route_group.GetTransitionRouteGroupRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.TransitionRouteGroupsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_transition_route_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_transition_route_group._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('language_code',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = transition_route_group.TransitionRouteGroup()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = transition_route_group.TransitionRouteGroup.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_transition_route_group(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_transition_route_group_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.TransitionRouteGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_transition_route_group._get_unset_required_fields({})
    assert set(unset_fields) == set(('languageCode',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_transition_route_group_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.TransitionRouteGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TransitionRouteGroupsRestInterceptor())
    client = TransitionRouteGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TransitionRouteGroupsRestInterceptor, 'post_get_transition_route_group') as post, mock.patch.object(transports.TransitionRouteGroupsRestInterceptor, 'pre_get_transition_route_group') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = transition_route_group.GetTransitionRouteGroupRequest.pb(transition_route_group.GetTransitionRouteGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = transition_route_group.TransitionRouteGroup.to_json(transition_route_group.TransitionRouteGroup())
        request = transition_route_group.GetTransitionRouteGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = transition_route_group.TransitionRouteGroup()
        client.get_transition_route_group(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_transition_route_group_rest_bad_request(transport: str='rest', request_type=transition_route_group.GetTransitionRouteGroupRequest):
    if False:
        for i in range(10):
            print('nop')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/agents/sample3/flows/sample4/transitionRouteGroups/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_transition_route_group(request)

def test_get_transition_route_group_rest_flattened():
    if False:
        print('Hello World!')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = transition_route_group.TransitionRouteGroup()
        sample_request = {'name': 'projects/sample1/locations/sample2/agents/sample3/flows/sample4/transitionRouteGroups/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = transition_route_group.TransitionRouteGroup.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_transition_route_group(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{name=projects/*/locations/*/agents/*/flows/*/transitionRouteGroups/*}' % client.transport._host, args[1])

def test_get_transition_route_group_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_transition_route_group(transition_route_group.GetTransitionRouteGroupRequest(), name='name_value')

def test_get_transition_route_group_rest_error():
    if False:
        while True:
            i = 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcdc_transition_route_group.CreateTransitionRouteGroupRequest, dict])
def test_create_transition_route_group_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/agents/sample3/flows/sample4'}
    request_init['transition_route_group'] = {'name': 'name_value', 'display_name': 'display_name_value', 'transition_routes': [{'name': 'name_value', 'description': 'description_value', 'intent': 'intent_value', 'condition': 'condition_value', 'trigger_fulfillment': {'messages': [{'text': {'text': ['text_value1', 'text_value2'], 'allow_playback_interruption': True}, 'payload': {'fields': {}}, 'conversation_success': {'metadata': {}}, 'output_audio_text': {'text': 'text_value', 'ssml': 'ssml_value', 'allow_playback_interruption': True}, 'live_agent_handoff': {'metadata': {}}, 'end_interaction': {}, 'play_audio': {'audio_uri': 'audio_uri_value', 'allow_playback_interruption': True}, 'mixed_audio': {'segments': [{'audio': b'audio_blob', 'uri': 'uri_value', 'allow_playback_interruption': True}]}, 'telephony_transfer_call': {'phone_number': 'phone_number_value'}, 'knowledge_info_card': {}, 'channel': 'channel_value'}], 'webhook': 'webhook_value', 'return_partial_responses': True, 'tag': 'tag_value', 'set_parameter_actions': [{'parameter': 'parameter_value', 'value': {'null_value': 0, 'number_value': 0.1285, 'string_value': 'string_value_value', 'bool_value': True, 'struct_value': {}, 'list_value': {'values': {}}}}], 'conditional_cases': [{'cases': [{'condition': 'condition_value', 'case_content': [{'message': {}, 'additional_cases': {}}]}]}], 'advanced_settings': {'audio_export_gcs_destination': {'uri': 'uri_value'}, 'dtmf_settings': {'enabled': True, 'max_digits': 1065, 'finish_digit': 'finish_digit_value'}, 'logging_settings': {'enable_stackdriver_logging': True, 'enable_interaction_logging': True}}, 'enable_generative_fallback': True}, 'target_page': 'target_page_value', 'target_flow': 'target_flow_value'}]}
    test_field = gcdc_transition_route_group.CreateTransitionRouteGroupRequest.meta.fields['transition_route_group']

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
    for (field, value) in request_init['transition_route_group'].items():
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
                for i in range(0, len(request_init['transition_route_group'][field])):
                    del request_init['transition_route_group'][field][i][subfield]
            else:
                del request_init['transition_route_group'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcdc_transition_route_group.TransitionRouteGroup(name='name_value', display_name='display_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcdc_transition_route_group.TransitionRouteGroup.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_transition_route_group(request)
    assert isinstance(response, gcdc_transition_route_group.TransitionRouteGroup)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_create_transition_route_group_rest_required_fields(request_type=gcdc_transition_route_group.CreateTransitionRouteGroupRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.TransitionRouteGroupsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_transition_route_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_transition_route_group._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('language_code',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcdc_transition_route_group.TransitionRouteGroup()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcdc_transition_route_group.TransitionRouteGroup.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_transition_route_group(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_transition_route_group_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TransitionRouteGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_transition_route_group._get_unset_required_fields({})
    assert set(unset_fields) == set(('languageCode',)) & set(('parent', 'transitionRouteGroup'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_transition_route_group_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TransitionRouteGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TransitionRouteGroupsRestInterceptor())
    client = TransitionRouteGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TransitionRouteGroupsRestInterceptor, 'post_create_transition_route_group') as post, mock.patch.object(transports.TransitionRouteGroupsRestInterceptor, 'pre_create_transition_route_group') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcdc_transition_route_group.CreateTransitionRouteGroupRequest.pb(gcdc_transition_route_group.CreateTransitionRouteGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcdc_transition_route_group.TransitionRouteGroup.to_json(gcdc_transition_route_group.TransitionRouteGroup())
        request = gcdc_transition_route_group.CreateTransitionRouteGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcdc_transition_route_group.TransitionRouteGroup()
        client.create_transition_route_group(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_transition_route_group_rest_bad_request(transport: str='rest', request_type=gcdc_transition_route_group.CreateTransitionRouteGroupRequest):
    if False:
        i = 10
        return i + 15
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/agents/sample3/flows/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_transition_route_group(request)

def test_create_transition_route_group_rest_flattened():
    if False:
        print('Hello World!')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcdc_transition_route_group.TransitionRouteGroup()
        sample_request = {'parent': 'projects/sample1/locations/sample2/agents/sample3/flows/sample4'}
        mock_args = dict(parent='parent_value', transition_route_group=gcdc_transition_route_group.TransitionRouteGroup(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcdc_transition_route_group.TransitionRouteGroup.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_transition_route_group(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{parent=projects/*/locations/*/agents/*/flows/*}/transitionRouteGroups' % client.transport._host, args[1])

def test_create_transition_route_group_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_transition_route_group(gcdc_transition_route_group.CreateTransitionRouteGroupRequest(), parent='parent_value', transition_route_group=gcdc_transition_route_group.TransitionRouteGroup(name='name_value'))

def test_create_transition_route_group_rest_error():
    if False:
        print('Hello World!')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gcdc_transition_route_group.UpdateTransitionRouteGroupRequest, dict])
def test_update_transition_route_group_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'transition_route_group': {'name': 'projects/sample1/locations/sample2/agents/sample3/flows/sample4/transitionRouteGroups/sample5'}}
    request_init['transition_route_group'] = {'name': 'projects/sample1/locations/sample2/agents/sample3/flows/sample4/transitionRouteGroups/sample5', 'display_name': 'display_name_value', 'transition_routes': [{'name': 'name_value', 'description': 'description_value', 'intent': 'intent_value', 'condition': 'condition_value', 'trigger_fulfillment': {'messages': [{'text': {'text': ['text_value1', 'text_value2'], 'allow_playback_interruption': True}, 'payload': {'fields': {}}, 'conversation_success': {'metadata': {}}, 'output_audio_text': {'text': 'text_value', 'ssml': 'ssml_value', 'allow_playback_interruption': True}, 'live_agent_handoff': {'metadata': {}}, 'end_interaction': {}, 'play_audio': {'audio_uri': 'audio_uri_value', 'allow_playback_interruption': True}, 'mixed_audio': {'segments': [{'audio': b'audio_blob', 'uri': 'uri_value', 'allow_playback_interruption': True}]}, 'telephony_transfer_call': {'phone_number': 'phone_number_value'}, 'knowledge_info_card': {}, 'channel': 'channel_value'}], 'webhook': 'webhook_value', 'return_partial_responses': True, 'tag': 'tag_value', 'set_parameter_actions': [{'parameter': 'parameter_value', 'value': {'null_value': 0, 'number_value': 0.1285, 'string_value': 'string_value_value', 'bool_value': True, 'struct_value': {}, 'list_value': {'values': {}}}}], 'conditional_cases': [{'cases': [{'condition': 'condition_value', 'case_content': [{'message': {}, 'additional_cases': {}}]}]}], 'advanced_settings': {'audio_export_gcs_destination': {'uri': 'uri_value'}, 'dtmf_settings': {'enabled': True, 'max_digits': 1065, 'finish_digit': 'finish_digit_value'}, 'logging_settings': {'enable_stackdriver_logging': True, 'enable_interaction_logging': True}}, 'enable_generative_fallback': True}, 'target_page': 'target_page_value', 'target_flow': 'target_flow_value'}]}
    test_field = gcdc_transition_route_group.UpdateTransitionRouteGroupRequest.meta.fields['transition_route_group']

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
    for (field, value) in request_init['transition_route_group'].items():
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
                for i in range(0, len(request_init['transition_route_group'][field])):
                    del request_init['transition_route_group'][field][i][subfield]
            else:
                del request_init['transition_route_group'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcdc_transition_route_group.TransitionRouteGroup(name='name_value', display_name='display_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcdc_transition_route_group.TransitionRouteGroup.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_transition_route_group(request)
    assert isinstance(response, gcdc_transition_route_group.TransitionRouteGroup)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_update_transition_route_group_rest_required_fields(request_type=gcdc_transition_route_group.UpdateTransitionRouteGroupRequest):
    if False:
        print('Hello World!')
    transport_class = transports.TransitionRouteGroupsRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_transition_route_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_transition_route_group._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('language_code', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcdc_transition_route_group.TransitionRouteGroup()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcdc_transition_route_group.TransitionRouteGroup.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_transition_route_group(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_transition_route_group_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.TransitionRouteGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_transition_route_group._get_unset_required_fields({})
    assert set(unset_fields) == set(('languageCode', 'updateMask')) & set(('transitionRouteGroup',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_transition_route_group_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.TransitionRouteGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TransitionRouteGroupsRestInterceptor())
    client = TransitionRouteGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TransitionRouteGroupsRestInterceptor, 'post_update_transition_route_group') as post, mock.patch.object(transports.TransitionRouteGroupsRestInterceptor, 'pre_update_transition_route_group') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcdc_transition_route_group.UpdateTransitionRouteGroupRequest.pb(gcdc_transition_route_group.UpdateTransitionRouteGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcdc_transition_route_group.TransitionRouteGroup.to_json(gcdc_transition_route_group.TransitionRouteGroup())
        request = gcdc_transition_route_group.UpdateTransitionRouteGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcdc_transition_route_group.TransitionRouteGroup()
        client.update_transition_route_group(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_transition_route_group_rest_bad_request(transport: str='rest', request_type=gcdc_transition_route_group.UpdateTransitionRouteGroupRequest):
    if False:
        return 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'transition_route_group': {'name': 'projects/sample1/locations/sample2/agents/sample3/flows/sample4/transitionRouteGroups/sample5'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_transition_route_group(request)

def test_update_transition_route_group_rest_flattened():
    if False:
        return 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcdc_transition_route_group.TransitionRouteGroup()
        sample_request = {'transition_route_group': {'name': 'projects/sample1/locations/sample2/agents/sample3/flows/sample4/transitionRouteGroups/sample5'}}
        mock_args = dict(transition_route_group=gcdc_transition_route_group.TransitionRouteGroup(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcdc_transition_route_group.TransitionRouteGroup.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_transition_route_group(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{transition_route_group.name=projects/*/locations/*/agents/*/flows/*/transitionRouteGroups/*}' % client.transport._host, args[1])

def test_update_transition_route_group_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_transition_route_group(gcdc_transition_route_group.UpdateTransitionRouteGroupRequest(), transition_route_group=gcdc_transition_route_group.TransitionRouteGroup(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_transition_route_group_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [transition_route_group.DeleteTransitionRouteGroupRequest, dict])
def test_delete_transition_route_group_rest(request_type):
    if False:
        while True:
            i = 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/agents/sample3/flows/sample4/transitionRouteGroups/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_transition_route_group(request)
    assert response is None

def test_delete_transition_route_group_rest_required_fields(request_type=transition_route_group.DeleteTransitionRouteGroupRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.TransitionRouteGroupsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_transition_route_group._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_transition_route_group._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('force',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_transition_route_group(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_transition_route_group_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.TransitionRouteGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_transition_route_group._get_unset_required_fields({})
    assert set(unset_fields) == set(('force',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_transition_route_group_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.TransitionRouteGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.TransitionRouteGroupsRestInterceptor())
    client = TransitionRouteGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.TransitionRouteGroupsRestInterceptor, 'pre_delete_transition_route_group') as pre:
        pre.assert_not_called()
        pb_message = transition_route_group.DeleteTransitionRouteGroupRequest.pb(transition_route_group.DeleteTransitionRouteGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = transition_route_group.DeleteTransitionRouteGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_transition_route_group(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_transition_route_group_rest_bad_request(transport: str='rest', request_type=transition_route_group.DeleteTransitionRouteGroupRequest):
    if False:
        while True:
            i = 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/agents/sample3/flows/sample4/transitionRouteGroups/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_transition_route_group(request)

def test_delete_transition_route_group_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/agents/sample3/flows/sample4/transitionRouteGroups/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_transition_route_group(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v3beta1/{name=projects/*/locations/*/agents/*/flows/*/transitionRouteGroups/*}' % client.transport._host, args[1])

def test_delete_transition_route_group_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_transition_route_group(transition_route_group.DeleteTransitionRouteGroupRequest(), name='name_value')

def test_delete_transition_route_group_rest_error():
    if False:
        while True:
            i = 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.TransitionRouteGroupsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.TransitionRouteGroupsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TransitionRouteGroupsClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.TransitionRouteGroupsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = TransitionRouteGroupsClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = TransitionRouteGroupsClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.TransitionRouteGroupsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = TransitionRouteGroupsClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.TransitionRouteGroupsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = TransitionRouteGroupsClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.TransitionRouteGroupsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.TransitionRouteGroupsGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.TransitionRouteGroupsGrpcTransport, transports.TransitionRouteGroupsGrpcAsyncIOTransport, transports.TransitionRouteGroupsRestTransport])
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
    transport = TransitionRouteGroupsClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        i = 10
        return i + 15
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.TransitionRouteGroupsGrpcTransport)

def test_transition_route_groups_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.TransitionRouteGroupsTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_transition_route_groups_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.dialogflowcx_v3beta1.services.transition_route_groups.transports.TransitionRouteGroupsTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.TransitionRouteGroupsTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_transition_route_groups', 'get_transition_route_group', 'create_transition_route_group', 'update_transition_route_group', 'delete_transition_route_group', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'list_operations')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_transition_route_groups_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.dialogflowcx_v3beta1.services.transition_route_groups.transports.TransitionRouteGroupsTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.TransitionRouteGroupsTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id='octopus')

def test_transition_route_groups_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.dialogflowcx_v3beta1.services.transition_route_groups.transports.TransitionRouteGroupsTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.TransitionRouteGroupsTransport()
        adc.assert_called_once()

def test_transition_route_groups_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        TransitionRouteGroupsClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.TransitionRouteGroupsGrpcTransport, transports.TransitionRouteGroupsGrpcAsyncIOTransport])
def test_transition_route_groups_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.TransitionRouteGroupsGrpcTransport, transports.TransitionRouteGroupsGrpcAsyncIOTransport, transports.TransitionRouteGroupsRestTransport])
def test_transition_route_groups_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.TransitionRouteGroupsGrpcTransport, grpc_helpers), (transports.TransitionRouteGroupsGrpcAsyncIOTransport, grpc_helpers_async)])
def test_transition_route_groups_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('dialogflow.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), scopes=['1', '2'], default_host='dialogflow.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.TransitionRouteGroupsGrpcTransport, transports.TransitionRouteGroupsGrpcAsyncIOTransport])
def test_transition_route_groups_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_transition_route_groups_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.TransitionRouteGroupsRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_transition_route_groups_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dialogflow.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_transition_route_groups_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dialogflow.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('dialogflow.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_transition_route_groups_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = TransitionRouteGroupsClient(credentials=creds1, transport=transport_name)
    client2 = TransitionRouteGroupsClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_transition_route_groups._session
    session2 = client2.transport.list_transition_route_groups._session
    assert session1 != session2
    session1 = client1.transport.get_transition_route_group._session
    session2 = client2.transport.get_transition_route_group._session
    assert session1 != session2
    session1 = client1.transport.create_transition_route_group._session
    session2 = client2.transport.create_transition_route_group._session
    assert session1 != session2
    session1 = client1.transport.update_transition_route_group._session
    session2 = client2.transport.update_transition_route_group._session
    assert session1 != session2
    session1 = client1.transport.delete_transition_route_group._session
    session2 = client2.transport.delete_transition_route_group._session
    assert session1 != session2

def test_transition_route_groups_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.TransitionRouteGroupsGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_transition_route_groups_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.TransitionRouteGroupsGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.TransitionRouteGroupsGrpcTransport, transports.TransitionRouteGroupsGrpcAsyncIOTransport])
def test_transition_route_groups_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.TransitionRouteGroupsGrpcTransport, transports.TransitionRouteGroupsGrpcAsyncIOTransport])
def test_transition_route_groups_transport_channel_mtls_with_adc(transport_class):
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

def test_flow_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    agent = 'whelk'
    flow = 'octopus'
    expected = 'projects/{project}/locations/{location}/agents/{agent}/flows/{flow}'.format(project=project, location=location, agent=agent, flow=flow)
    actual = TransitionRouteGroupsClient.flow_path(project, location, agent, flow)
    assert expected == actual

def test_parse_flow_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'oyster', 'location': 'nudibranch', 'agent': 'cuttlefish', 'flow': 'mussel'}
    path = TransitionRouteGroupsClient.flow_path(**expected)
    actual = TransitionRouteGroupsClient.parse_flow_path(path)
    assert expected == actual

def test_intent_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'winkle'
    location = 'nautilus'
    agent = 'scallop'
    intent = 'abalone'
    expected = 'projects/{project}/locations/{location}/agents/{agent}/intents/{intent}'.format(project=project, location=location, agent=agent, intent=intent)
    actual = TransitionRouteGroupsClient.intent_path(project, location, agent, intent)
    assert expected == actual

def test_parse_intent_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'squid', 'location': 'clam', 'agent': 'whelk', 'intent': 'octopus'}
    path = TransitionRouteGroupsClient.intent_path(**expected)
    actual = TransitionRouteGroupsClient.parse_intent_path(path)
    assert expected == actual

def test_page_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'oyster'
    location = 'nudibranch'
    agent = 'cuttlefish'
    flow = 'mussel'
    page = 'winkle'
    expected = 'projects/{project}/locations/{location}/agents/{agent}/flows/{flow}/pages/{page}'.format(project=project, location=location, agent=agent, flow=flow, page=page)
    actual = TransitionRouteGroupsClient.page_path(project, location, agent, flow, page)
    assert expected == actual

def test_parse_page_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'nautilus', 'location': 'scallop', 'agent': 'abalone', 'flow': 'squid', 'page': 'clam'}
    path = TransitionRouteGroupsClient.page_path(**expected)
    actual = TransitionRouteGroupsClient.parse_page_path(path)
    assert expected == actual

def test_transition_route_group_path():
    if False:
        print('Hello World!')
    project = 'whelk'
    location = 'octopus'
    agent = 'oyster'
    flow = 'nudibranch'
    transition_route_group = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/agents/{agent}/flows/{flow}/transitionRouteGroups/{transition_route_group}'.format(project=project, location=location, agent=agent, flow=flow, transition_route_group=transition_route_group)
    actual = TransitionRouteGroupsClient.transition_route_group_path(project, location, agent, flow, transition_route_group)
    assert expected == actual

def test_parse_transition_route_group_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'mussel', 'location': 'winkle', 'agent': 'nautilus', 'flow': 'scallop', 'transition_route_group': 'abalone'}
    path = TransitionRouteGroupsClient.transition_route_group_path(**expected)
    actual = TransitionRouteGroupsClient.parse_transition_route_group_path(path)
    assert expected == actual

def test_webhook_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    agent = 'whelk'
    webhook = 'octopus'
    expected = 'projects/{project}/locations/{location}/agents/{agent}/webhooks/{webhook}'.format(project=project, location=location, agent=agent, webhook=webhook)
    actual = TransitionRouteGroupsClient.webhook_path(project, location, agent, webhook)
    assert expected == actual

def test_parse_webhook_path():
    if False:
        print('Hello World!')
    expected = {'project': 'oyster', 'location': 'nudibranch', 'agent': 'cuttlefish', 'webhook': 'mussel'}
    path = TransitionRouteGroupsClient.webhook_path(**expected)
    actual = TransitionRouteGroupsClient.parse_webhook_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'winkle'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = TransitionRouteGroupsClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'nautilus'}
    path = TransitionRouteGroupsClient.common_billing_account_path(**expected)
    actual = TransitionRouteGroupsClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'scallop'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = TransitionRouteGroupsClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'abalone'}
    path = TransitionRouteGroupsClient.common_folder_path(**expected)
    actual = TransitionRouteGroupsClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'squid'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = TransitionRouteGroupsClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'clam'}
    path = TransitionRouteGroupsClient.common_organization_path(**expected)
    actual = TransitionRouteGroupsClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'whelk'
    expected = 'projects/{project}'.format(project=project)
    actual = TransitionRouteGroupsClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'octopus'}
    path = TransitionRouteGroupsClient.common_project_path(**expected)
    actual = TransitionRouteGroupsClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'oyster'
    location = 'nudibranch'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = TransitionRouteGroupsClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'cuttlefish', 'location': 'mussel'}
    path = TransitionRouteGroupsClient.common_location_path(**expected)
    actual = TransitionRouteGroupsClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.TransitionRouteGroupsTransport, '_prep_wrapped_messages') as prep:
        client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.TransitionRouteGroupsTransport, '_prep_wrapped_messages') as prep:
        transport_class = TransitionRouteGroupsClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        return 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        return 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = TransitionRouteGroupsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = TransitionRouteGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(TransitionRouteGroupsClient, transports.TransitionRouteGroupsGrpcTransport), (TransitionRouteGroupsAsyncClient, transports.TransitionRouteGroupsGrpcAsyncIOTransport)])
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