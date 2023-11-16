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
from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import empty_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.rpc import status_pb2
from google.type import datetime_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.video.live_stream_v1.services.livestream_service import LivestreamServiceAsyncClient, LivestreamServiceClient, pagers, transports
from google.cloud.video.live_stream_v1.types import outputs, resources, service

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
    assert LivestreamServiceClient._get_default_mtls_endpoint(None) is None
    assert LivestreamServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert LivestreamServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert LivestreamServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert LivestreamServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert LivestreamServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(LivestreamServiceClient, 'grpc'), (LivestreamServiceAsyncClient, 'grpc_asyncio'), (LivestreamServiceClient, 'rest')])
def test_livestream_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('livestream.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://livestream.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.LivestreamServiceGrpcTransport, 'grpc'), (transports.LivestreamServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.LivestreamServiceRestTransport, 'rest')])
def test_livestream_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(LivestreamServiceClient, 'grpc'), (LivestreamServiceAsyncClient, 'grpc_asyncio'), (LivestreamServiceClient, 'rest')])
def test_livestream_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('livestream.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://livestream.googleapis.com')

def test_livestream_service_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = LivestreamServiceClient.get_transport_class()
    available_transports = [transports.LivestreamServiceGrpcTransport, transports.LivestreamServiceRestTransport]
    assert transport in available_transports
    transport = LivestreamServiceClient.get_transport_class('grpc')
    assert transport == transports.LivestreamServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(LivestreamServiceClient, transports.LivestreamServiceGrpcTransport, 'grpc'), (LivestreamServiceAsyncClient, transports.LivestreamServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (LivestreamServiceClient, transports.LivestreamServiceRestTransport, 'rest')])
@mock.patch.object(LivestreamServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(LivestreamServiceClient))
@mock.patch.object(LivestreamServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(LivestreamServiceAsyncClient))
def test_livestream_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(LivestreamServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(LivestreamServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(LivestreamServiceClient, transports.LivestreamServiceGrpcTransport, 'grpc', 'true'), (LivestreamServiceAsyncClient, transports.LivestreamServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (LivestreamServiceClient, transports.LivestreamServiceGrpcTransport, 'grpc', 'false'), (LivestreamServiceAsyncClient, transports.LivestreamServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (LivestreamServiceClient, transports.LivestreamServiceRestTransport, 'rest', 'true'), (LivestreamServiceClient, transports.LivestreamServiceRestTransport, 'rest', 'false')])
@mock.patch.object(LivestreamServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(LivestreamServiceClient))
@mock.patch.object(LivestreamServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(LivestreamServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_livestream_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
    if False:
        while True:
            i = 10
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

@pytest.mark.parametrize('client_class', [LivestreamServiceClient, LivestreamServiceAsyncClient])
@mock.patch.object(LivestreamServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(LivestreamServiceClient))
@mock.patch.object(LivestreamServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(LivestreamServiceAsyncClient))
def test_livestream_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(LivestreamServiceClient, transports.LivestreamServiceGrpcTransport, 'grpc'), (LivestreamServiceAsyncClient, transports.LivestreamServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (LivestreamServiceClient, transports.LivestreamServiceRestTransport, 'rest')])
def test_livestream_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(LivestreamServiceClient, transports.LivestreamServiceGrpcTransport, 'grpc', grpc_helpers), (LivestreamServiceAsyncClient, transports.LivestreamServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (LivestreamServiceClient, transports.LivestreamServiceRestTransport, 'rest', None)])
def test_livestream_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_livestream_service_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.video.live_stream_v1.services.livestream_service.transports.LivestreamServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = LivestreamServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(LivestreamServiceClient, transports.LivestreamServiceGrpcTransport, 'grpc', grpc_helpers), (LivestreamServiceAsyncClient, transports.LivestreamServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_livestream_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
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
        create_channel.assert_called_with('livestream.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='livestream.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [service.CreateChannelRequest, dict])
def test_create_channel(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_channel_), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateChannelRequest()
    assert isinstance(response, future.Future)

def test_create_channel_empty_call():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_channel_), '__call__') as call:
        client.create_channel()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateChannelRequest()

@pytest.mark.asyncio
async def test_create_channel_async(transport: str='grpc_asyncio', request_type=service.CreateChannelRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_channel_), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateChannelRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_channel_async_from_dict():
    await test_create_channel_async(request_type=dict)

def test_create_channel_field_headers():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateChannelRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_channel_), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_channel_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateChannelRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_channel_), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_channel_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_channel_), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_channel(parent='parent_value', channel=resources.Channel(name='name_value'), channel_id='channel_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].channel
        mock_val = resources.Channel(name='name_value')
        assert arg == mock_val
        arg = args[0].channel_id
        mock_val = 'channel_id_value'
        assert arg == mock_val

def test_create_channel_flattened_error():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_channel(service.CreateChannelRequest(), parent='parent_value', channel=resources.Channel(name='name_value'), channel_id='channel_id_value')

@pytest.mark.asyncio
async def test_create_channel_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_channel_), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_channel(parent='parent_value', channel=resources.Channel(name='name_value'), channel_id='channel_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].channel
        mock_val = resources.Channel(name='name_value')
        assert arg == mock_val
        arg = args[0].channel_id
        mock_val = 'channel_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_channel_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_channel(service.CreateChannelRequest(), parent='parent_value', channel=resources.Channel(name='name_value'), channel_id='channel_id_value')

@pytest.mark.parametrize('request_type', [service.ListChannelsRequest, dict])
def test_list_channels(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_channels), '__call__') as call:
        call.return_value = service.ListChannelsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_channels(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListChannelsRequest()
    assert isinstance(response, pagers.ListChannelsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_channels_empty_call():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_channels), '__call__') as call:
        client.list_channels()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListChannelsRequest()

@pytest.mark.asyncio
async def test_list_channels_async(transport: str='grpc_asyncio', request_type=service.ListChannelsRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_channels), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListChannelsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_channels(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListChannelsRequest()
    assert isinstance(response, pagers.ListChannelsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_channels_async_from_dict():
    await test_list_channels_async(request_type=dict)

def test_list_channels_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListChannelsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_channels), '__call__') as call:
        call.return_value = service.ListChannelsResponse()
        client.list_channels(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_channels_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListChannelsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_channels), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListChannelsResponse())
        await client.list_channels(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_channels_flattened():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_channels), '__call__') as call:
        call.return_value = service.ListChannelsResponse()
        client.list_channels(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_channels_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_channels(service.ListChannelsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_channels_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_channels), '__call__') as call:
        call.return_value = service.ListChannelsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListChannelsResponse())
        response = await client.list_channels(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_channels_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_channels(service.ListChannelsRequest(), parent='parent_value')

def test_list_channels_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_channels), '__call__') as call:
        call.side_effect = (service.ListChannelsResponse(channels=[resources.Channel(), resources.Channel(), resources.Channel()], next_page_token='abc'), service.ListChannelsResponse(channels=[], next_page_token='def'), service.ListChannelsResponse(channels=[resources.Channel()], next_page_token='ghi'), service.ListChannelsResponse(channels=[resources.Channel(), resources.Channel()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_channels(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Channel) for i in results))

def test_list_channels_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_channels), '__call__') as call:
        call.side_effect = (service.ListChannelsResponse(channels=[resources.Channel(), resources.Channel(), resources.Channel()], next_page_token='abc'), service.ListChannelsResponse(channels=[], next_page_token='def'), service.ListChannelsResponse(channels=[resources.Channel()], next_page_token='ghi'), service.ListChannelsResponse(channels=[resources.Channel(), resources.Channel()]), RuntimeError)
        pages = list(client.list_channels(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_channels_async_pager():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_channels), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListChannelsResponse(channels=[resources.Channel(), resources.Channel(), resources.Channel()], next_page_token='abc'), service.ListChannelsResponse(channels=[], next_page_token='def'), service.ListChannelsResponse(channels=[resources.Channel()], next_page_token='ghi'), service.ListChannelsResponse(channels=[resources.Channel(), resources.Channel()]), RuntimeError)
        async_pager = await client.list_channels(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Channel) for i in responses))

@pytest.mark.asyncio
async def test_list_channels_async_pages():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_channels), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListChannelsResponse(channels=[resources.Channel(), resources.Channel(), resources.Channel()], next_page_token='abc'), service.ListChannelsResponse(channels=[], next_page_token='def'), service.ListChannelsResponse(channels=[resources.Channel()], next_page_token='ghi'), service.ListChannelsResponse(channels=[resources.Channel(), resources.Channel()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_channels(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetChannelRequest, dict])
def test_get_channel(request_type, transport: str='grpc'):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_channel), '__call__') as call:
        call.return_value = resources.Channel(name='name_value', active_input='active_input_value', streaming_state=resources.Channel.StreamingState.STREAMING)
        response = client.get_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetChannelRequest()
    assert isinstance(response, resources.Channel)
    assert response.name == 'name_value'
    assert response.active_input == 'active_input_value'
    assert response.streaming_state == resources.Channel.StreamingState.STREAMING

def test_get_channel_empty_call():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_channel), '__call__') as call:
        client.get_channel()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetChannelRequest()

@pytest.mark.asyncio
async def test_get_channel_async(transport: str='grpc_asyncio', request_type=service.GetChannelRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_channel), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Channel(name='name_value', active_input='active_input_value', streaming_state=resources.Channel.StreamingState.STREAMING))
        response = await client.get_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetChannelRequest()
    assert isinstance(response, resources.Channel)
    assert response.name == 'name_value'
    assert response.active_input == 'active_input_value'
    assert response.streaming_state == resources.Channel.StreamingState.STREAMING

@pytest.mark.asyncio
async def test_get_channel_async_from_dict():
    await test_get_channel_async(request_type=dict)

def test_get_channel_field_headers():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetChannelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_channel), '__call__') as call:
        call.return_value = resources.Channel()
        client.get_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_channel_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetChannelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_channel), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Channel())
        await client.get_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_channel_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_channel), '__call__') as call:
        call.return_value = resources.Channel()
        client.get_channel(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_channel_flattened_error():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_channel(service.GetChannelRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_channel_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_channel), '__call__') as call:
        call.return_value = resources.Channel()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Channel())
        response = await client.get_channel(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_channel_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_channel(service.GetChannelRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.DeleteChannelRequest, dict])
def test_delete_channel(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteChannelRequest()
    assert isinstance(response, future.Future)

def test_delete_channel_empty_call():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_channel), '__call__') as call:
        client.delete_channel()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteChannelRequest()

@pytest.mark.asyncio
async def test_delete_channel_async(transport: str='grpc_asyncio', request_type=service.DeleteChannelRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_channel), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteChannelRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_channel_async_from_dict():
    await test_delete_channel_async(request_type=dict)

def test_delete_channel_field_headers():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteChannelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_channel_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteChannelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_channel), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_channel_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_channel(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_channel_flattened_error():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_channel(service.DeleteChannelRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_channel_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_channel(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_channel_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_channel(service.DeleteChannelRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.UpdateChannelRequest, dict])
def test_update_channel(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateChannelRequest()
    assert isinstance(response, future.Future)

def test_update_channel_empty_call():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_channel), '__call__') as call:
        client.update_channel()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateChannelRequest()

@pytest.mark.asyncio
async def test_update_channel_async(transport: str='grpc_asyncio', request_type=service.UpdateChannelRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_channel), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateChannelRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_channel_async_from_dict():
    await test_update_channel_async(request_type=dict)

def test_update_channel_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateChannelRequest()
    request.channel.name = 'name_value'
    with mock.patch.object(type(client.transport.update_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'channel.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_channel_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateChannelRequest()
    request.channel.name = 'name_value'
    with mock.patch.object(type(client.transport.update_channel), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'channel.name=name_value') in kw['metadata']

def test_update_channel_flattened():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_channel(channel=resources.Channel(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].channel
        mock_val = resources.Channel(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_channel_flattened_error():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_channel(service.UpdateChannelRequest(), channel=resources.Channel(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_channel_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_channel(channel=resources.Channel(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].channel
        mock_val = resources.Channel(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_channel_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_channel(service.UpdateChannelRequest(), channel=resources.Channel(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [service.StartChannelRequest, dict])
def test_start_channel(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.start_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.start_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.StartChannelRequest()
    assert isinstance(response, future.Future)

def test_start_channel_empty_call():
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.start_channel), '__call__') as call:
        client.start_channel()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.StartChannelRequest()

@pytest.mark.asyncio
async def test_start_channel_async(transport: str='grpc_asyncio', request_type=service.StartChannelRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.start_channel), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.start_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.StartChannelRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_start_channel_async_from_dict():
    await test_start_channel_async(request_type=dict)

def test_start_channel_field_headers():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.StartChannelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.start_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.start_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_start_channel_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.StartChannelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.start_channel), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.start_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_start_channel_flattened():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.start_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.start_channel(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_start_channel_flattened_error():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.start_channel(service.StartChannelRequest(), name='name_value')

@pytest.mark.asyncio
async def test_start_channel_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.start_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.start_channel(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_start_channel_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.start_channel(service.StartChannelRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.StopChannelRequest, dict])
def test_stop_channel(request_type, transport: str='grpc'):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.stop_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.stop_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.StopChannelRequest()
    assert isinstance(response, future.Future)

def test_stop_channel_empty_call():
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.stop_channel), '__call__') as call:
        client.stop_channel()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.StopChannelRequest()

@pytest.mark.asyncio
async def test_stop_channel_async(transport: str='grpc_asyncio', request_type=service.StopChannelRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.stop_channel), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.stop_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.StopChannelRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_stop_channel_async_from_dict():
    await test_stop_channel_async(request_type=dict)

def test_stop_channel_field_headers():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.StopChannelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.stop_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.stop_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_stop_channel_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.StopChannelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.stop_channel), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.stop_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_stop_channel_flattened():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.stop_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.stop_channel(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_stop_channel_flattened_error():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.stop_channel(service.StopChannelRequest(), name='name_value')

@pytest.mark.asyncio
async def test_stop_channel_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.stop_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.stop_channel(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_stop_channel_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.stop_channel(service.StopChannelRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.CreateInputRequest, dict])
def test_create_input(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_input), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_input(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateInputRequest()
    assert isinstance(response, future.Future)

def test_create_input_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_input), '__call__') as call:
        client.create_input()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateInputRequest()

@pytest.mark.asyncio
async def test_create_input_async(transport: str='grpc_asyncio', request_type=service.CreateInputRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_input), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_input(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateInputRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_input_async_from_dict():
    await test_create_input_async(request_type=dict)

def test_create_input_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateInputRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_input), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_input(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_input_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateInputRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_input), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_input(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_input_flattened():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_input), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_input(parent='parent_value', input=resources.Input(name='name_value'), input_id='input_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].input
        mock_val = resources.Input(name='name_value')
        assert arg == mock_val
        arg = args[0].input_id
        mock_val = 'input_id_value'
        assert arg == mock_val

def test_create_input_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_input(service.CreateInputRequest(), parent='parent_value', input=resources.Input(name='name_value'), input_id='input_id_value')

@pytest.mark.asyncio
async def test_create_input_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_input), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_input(parent='parent_value', input=resources.Input(name='name_value'), input_id='input_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].input
        mock_val = resources.Input(name='name_value')
        assert arg == mock_val
        arg = args[0].input_id
        mock_val = 'input_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_input_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_input(service.CreateInputRequest(), parent='parent_value', input=resources.Input(name='name_value'), input_id='input_id_value')

@pytest.mark.parametrize('request_type', [service.ListInputsRequest, dict])
def test_list_inputs(request_type, transport: str='grpc'):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_inputs), '__call__') as call:
        call.return_value = service.ListInputsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_inputs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListInputsRequest()
    assert isinstance(response, pagers.ListInputsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_inputs_empty_call():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_inputs), '__call__') as call:
        client.list_inputs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListInputsRequest()

@pytest.mark.asyncio
async def test_list_inputs_async(transport: str='grpc_asyncio', request_type=service.ListInputsRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_inputs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListInputsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_inputs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListInputsRequest()
    assert isinstance(response, pagers.ListInputsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_inputs_async_from_dict():
    await test_list_inputs_async(request_type=dict)

def test_list_inputs_field_headers():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListInputsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_inputs), '__call__') as call:
        call.return_value = service.ListInputsResponse()
        client.list_inputs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_inputs_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListInputsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_inputs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListInputsResponse())
        await client.list_inputs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_inputs_flattened():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_inputs), '__call__') as call:
        call.return_value = service.ListInputsResponse()
        client.list_inputs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_inputs_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_inputs(service.ListInputsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_inputs_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_inputs), '__call__') as call:
        call.return_value = service.ListInputsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListInputsResponse())
        response = await client.list_inputs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_inputs_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_inputs(service.ListInputsRequest(), parent='parent_value')

def test_list_inputs_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_inputs), '__call__') as call:
        call.side_effect = (service.ListInputsResponse(inputs=[resources.Input(), resources.Input(), resources.Input()], next_page_token='abc'), service.ListInputsResponse(inputs=[], next_page_token='def'), service.ListInputsResponse(inputs=[resources.Input()], next_page_token='ghi'), service.ListInputsResponse(inputs=[resources.Input(), resources.Input()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_inputs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Input) for i in results))

def test_list_inputs_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_inputs), '__call__') as call:
        call.side_effect = (service.ListInputsResponse(inputs=[resources.Input(), resources.Input(), resources.Input()], next_page_token='abc'), service.ListInputsResponse(inputs=[], next_page_token='def'), service.ListInputsResponse(inputs=[resources.Input()], next_page_token='ghi'), service.ListInputsResponse(inputs=[resources.Input(), resources.Input()]), RuntimeError)
        pages = list(client.list_inputs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_inputs_async_pager():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_inputs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListInputsResponse(inputs=[resources.Input(), resources.Input(), resources.Input()], next_page_token='abc'), service.ListInputsResponse(inputs=[], next_page_token='def'), service.ListInputsResponse(inputs=[resources.Input()], next_page_token='ghi'), service.ListInputsResponse(inputs=[resources.Input(), resources.Input()]), RuntimeError)
        async_pager = await client.list_inputs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Input) for i in responses))

@pytest.mark.asyncio
async def test_list_inputs_async_pages():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_inputs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListInputsResponse(inputs=[resources.Input(), resources.Input(), resources.Input()], next_page_token='abc'), service.ListInputsResponse(inputs=[], next_page_token='def'), service.ListInputsResponse(inputs=[resources.Input()], next_page_token='ghi'), service.ListInputsResponse(inputs=[resources.Input(), resources.Input()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_inputs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetInputRequest, dict])
def test_get_input(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_input), '__call__') as call:
        call.return_value = resources.Input(name='name_value', type_=resources.Input.Type.RTMP_PUSH, tier=resources.Input.Tier.SD, uri='uri_value')
        response = client.get_input(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetInputRequest()
    assert isinstance(response, resources.Input)
    assert response.name == 'name_value'
    assert response.type_ == resources.Input.Type.RTMP_PUSH
    assert response.tier == resources.Input.Tier.SD
    assert response.uri == 'uri_value'

def test_get_input_empty_call():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_input), '__call__') as call:
        client.get_input()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetInputRequest()

@pytest.mark.asyncio
async def test_get_input_async(transport: str='grpc_asyncio', request_type=service.GetInputRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_input), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Input(name='name_value', type_=resources.Input.Type.RTMP_PUSH, tier=resources.Input.Tier.SD, uri='uri_value'))
        response = await client.get_input(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetInputRequest()
    assert isinstance(response, resources.Input)
    assert response.name == 'name_value'
    assert response.type_ == resources.Input.Type.RTMP_PUSH
    assert response.tier == resources.Input.Tier.SD
    assert response.uri == 'uri_value'

@pytest.mark.asyncio
async def test_get_input_async_from_dict():
    await test_get_input_async(request_type=dict)

def test_get_input_field_headers():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetInputRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_input), '__call__') as call:
        call.return_value = resources.Input()
        client.get_input(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_input_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetInputRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_input), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Input())
        await client.get_input(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_input_flattened():
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_input), '__call__') as call:
        call.return_value = resources.Input()
        client.get_input(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_input_flattened_error():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_input(service.GetInputRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_input_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_input), '__call__') as call:
        call.return_value = resources.Input()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Input())
        response = await client.get_input(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_input_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_input(service.GetInputRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.DeleteInputRequest, dict])
def test_delete_input(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_input), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_input(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteInputRequest()
    assert isinstance(response, future.Future)

def test_delete_input_empty_call():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_input), '__call__') as call:
        client.delete_input()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteInputRequest()

@pytest.mark.asyncio
async def test_delete_input_async(transport: str='grpc_asyncio', request_type=service.DeleteInputRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_input), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_input(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteInputRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_input_async_from_dict():
    await test_delete_input_async(request_type=dict)

def test_delete_input_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteInputRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_input), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_input(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_input_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteInputRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_input), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_input(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_input_flattened():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_input), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_input(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_input_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_input(service.DeleteInputRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_input_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_input), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_input(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_input_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_input(service.DeleteInputRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.UpdateInputRequest, dict])
def test_update_input(request_type, transport: str='grpc'):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_input), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_input(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateInputRequest()
    assert isinstance(response, future.Future)

def test_update_input_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_input), '__call__') as call:
        client.update_input()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateInputRequest()

@pytest.mark.asyncio
async def test_update_input_async(transport: str='grpc_asyncio', request_type=service.UpdateInputRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_input), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_input(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdateInputRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_input_async_from_dict():
    await test_update_input_async(request_type=dict)

def test_update_input_field_headers():
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateInputRequest()
    request.input.name = 'name_value'
    with mock.patch.object(type(client.transport.update_input), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_input(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'input.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_input_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdateInputRequest()
    request.input.name = 'name_value'
    with mock.patch.object(type(client.transport.update_input), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_input(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'input.name=name_value') in kw['metadata']

def test_update_input_flattened():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_input), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_input(input=resources.Input(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].input
        mock_val = resources.Input(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_input_flattened_error():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_input(service.UpdateInputRequest(), input=resources.Input(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_input_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_input), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_input(input=resources.Input(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].input
        mock_val = resources.Input(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_input_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_input(service.UpdateInputRequest(), input=resources.Input(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [service.CreateEventRequest, dict])
def test_create_event(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_event), '__call__') as call:
        call.return_value = resources.Event(name='name_value', execute_now=True, state=resources.Event.State.SCHEDULED)
        response = client.create_event(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateEventRequest()
    assert isinstance(response, resources.Event)
    assert response.name == 'name_value'
    assert response.execute_now is True
    assert response.state == resources.Event.State.SCHEDULED

def test_create_event_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_event), '__call__') as call:
        client.create_event()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateEventRequest()

@pytest.mark.asyncio
async def test_create_event_async(transport: str='grpc_asyncio', request_type=service.CreateEventRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_event), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Event(name='name_value', execute_now=True, state=resources.Event.State.SCHEDULED))
        response = await client.create_event(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateEventRequest()
    assert isinstance(response, resources.Event)
    assert response.name == 'name_value'
    assert response.execute_now is True
    assert response.state == resources.Event.State.SCHEDULED

@pytest.mark.asyncio
async def test_create_event_async_from_dict():
    await test_create_event_async(request_type=dict)

def test_create_event_field_headers():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateEventRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_event), '__call__') as call:
        call.return_value = resources.Event()
        client.create_event(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_event_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateEventRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_event), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Event())
        await client.create_event(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_event_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_event), '__call__') as call:
        call.return_value = resources.Event()
        client.create_event(parent='parent_value', event=resources.Event(name='name_value'), event_id='event_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].event
        mock_val = resources.Event(name='name_value')
        assert arg == mock_val
        arg = args[0].event_id
        mock_val = 'event_id_value'
        assert arg == mock_val

def test_create_event_flattened_error():
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_event(service.CreateEventRequest(), parent='parent_value', event=resources.Event(name='name_value'), event_id='event_id_value')

@pytest.mark.asyncio
async def test_create_event_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_event), '__call__') as call:
        call.return_value = resources.Event()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Event())
        response = await client.create_event(parent='parent_value', event=resources.Event(name='name_value'), event_id='event_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].event
        mock_val = resources.Event(name='name_value')
        assert arg == mock_val
        arg = args[0].event_id
        mock_val = 'event_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_event_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_event(service.CreateEventRequest(), parent='parent_value', event=resources.Event(name='name_value'), event_id='event_id_value')

@pytest.mark.parametrize('request_type', [service.ListEventsRequest, dict])
def test_list_events(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_events), '__call__') as call:
        call.return_value = service.ListEventsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_events(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListEventsRequest()
    assert isinstance(response, pagers.ListEventsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_events_empty_call():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_events), '__call__') as call:
        client.list_events()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListEventsRequest()

@pytest.mark.asyncio
async def test_list_events_async(transport: str='grpc_asyncio', request_type=service.ListEventsRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_events), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListEventsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_events(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListEventsRequest()
    assert isinstance(response, pagers.ListEventsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_events_async_from_dict():
    await test_list_events_async(request_type=dict)

def test_list_events_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListEventsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_events), '__call__') as call:
        call.return_value = service.ListEventsResponse()
        client.list_events(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_events_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListEventsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_events), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListEventsResponse())
        await client.list_events(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_events_flattened():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_events), '__call__') as call:
        call.return_value = service.ListEventsResponse()
        client.list_events(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_events_flattened_error():
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_events(service.ListEventsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_events_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_events), '__call__') as call:
        call.return_value = service.ListEventsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListEventsResponse())
        response = await client.list_events(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_events_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_events(service.ListEventsRequest(), parent='parent_value')

def test_list_events_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_events), '__call__') as call:
        call.side_effect = (service.ListEventsResponse(events=[resources.Event(), resources.Event(), resources.Event()], next_page_token='abc'), service.ListEventsResponse(events=[], next_page_token='def'), service.ListEventsResponse(events=[resources.Event()], next_page_token='ghi'), service.ListEventsResponse(events=[resources.Event(), resources.Event()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_events(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Event) for i in results))

def test_list_events_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_events), '__call__') as call:
        call.side_effect = (service.ListEventsResponse(events=[resources.Event(), resources.Event(), resources.Event()], next_page_token='abc'), service.ListEventsResponse(events=[], next_page_token='def'), service.ListEventsResponse(events=[resources.Event()], next_page_token='ghi'), service.ListEventsResponse(events=[resources.Event(), resources.Event()]), RuntimeError)
        pages = list(client.list_events(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_events_async_pager():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_events), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListEventsResponse(events=[resources.Event(), resources.Event(), resources.Event()], next_page_token='abc'), service.ListEventsResponse(events=[], next_page_token='def'), service.ListEventsResponse(events=[resources.Event()], next_page_token='ghi'), service.ListEventsResponse(events=[resources.Event(), resources.Event()]), RuntimeError)
        async_pager = await client.list_events(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Event) for i in responses))

@pytest.mark.asyncio
async def test_list_events_async_pages():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_events), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListEventsResponse(events=[resources.Event(), resources.Event(), resources.Event()], next_page_token='abc'), service.ListEventsResponse(events=[], next_page_token='def'), service.ListEventsResponse(events=[resources.Event()], next_page_token='ghi'), service.ListEventsResponse(events=[resources.Event(), resources.Event()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_events(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetEventRequest, dict])
def test_get_event(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_event), '__call__') as call:
        call.return_value = resources.Event(name='name_value', execute_now=True, state=resources.Event.State.SCHEDULED)
        response = client.get_event(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetEventRequest()
    assert isinstance(response, resources.Event)
    assert response.name == 'name_value'
    assert response.execute_now is True
    assert response.state == resources.Event.State.SCHEDULED

def test_get_event_empty_call():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_event), '__call__') as call:
        client.get_event()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetEventRequest()

@pytest.mark.asyncio
async def test_get_event_async(transport: str='grpc_asyncio', request_type=service.GetEventRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_event), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Event(name='name_value', execute_now=True, state=resources.Event.State.SCHEDULED))
        response = await client.get_event(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetEventRequest()
    assert isinstance(response, resources.Event)
    assert response.name == 'name_value'
    assert response.execute_now is True
    assert response.state == resources.Event.State.SCHEDULED

@pytest.mark.asyncio
async def test_get_event_async_from_dict():
    await test_get_event_async(request_type=dict)

def test_get_event_field_headers():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetEventRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_event), '__call__') as call:
        call.return_value = resources.Event()
        client.get_event(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_event_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetEventRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_event), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Event())
        await client.get_event(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_event_flattened():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_event), '__call__') as call:
        call.return_value = resources.Event()
        client.get_event(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_event_flattened_error():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_event(service.GetEventRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_event_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_event), '__call__') as call:
        call.return_value = resources.Event()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Event())
        response = await client.get_event(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_event_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_event(service.GetEventRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.DeleteEventRequest, dict])
def test_delete_event(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_event), '__call__') as call:
        call.return_value = None
        response = client.delete_event(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteEventRequest()
    assert response is None

def test_delete_event_empty_call():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_event), '__call__') as call:
        client.delete_event()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteEventRequest()

@pytest.mark.asyncio
async def test_delete_event_async(transport: str='grpc_asyncio', request_type=service.DeleteEventRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_event), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_event(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteEventRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_event_async_from_dict():
    await test_delete_event_async(request_type=dict)

def test_delete_event_field_headers():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteEventRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_event), '__call__') as call:
        call.return_value = None
        client.delete_event(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_event_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteEventRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_event), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_event(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_event_flattened():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_event), '__call__') as call:
        call.return_value = None
        client.delete_event(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_event_flattened_error():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_event(service.DeleteEventRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_event_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_event), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_event(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_event_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_event(service.DeleteEventRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.CreateAssetRequest, dict])
def test_create_asset(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_asset), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_asset(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateAssetRequest()
    assert isinstance(response, future.Future)

def test_create_asset_empty_call():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_asset), '__call__') as call:
        client.create_asset()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateAssetRequest()

@pytest.mark.asyncio
async def test_create_asset_async(transport: str='grpc_asyncio', request_type=service.CreateAssetRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_asset), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_asset(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.CreateAssetRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_asset_async_from_dict():
    await test_create_asset_async(request_type=dict)

def test_create_asset_field_headers():
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateAssetRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_asset), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_asset(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_asset_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.CreateAssetRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_asset), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_asset(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_asset_flattened():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_asset), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_asset(parent='parent_value', asset=resources.Asset(name='name_value'), asset_id='asset_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].asset
        mock_val = resources.Asset(name='name_value')
        assert arg == mock_val
        arg = args[0].asset_id
        mock_val = 'asset_id_value'
        assert arg == mock_val

def test_create_asset_flattened_error():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_asset(service.CreateAssetRequest(), parent='parent_value', asset=resources.Asset(name='name_value'), asset_id='asset_id_value')

@pytest.mark.asyncio
async def test_create_asset_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_asset), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_asset(parent='parent_value', asset=resources.Asset(name='name_value'), asset_id='asset_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].asset
        mock_val = resources.Asset(name='name_value')
        assert arg == mock_val
        arg = args[0].asset_id
        mock_val = 'asset_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_asset_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_asset(service.CreateAssetRequest(), parent='parent_value', asset=resources.Asset(name='name_value'), asset_id='asset_id_value')

@pytest.mark.parametrize('request_type', [service.DeleteAssetRequest, dict])
def test_delete_asset(request_type, transport: str='grpc'):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_asset), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_asset(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteAssetRequest()
    assert isinstance(response, future.Future)

def test_delete_asset_empty_call():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_asset), '__call__') as call:
        client.delete_asset()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteAssetRequest()

@pytest.mark.asyncio
async def test_delete_asset_async(transport: str='grpc_asyncio', request_type=service.DeleteAssetRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_asset), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_asset(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.DeleteAssetRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_asset_async_from_dict():
    await test_delete_asset_async(request_type=dict)

def test_delete_asset_field_headers():
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteAssetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_asset), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_asset(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_asset_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.DeleteAssetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_asset), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_asset(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_asset_flattened():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_asset), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_asset(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_asset_flattened_error():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_asset(service.DeleteAssetRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_asset_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_asset), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_asset(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_asset_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_asset(service.DeleteAssetRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.GetAssetRequest, dict])
def test_get_asset(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_asset), '__call__') as call:
        call.return_value = resources.Asset(name='name_value', crc32c='crc32c_value', state=resources.Asset.State.CREATING)
        response = client.get_asset(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetAssetRequest()
    assert isinstance(response, resources.Asset)
    assert response.name == 'name_value'
    assert response.crc32c == 'crc32c_value'
    assert response.state == resources.Asset.State.CREATING

def test_get_asset_empty_call():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_asset), '__call__') as call:
        client.get_asset()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetAssetRequest()

@pytest.mark.asyncio
async def test_get_asset_async(transport: str='grpc_asyncio', request_type=service.GetAssetRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_asset), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Asset(name='name_value', crc32c='crc32c_value', state=resources.Asset.State.CREATING))
        response = await client.get_asset(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetAssetRequest()
    assert isinstance(response, resources.Asset)
    assert response.name == 'name_value'
    assert response.crc32c == 'crc32c_value'
    assert response.state == resources.Asset.State.CREATING

@pytest.mark.asyncio
async def test_get_asset_async_from_dict():
    await test_get_asset_async(request_type=dict)

def test_get_asset_field_headers():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetAssetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_asset), '__call__') as call:
        call.return_value = resources.Asset()
        client.get_asset(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_asset_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetAssetRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_asset), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Asset())
        await client.get_asset(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_asset_flattened():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_asset), '__call__') as call:
        call.return_value = resources.Asset()
        client.get_asset(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_asset_flattened_error():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_asset(service.GetAssetRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_asset_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_asset), '__call__') as call:
        call.return_value = resources.Asset()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Asset())
        response = await client.get_asset(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_asset_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_asset(service.GetAssetRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.ListAssetsRequest, dict])
def test_list_assets(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_assets), '__call__') as call:
        call.return_value = service.ListAssetsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_assets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListAssetsRequest()
    assert isinstance(response, pagers.ListAssetsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_assets_empty_call():
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_assets), '__call__') as call:
        client.list_assets()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListAssetsRequest()

@pytest.mark.asyncio
async def test_list_assets_async(transport: str='grpc_asyncio', request_type=service.ListAssetsRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_assets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListAssetsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_assets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.ListAssetsRequest()
    assert isinstance(response, pagers.ListAssetsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_assets_async_from_dict():
    await test_list_assets_async(request_type=dict)

def test_list_assets_field_headers():
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListAssetsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_assets), '__call__') as call:
        call.return_value = service.ListAssetsResponse()
        client.list_assets(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_assets_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.ListAssetsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_assets), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListAssetsResponse())
        await client.list_assets(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_assets_flattened():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_assets), '__call__') as call:
        call.return_value = service.ListAssetsResponse()
        client.list_assets(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_assets_flattened_error():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_assets(service.ListAssetsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_assets_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_assets), '__call__') as call:
        call.return_value = service.ListAssetsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.ListAssetsResponse())
        response = await client.list_assets(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_assets_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_assets(service.ListAssetsRequest(), parent='parent_value')

def test_list_assets_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_assets), '__call__') as call:
        call.side_effect = (service.ListAssetsResponse(assets=[resources.Asset(), resources.Asset(), resources.Asset()], next_page_token='abc'), service.ListAssetsResponse(assets=[], next_page_token='def'), service.ListAssetsResponse(assets=[resources.Asset()], next_page_token='ghi'), service.ListAssetsResponse(assets=[resources.Asset(), resources.Asset()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_assets(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Asset) for i in results))

def test_list_assets_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_assets), '__call__') as call:
        call.side_effect = (service.ListAssetsResponse(assets=[resources.Asset(), resources.Asset(), resources.Asset()], next_page_token='abc'), service.ListAssetsResponse(assets=[], next_page_token='def'), service.ListAssetsResponse(assets=[resources.Asset()], next_page_token='ghi'), service.ListAssetsResponse(assets=[resources.Asset(), resources.Asset()]), RuntimeError)
        pages = list(client.list_assets(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_assets_async_pager():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_assets), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListAssetsResponse(assets=[resources.Asset(), resources.Asset(), resources.Asset()], next_page_token='abc'), service.ListAssetsResponse(assets=[], next_page_token='def'), service.ListAssetsResponse(assets=[resources.Asset()], next_page_token='ghi'), service.ListAssetsResponse(assets=[resources.Asset(), resources.Asset()]), RuntimeError)
        async_pager = await client.list_assets(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Asset) for i in responses))

@pytest.mark.asyncio
async def test_list_assets_async_pages():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_assets), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (service.ListAssetsResponse(assets=[resources.Asset(), resources.Asset(), resources.Asset()], next_page_token='abc'), service.ListAssetsResponse(assets=[], next_page_token='def'), service.ListAssetsResponse(assets=[resources.Asset()], next_page_token='ghi'), service.ListAssetsResponse(assets=[resources.Asset(), resources.Asset()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_assets(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetPoolRequest, dict])
def test_get_pool(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_pool), '__call__') as call:
        call.return_value = resources.Pool(name='name_value')
        response = client.get_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetPoolRequest()
    assert isinstance(response, resources.Pool)
    assert response.name == 'name_value'

def test_get_pool_empty_call():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_pool), '__call__') as call:
        client.get_pool()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetPoolRequest()

@pytest.mark.asyncio
async def test_get_pool_async(transport: str='grpc_asyncio', request_type=service.GetPoolRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Pool(name='name_value'))
        response = await client.get_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.GetPoolRequest()
    assert isinstance(response, resources.Pool)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_pool_async_from_dict():
    await test_get_pool_async(request_type=dict)

def test_get_pool_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetPoolRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_pool), '__call__') as call:
        call.return_value = resources.Pool()
        client.get_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_pool_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.GetPoolRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Pool())
        await client.get_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_pool_flattened():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_pool), '__call__') as call:
        call.return_value = resources.Pool()
        client.get_pool(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_pool_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_pool(service.GetPoolRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_pool_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_pool), '__call__') as call:
        call.return_value = resources.Pool()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Pool())
        response = await client.get_pool(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_pool_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_pool(service.GetPoolRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [service.UpdatePoolRequest, dict])
def test_update_pool(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdatePoolRequest()
    assert isinstance(response, future.Future)

def test_update_pool_empty_call():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_pool), '__call__') as call:
        client.update_pool()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdatePoolRequest()

@pytest.mark.asyncio
async def test_update_pool_async(transport: str='grpc_asyncio', request_type=service.UpdatePoolRequest):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == service.UpdatePoolRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_pool_async_from_dict():
    await test_update_pool_async(request_type=dict)

def test_update_pool_field_headers():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdatePoolRequest()
    request.pool.name = 'name_value'
    with mock.patch.object(type(client.transport.update_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_pool(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'pool.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_pool_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = service.UpdatePoolRequest()
    request.pool.name = 'name_value'
    with mock.patch.object(type(client.transport.update_pool), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_pool(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'pool.name=name_value') in kw['metadata']

def test_update_pool_flattened():
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_pool(pool=resources.Pool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].pool
        mock_val = resources.Pool(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_pool_flattened_error():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_pool(service.UpdatePoolRequest(), pool=resources.Pool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_pool_flattened_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_pool), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_pool(pool=resources.Pool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].pool
        mock_val = resources.Pool(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_pool_flattened_error_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_pool(service.UpdatePoolRequest(), pool=resources.Pool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [service.CreateChannelRequest, dict])
def test_create_channel_rest(request_type):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['channel'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'input_attachments': [{'key': 'key_value', 'input': 'input_value', 'automatic_failover': {'input_keys': ['input_keys_value1', 'input_keys_value2']}}], 'active_input': 'active_input_value', 'output': {'uri': 'uri_value'}, 'elementary_streams': [{'key': 'key_value', 'video_stream': {'h264': {'width_pixels': 1300, 'height_pixels': 1389, 'frame_rate': 0.1046, 'bitrate_bps': 1167, 'allow_open_gop': True, 'gop_frame_count': 1592, 'gop_duration': {'seconds': 751, 'nanos': 543}, 'vbv_size_bits': 1401, 'vbv_fullness_bits': 1834, 'entropy_coder': 'entropy_coder_value', 'b_pyramid': True, 'b_frame_count': 1364, 'aq_strength': 0.1184, 'profile': 'profile_value', 'tune': 'tune_value'}}, 'audio_stream': {'transmux': True, 'codec': 'codec_value', 'bitrate_bps': 1167, 'channel_count': 1377, 'channel_layout': ['channel_layout_value1', 'channel_layout_value2'], 'mapping_': [{'input_key': 'input_key_value', 'input_track': 1188, 'input_channel': 1384, 'output_channel': 1513, 'gain_db': 0.708}], 'sample_rate_hertz': 1817}, 'text_stream': {'codec': 'codec_value'}}], 'mux_streams': [{'key': 'key_value', 'container': 'container_value', 'elementary_streams': ['elementary_streams_value1', 'elementary_streams_value2'], 'segment_settings': {'segment_duration': {}}, 'encryption_id': 'encryption_id_value'}], 'manifests': [{'file_name': 'file_name_value', 'type_': 1, 'mux_streams': ['mux_streams_value1', 'mux_streams_value2'], 'max_segment_count': 1824, 'segment_keep_duration': {}, 'use_timecode_as_timeline': True}], 'sprite_sheets': [{'format_': 'format__value', 'file_prefix': 'file_prefix_value', 'sprite_width_pixels': 2058, 'sprite_height_pixels': 2147, 'column_count': 1302, 'row_count': 992, 'interval': {}, 'quality': 777}], 'streaming_state': 1, 'streaming_error': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}, 'log_config': {'log_severity': 1}, 'timecode_config': {'source': 1, 'utc_offset': {}, 'time_zone': {'id': 'id_value', 'version': 'version_value'}}, 'encryptions': [{'id': 'id_value', 'secret_manager_key_source': {'secret_version': 'secret_version_value'}, 'drm_systems': {'widevine': {}, 'fairplay': {}, 'playready': {}, 'clearkey': {}}, 'aes128': {}, 'sample_aes': {}, 'mpeg_cenc': {'scheme': 'scheme_value'}}], 'input_config': {'input_switch_mode': 1}}
    test_field = service.CreateChannelRequest.meta.fields['channel']

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
    for (field, value) in request_init['channel'].items():
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
                for i in range(0, len(request_init['channel'][field])):
                    del request_init['channel'][field][i][subfield]
            else:
                del request_init['channel'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_channel(request)
    assert response.operation.name == 'operations/spam'

def test_create_channel_rest_required_fields(request_type=service.CreateChannelRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['channel_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'channelId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_channel_._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'channelId' in jsonified_request
    assert jsonified_request['channelId'] == request_init['channel_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['channelId'] = 'channel_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_channel_._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('channel_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'channelId' in jsonified_request
    assert jsonified_request['channelId'] == 'channel_id_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_channel(request)
            expected_params = [('channelId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_channel_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_channel_._get_unset_required_fields({})
    assert set(unset_fields) == set(('channelId', 'requestId')) & set(('parent', 'channel', 'channelId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_channel_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_create_channel') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_create_channel') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateChannelRequest.pb(service.CreateChannelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.CreateChannelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_channel(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_channel_rest_bad_request(transport: str='rest', request_type=service.CreateChannelRequest):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_channel(request)

def test_create_channel_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', channel=resources.Channel(name='name_value'), channel_id='channel_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_channel(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/channels' % client.transport._host, args[1])

def test_create_channel_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_channel(service.CreateChannelRequest(), parent='parent_value', channel=resources.Channel(name='name_value'), channel_id='channel_id_value')

def test_create_channel_rest_error():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListChannelsRequest, dict])
def test_list_channels_rest(request_type):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListChannelsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListChannelsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_channels(request)
    assert isinstance(response, pagers.ListChannelsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_channels_rest_required_fields(request_type=service.ListChannelsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_channels._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_channels._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListChannelsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListChannelsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_channels(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_channels_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_channels._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_channels_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_list_channels') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_list_channels') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListChannelsRequest.pb(service.ListChannelsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListChannelsResponse.to_json(service.ListChannelsResponse())
        request = service.ListChannelsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListChannelsResponse()
        client.list_channels(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_channels_rest_bad_request(transport: str='rest', request_type=service.ListChannelsRequest):
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_channels(request)

def test_list_channels_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListChannelsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListChannelsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_channels(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/channels' % client.transport._host, args[1])

def test_list_channels_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_channels(service.ListChannelsRequest(), parent='parent_value')

def test_list_channels_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListChannelsResponse(channels=[resources.Channel(), resources.Channel(), resources.Channel()], next_page_token='abc'), service.ListChannelsResponse(channels=[], next_page_token='def'), service.ListChannelsResponse(channels=[resources.Channel()], next_page_token='ghi'), service.ListChannelsResponse(channels=[resources.Channel(), resources.Channel()]))
        response = response + response
        response = tuple((service.ListChannelsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_channels(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Channel) for i in results))
        pages = list(client.list_channels(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetChannelRequest, dict])
def test_get_channel_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/channels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Channel(name='name_value', active_input='active_input_value', streaming_state=resources.Channel.StreamingState.STREAMING)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Channel.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_channel(request)
    assert isinstance(response, resources.Channel)
    assert response.name == 'name_value'
    assert response.active_input == 'active_input_value'
    assert response.streaming_state == resources.Channel.StreamingState.STREAMING

def test_get_channel_rest_required_fields(request_type=service.GetChannelRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_channel._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_channel._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Channel()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Channel.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_channel(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_channel_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_channel._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_channel_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_get_channel') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_get_channel') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetChannelRequest.pb(service.GetChannelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Channel.to_json(resources.Channel())
        request = service.GetChannelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Channel()
        client.get_channel(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_channel_rest_bad_request(transport: str='rest', request_type=service.GetChannelRequest):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/channels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_channel(request)

def test_get_channel_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Channel()
        sample_request = {'name': 'projects/sample1/locations/sample2/channels/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Channel.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_channel(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/channels/*}' % client.transport._host, args[1])

def test_get_channel_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_channel(service.GetChannelRequest(), name='name_value')

def test_get_channel_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DeleteChannelRequest, dict])
def test_delete_channel_rest(request_type):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/channels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_channel(request)
    assert response.operation.name == 'operations/spam'

def test_delete_channel_rest_required_fields(request_type=service.DeleteChannelRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_channel._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_channel._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('force', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_channel(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_channel_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_channel._get_unset_required_fields({})
    assert set(unset_fields) == set(('force', 'requestId')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_channel_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_delete_channel') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_delete_channel') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DeleteChannelRequest.pb(service.DeleteChannelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.DeleteChannelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_channel(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_channel_rest_bad_request(transport: str='rest', request_type=service.DeleteChannelRequest):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/channels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_channel(request)

def test_delete_channel_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/channels/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_channel(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/channels/*}' % client.transport._host, args[1])

def test_delete_channel_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_channel(service.DeleteChannelRequest(), name='name_value')

def test_delete_channel_rest_error():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateChannelRequest, dict])
def test_update_channel_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'channel': {'name': 'projects/sample1/locations/sample2/channels/sample3'}}
    request_init['channel'] = {'name': 'projects/sample1/locations/sample2/channels/sample3', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'input_attachments': [{'key': 'key_value', 'input': 'input_value', 'automatic_failover': {'input_keys': ['input_keys_value1', 'input_keys_value2']}}], 'active_input': 'active_input_value', 'output': {'uri': 'uri_value'}, 'elementary_streams': [{'key': 'key_value', 'video_stream': {'h264': {'width_pixels': 1300, 'height_pixels': 1389, 'frame_rate': 0.1046, 'bitrate_bps': 1167, 'allow_open_gop': True, 'gop_frame_count': 1592, 'gop_duration': {'seconds': 751, 'nanos': 543}, 'vbv_size_bits': 1401, 'vbv_fullness_bits': 1834, 'entropy_coder': 'entropy_coder_value', 'b_pyramid': True, 'b_frame_count': 1364, 'aq_strength': 0.1184, 'profile': 'profile_value', 'tune': 'tune_value'}}, 'audio_stream': {'transmux': True, 'codec': 'codec_value', 'bitrate_bps': 1167, 'channel_count': 1377, 'channel_layout': ['channel_layout_value1', 'channel_layout_value2'], 'mapping_': [{'input_key': 'input_key_value', 'input_track': 1188, 'input_channel': 1384, 'output_channel': 1513, 'gain_db': 0.708}], 'sample_rate_hertz': 1817}, 'text_stream': {'codec': 'codec_value'}}], 'mux_streams': [{'key': 'key_value', 'container': 'container_value', 'elementary_streams': ['elementary_streams_value1', 'elementary_streams_value2'], 'segment_settings': {'segment_duration': {}}, 'encryption_id': 'encryption_id_value'}], 'manifests': [{'file_name': 'file_name_value', 'type_': 1, 'mux_streams': ['mux_streams_value1', 'mux_streams_value2'], 'max_segment_count': 1824, 'segment_keep_duration': {}, 'use_timecode_as_timeline': True}], 'sprite_sheets': [{'format_': 'format__value', 'file_prefix': 'file_prefix_value', 'sprite_width_pixels': 2058, 'sprite_height_pixels': 2147, 'column_count': 1302, 'row_count': 992, 'interval': {}, 'quality': 777}], 'streaming_state': 1, 'streaming_error': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}, 'log_config': {'log_severity': 1}, 'timecode_config': {'source': 1, 'utc_offset': {}, 'time_zone': {'id': 'id_value', 'version': 'version_value'}}, 'encryptions': [{'id': 'id_value', 'secret_manager_key_source': {'secret_version': 'secret_version_value'}, 'drm_systems': {'widevine': {}, 'fairplay': {}, 'playready': {}, 'clearkey': {}}, 'aes128': {}, 'sample_aes': {}, 'mpeg_cenc': {'scheme': 'scheme_value'}}], 'input_config': {'input_switch_mode': 1}}
    test_field = service.UpdateChannelRequest.meta.fields['channel']

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
    for (field, value) in request_init['channel'].items():
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
                for i in range(0, len(request_init['channel'][field])):
                    del request_init['channel'][field][i][subfield]
            else:
                del request_init['channel'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_channel(request)
    assert response.operation.name == 'operations/spam'

def test_update_channel_rest_required_fields(request_type=service.UpdateChannelRequest):
    if False:
        print('Hello World!')
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_channel._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_channel._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_channel(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_channel_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_channel._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask')) & set(('channel',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_channel_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_update_channel') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_update_channel') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateChannelRequest.pb(service.UpdateChannelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.UpdateChannelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_channel(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_channel_rest_bad_request(transport: str='rest', request_type=service.UpdateChannelRequest):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'channel': {'name': 'projects/sample1/locations/sample2/channels/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_channel(request)

def test_update_channel_rest_flattened():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'channel': {'name': 'projects/sample1/locations/sample2/channels/sample3'}}
        mock_args = dict(channel=resources.Channel(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_channel(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{channel.name=projects/*/locations/*/channels/*}' % client.transport._host, args[1])

def test_update_channel_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_channel(service.UpdateChannelRequest(), channel=resources.Channel(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_channel_rest_error():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.StartChannelRequest, dict])
def test_start_channel_rest(request_type):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/channels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.start_channel(request)
    assert response.operation.name == 'operations/spam'

def test_start_channel_rest_required_fields(request_type=service.StartChannelRequest):
    if False:
        return 10
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).start_channel._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).start_channel._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.start_channel(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_start_channel_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.start_channel._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_start_channel_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_start_channel') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_start_channel') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.StartChannelRequest.pb(service.StartChannelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.StartChannelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.start_channel(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_start_channel_rest_bad_request(transport: str='rest', request_type=service.StartChannelRequest):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/channels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.start_channel(request)

def test_start_channel_rest_flattened():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/channels/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.start_channel(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/channels/*}:start' % client.transport._host, args[1])

def test_start_channel_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.start_channel(service.StartChannelRequest(), name='name_value')

def test_start_channel_rest_error():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.StopChannelRequest, dict])
def test_stop_channel_rest(request_type):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/channels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.stop_channel(request)
    assert response.operation.name == 'operations/spam'

def test_stop_channel_rest_required_fields(request_type=service.StopChannelRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).stop_channel._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).stop_channel._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.stop_channel(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_stop_channel_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.stop_channel._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_stop_channel_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_stop_channel') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_stop_channel') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.StopChannelRequest.pb(service.StopChannelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.StopChannelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.stop_channel(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_stop_channel_rest_bad_request(transport: str='rest', request_type=service.StopChannelRequest):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/channels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.stop_channel(request)

def test_stop_channel_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/channels/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.stop_channel(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/channels/*}:stop' % client.transport._host, args[1])

def test_stop_channel_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.stop_channel(service.StopChannelRequest(), name='name_value')

def test_stop_channel_rest_error():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateInputRequest, dict])
def test_create_input_rest(request_type):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['input'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'type_': 1, 'tier': 1, 'uri': 'uri_value', 'preprocessing_config': {'audio': {'lufs': 0.442}, 'crop': {'top_pixels': 1095, 'bottom_pixels': 1417, 'left_pixels': 1183, 'right_pixels': 1298}, 'pad': {'top_pixels': 1095, 'bottom_pixels': 1417, 'left_pixels': 1183, 'right_pixels': 1298}}, 'security_rules': {'ip_ranges': ['ip_ranges_value1', 'ip_ranges_value2']}, 'input_stream_property': {'last_establish_time': {}, 'video_streams': [{'index': 536, 'video_format': {'codec': 'codec_value', 'width_pixels': 1300, 'height_pixels': 1389, 'frame_rate': 0.1046}}], 'audio_streams': [{'index': 536, 'audio_format': {'codec': 'codec_value', 'channel_count': 1377, 'channel_layout': ['channel_layout_value1', 'channel_layout_value2']}}]}}
    test_field = service.CreateInputRequest.meta.fields['input']

    def get_message_fields(field):
        if False:
            return 10
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
    for (field, value) in request_init['input'].items():
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
                for i in range(0, len(request_init['input'][field])):
                    del request_init['input'][field][i][subfield]
            else:
                del request_init['input'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_input(request)
    assert response.operation.name == 'operations/spam'

def test_create_input_rest_required_fields(request_type=service.CreateInputRequest):
    if False:
        print('Hello World!')
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['input_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'inputId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_input._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'inputId' in jsonified_request
    assert jsonified_request['inputId'] == request_init['input_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['inputId'] = 'input_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_input._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('input_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'inputId' in jsonified_request
    assert jsonified_request['inputId'] == 'input_id_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_input(request)
            expected_params = [('inputId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_input_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_input._get_unset_required_fields({})
    assert set(unset_fields) == set(('inputId', 'requestId')) & set(('parent', 'input', 'inputId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_input_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_create_input') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_create_input') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateInputRequest.pb(service.CreateInputRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.CreateInputRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_input(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_input_rest_bad_request(transport: str='rest', request_type=service.CreateInputRequest):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_input(request)

def test_create_input_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', input=resources.Input(name='name_value'), input_id='input_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_input(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/inputs' % client.transport._host, args[1])

def test_create_input_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_input(service.CreateInputRequest(), parent='parent_value', input=resources.Input(name='name_value'), input_id='input_id_value')

def test_create_input_rest_error():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListInputsRequest, dict])
def test_list_inputs_rest(request_type):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListInputsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListInputsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_inputs(request)
    assert isinstance(response, pagers.ListInputsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_inputs_rest_required_fields(request_type=service.ListInputsRequest):
    if False:
        return 10
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_inputs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_inputs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListInputsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListInputsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_inputs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_inputs_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_inputs._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_inputs_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_list_inputs') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_list_inputs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListInputsRequest.pb(service.ListInputsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListInputsResponse.to_json(service.ListInputsResponse())
        request = service.ListInputsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListInputsResponse()
        client.list_inputs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_inputs_rest_bad_request(transport: str='rest', request_type=service.ListInputsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_inputs(request)

def test_list_inputs_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListInputsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListInputsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_inputs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/inputs' % client.transport._host, args[1])

def test_list_inputs_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_inputs(service.ListInputsRequest(), parent='parent_value')

def test_list_inputs_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListInputsResponse(inputs=[resources.Input(), resources.Input(), resources.Input()], next_page_token='abc'), service.ListInputsResponse(inputs=[], next_page_token='def'), service.ListInputsResponse(inputs=[resources.Input()], next_page_token='ghi'), service.ListInputsResponse(inputs=[resources.Input(), resources.Input()]))
        response = response + response
        response = tuple((service.ListInputsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_inputs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Input) for i in results))
        pages = list(client.list_inputs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetInputRequest, dict])
def test_get_input_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/inputs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Input(name='name_value', type_=resources.Input.Type.RTMP_PUSH, tier=resources.Input.Tier.SD, uri='uri_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Input.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_input(request)
    assert isinstance(response, resources.Input)
    assert response.name == 'name_value'
    assert response.type_ == resources.Input.Type.RTMP_PUSH
    assert response.tier == resources.Input.Tier.SD
    assert response.uri == 'uri_value'

def test_get_input_rest_required_fields(request_type=service.GetInputRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_input._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_input._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Input()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Input.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_input(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_input_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_input._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_input_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_get_input') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_get_input') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetInputRequest.pb(service.GetInputRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Input.to_json(resources.Input())
        request = service.GetInputRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Input()
        client.get_input(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_input_rest_bad_request(transport: str='rest', request_type=service.GetInputRequest):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/inputs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_input(request)

def test_get_input_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Input()
        sample_request = {'name': 'projects/sample1/locations/sample2/inputs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Input.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_input(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/inputs/*}' % client.transport._host, args[1])

def test_get_input_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_input(service.GetInputRequest(), name='name_value')

def test_get_input_rest_error():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DeleteInputRequest, dict])
def test_delete_input_rest(request_type):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/inputs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_input(request)
    assert response.operation.name == 'operations/spam'

def test_delete_input_rest_required_fields(request_type=service.DeleteInputRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_input._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_input._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_input(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_input_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_input._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_input_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_delete_input') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_delete_input') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DeleteInputRequest.pb(service.DeleteInputRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.DeleteInputRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_input(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_input_rest_bad_request(transport: str='rest', request_type=service.DeleteInputRequest):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/inputs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_input(request)

def test_delete_input_rest_flattened():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/inputs/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_input(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/inputs/*}' % client.transport._host, args[1])

def test_delete_input_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_input(service.DeleteInputRequest(), name='name_value')

def test_delete_input_rest_error():
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdateInputRequest, dict])
def test_update_input_rest(request_type):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'input': {'name': 'projects/sample1/locations/sample2/inputs/sample3'}}
    request_init['input'] = {'name': 'projects/sample1/locations/sample2/inputs/sample3', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'type_': 1, 'tier': 1, 'uri': 'uri_value', 'preprocessing_config': {'audio': {'lufs': 0.442}, 'crop': {'top_pixels': 1095, 'bottom_pixels': 1417, 'left_pixels': 1183, 'right_pixels': 1298}, 'pad': {'top_pixels': 1095, 'bottom_pixels': 1417, 'left_pixels': 1183, 'right_pixels': 1298}}, 'security_rules': {'ip_ranges': ['ip_ranges_value1', 'ip_ranges_value2']}, 'input_stream_property': {'last_establish_time': {}, 'video_streams': [{'index': 536, 'video_format': {'codec': 'codec_value', 'width_pixels': 1300, 'height_pixels': 1389, 'frame_rate': 0.1046}}], 'audio_streams': [{'index': 536, 'audio_format': {'codec': 'codec_value', 'channel_count': 1377, 'channel_layout': ['channel_layout_value1', 'channel_layout_value2']}}]}}
    test_field = service.UpdateInputRequest.meta.fields['input']

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
    for (field, value) in request_init['input'].items():
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
                for i in range(0, len(request_init['input'][field])):
                    del request_init['input'][field][i][subfield]
            else:
                del request_init['input'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_input(request)
    assert response.operation.name == 'operations/spam'

def test_update_input_rest_required_fields(request_type=service.UpdateInputRequest):
    if False:
        print('Hello World!')
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_input._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_input._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_input(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_input_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_input._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask')) & set(('input',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_input_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_update_input') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_update_input') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdateInputRequest.pb(service.UpdateInputRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.UpdateInputRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_input(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_input_rest_bad_request(transport: str='rest', request_type=service.UpdateInputRequest):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'input': {'name': 'projects/sample1/locations/sample2/inputs/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_input(request)

def test_update_input_rest_flattened():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'input': {'name': 'projects/sample1/locations/sample2/inputs/sample3'}}
        mock_args = dict(input=resources.Input(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_input(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{input.name=projects/*/locations/*/inputs/*}' % client.transport._host, args[1])

def test_update_input_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_input(service.UpdateInputRequest(), input=resources.Input(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_input_rest_error():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateEventRequest, dict])
def test_create_event_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/channels/sample3'}
    request_init['event'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'input_switch': {'input_key': 'input_key_value'}, 'ad_break': {'duration': {'seconds': 751, 'nanos': 543}}, 'return_to_program': {}, 'slate': {'duration': {}, 'asset': 'asset_value'}, 'mute': {'duration': {}}, 'unmute': {}, 'execute_now': True, 'execution_time': {}, 'state': 1, 'error': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}}
    test_field = service.CreateEventRequest.meta.fields['event']

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
    for (field, value) in request_init['event'].items():
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
                for i in range(0, len(request_init['event'][field])):
                    del request_init['event'][field][i][subfield]
            else:
                del request_init['event'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Event(name='name_value', execute_now=True, state=resources.Event.State.SCHEDULED)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Event.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_event(request)
    assert isinstance(response, resources.Event)
    assert response.name == 'name_value'
    assert response.execute_now is True
    assert response.state == resources.Event.State.SCHEDULED

def test_create_event_rest_required_fields(request_type=service.CreateEventRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['event_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'eventId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_event._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'eventId' in jsonified_request
    assert jsonified_request['eventId'] == request_init['event_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['eventId'] = 'event_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_event._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('event_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'eventId' in jsonified_request
    assert jsonified_request['eventId'] == 'event_id_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Event()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Event.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_event(request)
            expected_params = [('eventId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_event_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_event._get_unset_required_fields({})
    assert set(unset_fields) == set(('eventId', 'requestId')) & set(('parent', 'event', 'eventId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_event_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_create_event') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_create_event') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateEventRequest.pb(service.CreateEventRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Event.to_json(resources.Event())
        request = service.CreateEventRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Event()
        client.create_event(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_event_rest_bad_request(transport: str='rest', request_type=service.CreateEventRequest):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/channels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_event(request)

def test_create_event_rest_flattened():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Event()
        sample_request = {'parent': 'projects/sample1/locations/sample2/channels/sample3'}
        mock_args = dict(parent='parent_value', event=resources.Event(name='name_value'), event_id='event_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Event.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_event(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/channels/*}/events' % client.transport._host, args[1])

def test_create_event_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_event(service.CreateEventRequest(), parent='parent_value', event=resources.Event(name='name_value'), event_id='event_id_value')

def test_create_event_rest_error():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListEventsRequest, dict])
def test_list_events_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/channels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListEventsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListEventsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_events(request)
    assert isinstance(response, pagers.ListEventsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_events_rest_required_fields(request_type=service.ListEventsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_events._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_events._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListEventsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListEventsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_events(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_events_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_events._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_events_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_list_events') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_list_events') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListEventsRequest.pb(service.ListEventsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListEventsResponse.to_json(service.ListEventsResponse())
        request = service.ListEventsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListEventsResponse()
        client.list_events(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_events_rest_bad_request(transport: str='rest', request_type=service.ListEventsRequest):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/channels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_events(request)

def test_list_events_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListEventsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/channels/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListEventsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_events(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/channels/*}/events' % client.transport._host, args[1])

def test_list_events_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_events(service.ListEventsRequest(), parent='parent_value')

def test_list_events_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListEventsResponse(events=[resources.Event(), resources.Event(), resources.Event()], next_page_token='abc'), service.ListEventsResponse(events=[], next_page_token='def'), service.ListEventsResponse(events=[resources.Event()], next_page_token='ghi'), service.ListEventsResponse(events=[resources.Event(), resources.Event()]))
        response = response + response
        response = tuple((service.ListEventsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/channels/sample3'}
        pager = client.list_events(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Event) for i in results))
        pages = list(client.list_events(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetEventRequest, dict])
def test_get_event_rest(request_type):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/channels/sample3/events/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Event(name='name_value', execute_now=True, state=resources.Event.State.SCHEDULED)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Event.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_event(request)
    assert isinstance(response, resources.Event)
    assert response.name == 'name_value'
    assert response.execute_now is True
    assert response.state == resources.Event.State.SCHEDULED

def test_get_event_rest_required_fields(request_type=service.GetEventRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_event._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_event._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Event()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Event.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_event(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_event_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_event._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_event_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_get_event') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_get_event') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetEventRequest.pb(service.GetEventRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Event.to_json(resources.Event())
        request = service.GetEventRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Event()
        client.get_event(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_event_rest_bad_request(transport: str='rest', request_type=service.GetEventRequest):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/channels/sample3/events/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_event(request)

def test_get_event_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Event()
        sample_request = {'name': 'projects/sample1/locations/sample2/channels/sample3/events/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Event.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_event(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/channels/*/events/*}' % client.transport._host, args[1])

def test_get_event_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_event(service.GetEventRequest(), name='name_value')

def test_get_event_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DeleteEventRequest, dict])
def test_delete_event_rest(request_type):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/channels/sample3/events/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_event(request)
    assert response is None

def test_delete_event_rest_required_fields(request_type=service.DeleteEventRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_event._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_event._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_event(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_event_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_event._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_event_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_delete_event') as pre:
        pre.assert_not_called()
        pb_message = service.DeleteEventRequest.pb(service.DeleteEventRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = service.DeleteEventRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_event(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_event_rest_bad_request(transport: str='rest', request_type=service.DeleteEventRequest):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/channels/sample3/events/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_event(request)

def test_delete_event_rest_flattened():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/channels/sample3/events/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_event(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/channels/*/events/*}' % client.transport._host, args[1])

def test_delete_event_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_event(service.DeleteEventRequest(), name='name_value')

def test_delete_event_rest_error():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.CreateAssetRequest, dict])
def test_create_asset_rest(request_type):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['asset'] = {'name': 'name_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'video': {'uri': 'uri_value'}, 'image': {'uri': 'uri_value'}, 'crc32c': 'crc32c_value', 'state': 1, 'error': {'code': 411, 'message': 'message_value', 'details': [{'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}]}}
    test_field = service.CreateAssetRequest.meta.fields['asset']

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
    for (field, value) in request_init['asset'].items():
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
                for i in range(0, len(request_init['asset'][field])):
                    del request_init['asset'][field][i][subfield]
            else:
                del request_init['asset'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_asset(request)
    assert response.operation.name == 'operations/spam'

def test_create_asset_rest_required_fields(request_type=service.CreateAssetRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['asset_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'assetId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_asset._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'assetId' in jsonified_request
    assert jsonified_request['assetId'] == request_init['asset_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['assetId'] = 'asset_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_asset._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('asset_id', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'assetId' in jsonified_request
    assert jsonified_request['assetId'] == 'asset_id_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_asset(request)
            expected_params = [('assetId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_asset_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_asset._get_unset_required_fields({})
    assert set(unset_fields) == set(('assetId', 'requestId')) & set(('parent', 'asset', 'assetId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_asset_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_create_asset') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_create_asset') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.CreateAssetRequest.pb(service.CreateAssetRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.CreateAssetRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_asset(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_asset_rest_bad_request(transport: str='rest', request_type=service.CreateAssetRequest):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_asset(request)

def test_create_asset_rest_flattened():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', asset=resources.Asset(name='name_value'), asset_id='asset_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_asset(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/assets' % client.transport._host, args[1])

def test_create_asset_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_asset(service.CreateAssetRequest(), parent='parent_value', asset=resources.Asset(name='name_value'), asset_id='asset_id_value')

def test_create_asset_rest_error():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.DeleteAssetRequest, dict])
def test_delete_asset_rest(request_type):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/assets/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_asset(request)
    assert response.operation.name == 'operations/spam'

def test_delete_asset_rest_required_fields(request_type=service.DeleteAssetRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_asset._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_asset._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_asset(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_asset_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_asset._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_asset_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_delete_asset') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_delete_asset') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.DeleteAssetRequest.pb(service.DeleteAssetRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.DeleteAssetRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_asset(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_asset_rest_bad_request(transport: str='rest', request_type=service.DeleteAssetRequest):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/assets/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_asset(request)

def test_delete_asset_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/assets/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_asset(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/assets/*}' % client.transport._host, args[1])

def test_delete_asset_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_asset(service.DeleteAssetRequest(), name='name_value')

def test_delete_asset_rest_error():
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.GetAssetRequest, dict])
def test_get_asset_rest(request_type):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/assets/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Asset(name='name_value', crc32c='crc32c_value', state=resources.Asset.State.CREATING)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Asset.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_asset(request)
    assert isinstance(response, resources.Asset)
    assert response.name == 'name_value'
    assert response.crc32c == 'crc32c_value'
    assert response.state == resources.Asset.State.CREATING

def test_get_asset_rest_required_fields(request_type=service.GetAssetRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_asset._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_asset._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Asset()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Asset.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_asset(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_asset_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_asset._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_asset_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_get_asset') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_get_asset') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetAssetRequest.pb(service.GetAssetRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Asset.to_json(resources.Asset())
        request = service.GetAssetRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Asset()
        client.get_asset(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_asset_rest_bad_request(transport: str='rest', request_type=service.GetAssetRequest):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/assets/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_asset(request)

def test_get_asset_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Asset()
        sample_request = {'name': 'projects/sample1/locations/sample2/assets/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Asset.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_asset(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/assets/*}' % client.transport._host, args[1])

def test_get_asset_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_asset(service.GetAssetRequest(), name='name_value')

def test_get_asset_rest_error():
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.ListAssetsRequest, dict])
def test_list_assets_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListAssetsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListAssetsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_assets(request)
    assert isinstance(response, pagers.ListAssetsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_assets_rest_required_fields(request_type=service.ListAssetsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_assets._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_assets._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.ListAssetsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.ListAssetsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_assets(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_assets_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_assets._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_assets_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_list_assets') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_list_assets') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.ListAssetsRequest.pb(service.ListAssetsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.ListAssetsResponse.to_json(service.ListAssetsResponse())
        request = service.ListAssetsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.ListAssetsResponse()
        client.list_assets(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_assets_rest_bad_request(transport: str='rest', request_type=service.ListAssetsRequest):
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_assets(request)

def test_list_assets_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.ListAssetsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.ListAssetsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_assets(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/assets' % client.transport._host, args[1])

def test_list_assets_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_assets(service.ListAssetsRequest(), parent='parent_value')

def test_list_assets_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (service.ListAssetsResponse(assets=[resources.Asset(), resources.Asset(), resources.Asset()], next_page_token='abc'), service.ListAssetsResponse(assets=[], next_page_token='def'), service.ListAssetsResponse(assets=[resources.Asset()], next_page_token='ghi'), service.ListAssetsResponse(assets=[resources.Asset(), resources.Asset()]))
        response = response + response
        response = tuple((service.ListAssetsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_assets(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Asset) for i in results))
        pages = list(client.list_assets(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [service.GetPoolRequest, dict])
def test_get_pool_rest(request_type):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/pools/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Pool(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Pool.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_pool(request)
    assert isinstance(response, resources.Pool)
    assert response.name == 'name_value'

def test_get_pool_rest_required_fields(request_type=service.GetPoolRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Pool()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Pool.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_pool(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_pool_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_pool._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_pool_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_get_pool') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_get_pool') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.GetPoolRequest.pb(service.GetPoolRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Pool.to_json(resources.Pool())
        request = service.GetPoolRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Pool()
        client.get_pool(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_pool_rest_bad_request(transport: str='rest', request_type=service.GetPoolRequest):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/pools/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_pool(request)

def test_get_pool_rest_flattened():
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Pool()
        sample_request = {'name': 'projects/sample1/locations/sample2/pools/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Pool.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_pool(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/pools/*}' % client.transport._host, args[1])

def test_get_pool_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_pool(service.GetPoolRequest(), name='name_value')

def test_get_pool_rest_error():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [service.UpdatePoolRequest, dict])
def test_update_pool_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'pool': {'name': 'projects/sample1/locations/sample2/pools/sample3'}}
    request_init['pool'] = {'name': 'projects/sample1/locations/sample2/pools/sample3', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'labels': {}, 'network_config': {'peered_network': 'peered_network_value'}}
    test_field = service.UpdatePoolRequest.meta.fields['pool']

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
    for (field, value) in request_init['pool'].items():
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
                for i in range(0, len(request_init['pool'][field])):
                    del request_init['pool'][field][i][subfield]
            else:
                del request_init['pool'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_pool(request)
    assert response.operation.name == 'operations/spam'

def test_update_pool_rest_required_fields(request_type=service.UpdatePoolRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.LivestreamServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_pool._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_pool._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_pool(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_pool_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_pool._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'updateMask')) & set(('pool',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_pool_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.LivestreamServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.LivestreamServiceRestInterceptor())
    client = LivestreamServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.LivestreamServiceRestInterceptor, 'post_update_pool') as post, mock.patch.object(transports.LivestreamServiceRestInterceptor, 'pre_update_pool') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = service.UpdatePoolRequest.pb(service.UpdatePoolRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = service.UpdatePoolRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_pool(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_pool_rest_bad_request(transport: str='rest', request_type=service.UpdatePoolRequest):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'pool': {'name': 'projects/sample1/locations/sample2/pools/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_pool(request)

def test_update_pool_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'pool': {'name': 'projects/sample1/locations/sample2/pools/sample3'}}
        mock_args = dict(pool=resources.Pool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_pool(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{pool.name=projects/*/locations/*/pools/*}' % client.transport._host, args[1])

def test_update_pool_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_pool(service.UpdatePoolRequest(), pool=resources.Pool(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_pool_rest_error():
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.LivestreamServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.LivestreamServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = LivestreamServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.LivestreamServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = LivestreamServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = LivestreamServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.LivestreamServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = LivestreamServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.LivestreamServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = LivestreamServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.LivestreamServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.LivestreamServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.LivestreamServiceGrpcTransport, transports.LivestreamServiceGrpcAsyncIOTransport, transports.LivestreamServiceRestTransport])
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
    transport = LivestreamServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.LivestreamServiceGrpcTransport)

def test_livestream_service_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.LivestreamServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_livestream_service_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.video.live_stream_v1.services.livestream_service.transports.LivestreamServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.LivestreamServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_channel_', 'list_channels', 'get_channel', 'delete_channel', 'update_channel', 'start_channel', 'stop_channel', 'create_input', 'list_inputs', 'get_input', 'delete_input', 'update_input', 'create_event', 'list_events', 'get_event', 'delete_event', 'create_asset', 'delete_asset', 'get_asset', 'list_assets', 'get_pool', 'update_pool', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_livestream_service_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.video.live_stream_v1.services.livestream_service.transports.LivestreamServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.LivestreamServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_livestream_service_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.video.live_stream_v1.services.livestream_service.transports.LivestreamServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.LivestreamServiceTransport()
        adc.assert_called_once()

def test_livestream_service_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        LivestreamServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.LivestreamServiceGrpcTransport, transports.LivestreamServiceGrpcAsyncIOTransport])
def test_livestream_service_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.LivestreamServiceGrpcTransport, transports.LivestreamServiceGrpcAsyncIOTransport, transports.LivestreamServiceRestTransport])
def test_livestream_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.LivestreamServiceGrpcTransport, grpc_helpers), (transports.LivestreamServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_livestream_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('livestream.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='livestream.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.LivestreamServiceGrpcTransport, transports.LivestreamServiceGrpcAsyncIOTransport])
def test_livestream_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_livestream_service_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.LivestreamServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_livestream_service_rest_lro_client():
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_livestream_service_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='livestream.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('livestream.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://livestream.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_livestream_service_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='livestream.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('livestream.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://livestream.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_livestream_service_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = LivestreamServiceClient(credentials=creds1, transport=transport_name)
    client2 = LivestreamServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_channel_._session
    session2 = client2.transport.create_channel_._session
    assert session1 != session2
    session1 = client1.transport.list_channels._session
    session2 = client2.transport.list_channels._session
    assert session1 != session2
    session1 = client1.transport.get_channel._session
    session2 = client2.transport.get_channel._session
    assert session1 != session2
    session1 = client1.transport.delete_channel._session
    session2 = client2.transport.delete_channel._session
    assert session1 != session2
    session1 = client1.transport.update_channel._session
    session2 = client2.transport.update_channel._session
    assert session1 != session2
    session1 = client1.transport.start_channel._session
    session2 = client2.transport.start_channel._session
    assert session1 != session2
    session1 = client1.transport.stop_channel._session
    session2 = client2.transport.stop_channel._session
    assert session1 != session2
    session1 = client1.transport.create_input._session
    session2 = client2.transport.create_input._session
    assert session1 != session2
    session1 = client1.transport.list_inputs._session
    session2 = client2.transport.list_inputs._session
    assert session1 != session2
    session1 = client1.transport.get_input._session
    session2 = client2.transport.get_input._session
    assert session1 != session2
    session1 = client1.transport.delete_input._session
    session2 = client2.transport.delete_input._session
    assert session1 != session2
    session1 = client1.transport.update_input._session
    session2 = client2.transport.update_input._session
    assert session1 != session2
    session1 = client1.transport.create_event._session
    session2 = client2.transport.create_event._session
    assert session1 != session2
    session1 = client1.transport.list_events._session
    session2 = client2.transport.list_events._session
    assert session1 != session2
    session1 = client1.transport.get_event._session
    session2 = client2.transport.get_event._session
    assert session1 != session2
    session1 = client1.transport.delete_event._session
    session2 = client2.transport.delete_event._session
    assert session1 != session2
    session1 = client1.transport.create_asset._session
    session2 = client2.transport.create_asset._session
    assert session1 != session2
    session1 = client1.transport.delete_asset._session
    session2 = client2.transport.delete_asset._session
    assert session1 != session2
    session1 = client1.transport.get_asset._session
    session2 = client2.transport.get_asset._session
    assert session1 != session2
    session1 = client1.transport.list_assets._session
    session2 = client2.transport.list_assets._session
    assert session1 != session2
    session1 = client1.transport.get_pool._session
    session2 = client2.transport.get_pool._session
    assert session1 != session2
    session1 = client1.transport.update_pool._session
    session2 = client2.transport.update_pool._session
    assert session1 != session2

def test_livestream_service_grpc_transport_channel():
    if False:
        print('Hello World!')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.LivestreamServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_livestream_service_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.LivestreamServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.LivestreamServiceGrpcTransport, transports.LivestreamServiceGrpcAsyncIOTransport])
def test_livestream_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.LivestreamServiceGrpcTransport, transports.LivestreamServiceGrpcAsyncIOTransport])
def test_livestream_service_transport_channel_mtls_with_adc(transport_class):
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

def test_livestream_service_grpc_lro_client():
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_livestream_service_grpc_lro_async_client():
    if False:
        while True:
            i = 10
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_asset_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    asset = 'whelk'
    expected = 'projects/{project}/locations/{location}/assets/{asset}'.format(project=project, location=location, asset=asset)
    actual = LivestreamServiceClient.asset_path(project, location, asset)
    assert expected == actual

def test_parse_asset_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'octopus', 'location': 'oyster', 'asset': 'nudibranch'}
    path = LivestreamServiceClient.asset_path(**expected)
    actual = LivestreamServiceClient.parse_asset_path(path)
    assert expected == actual

def test_channel_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    location = 'mussel'
    channel = 'winkle'
    expected = 'projects/{project}/locations/{location}/channels/{channel}'.format(project=project, location=location, channel=channel)
    actual = LivestreamServiceClient.channel_path(project, location, channel)
    assert expected == actual

def test_parse_channel_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'nautilus', 'location': 'scallop', 'channel': 'abalone'}
    path = LivestreamServiceClient.channel_path(**expected)
    actual = LivestreamServiceClient.parse_channel_path(path)
    assert expected == actual

def test_event_path():
    if False:
        return 10
    project = 'squid'
    location = 'clam'
    channel = 'whelk'
    event = 'octopus'
    expected = 'projects/{project}/locations/{location}/channels/{channel}/events/{event}'.format(project=project, location=location, channel=channel, event=event)
    actual = LivestreamServiceClient.event_path(project, location, channel, event)
    assert expected == actual

def test_parse_event_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'oyster', 'location': 'nudibranch', 'channel': 'cuttlefish', 'event': 'mussel'}
    path = LivestreamServiceClient.event_path(**expected)
    actual = LivestreamServiceClient.parse_event_path(path)
    assert expected == actual

def test_input_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    input = 'scallop'
    expected = 'projects/{project}/locations/{location}/inputs/{input}'.format(project=project, location=location, input=input)
    actual = LivestreamServiceClient.input_path(project, location, input)
    assert expected == actual

def test_parse_input_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'abalone', 'location': 'squid', 'input': 'clam'}
    path = LivestreamServiceClient.input_path(**expected)
    actual = LivestreamServiceClient.parse_input_path(path)
    assert expected == actual

def test_network_path():
    if False:
        while True:
            i = 10
    project = 'whelk'
    network = 'octopus'
    expected = 'projects/{project}/global/networks/{network}'.format(project=project, network=network)
    actual = LivestreamServiceClient.network_path(project, network)
    assert expected == actual

def test_parse_network_path():
    if False:
        return 10
    expected = {'project': 'oyster', 'network': 'nudibranch'}
    path = LivestreamServiceClient.network_path(**expected)
    actual = LivestreamServiceClient.parse_network_path(path)
    assert expected == actual

def test_pool_path():
    if False:
        return 10
    project = 'cuttlefish'
    location = 'mussel'
    pool = 'winkle'
    expected = 'projects/{project}/locations/{location}/pools/{pool}'.format(project=project, location=location, pool=pool)
    actual = LivestreamServiceClient.pool_path(project, location, pool)
    assert expected == actual

def test_parse_pool_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'nautilus', 'location': 'scallop', 'pool': 'abalone'}
    path = LivestreamServiceClient.pool_path(**expected)
    actual = LivestreamServiceClient.parse_pool_path(path)
    assert expected == actual

def test_secret_version_path():
    if False:
        return 10
    project = 'squid'
    secret = 'clam'
    version = 'whelk'
    expected = 'projects/{project}/secrets/{secret}/versions/{version}'.format(project=project, secret=secret, version=version)
    actual = LivestreamServiceClient.secret_version_path(project, secret, version)
    assert expected == actual

def test_parse_secret_version_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'octopus', 'secret': 'oyster', 'version': 'nudibranch'}
    path = LivestreamServiceClient.secret_version_path(**expected)
    actual = LivestreamServiceClient.parse_secret_version_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = LivestreamServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'mussel'}
    path = LivestreamServiceClient.common_billing_account_path(**expected)
    actual = LivestreamServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = LivestreamServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'nautilus'}
    path = LivestreamServiceClient.common_folder_path(**expected)
    actual = LivestreamServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = LivestreamServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'abalone'}
    path = LivestreamServiceClient.common_organization_path(**expected)
    actual = LivestreamServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = LivestreamServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'clam'}
    path = LivestreamServiceClient.common_project_path(**expected)
    actual = LivestreamServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = LivestreamServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = LivestreamServiceClient.common_location_path(**expected)
    actual = LivestreamServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.LivestreamServiceTransport, '_prep_wrapped_messages') as prep:
        client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.LivestreamServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = LivestreamServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2/operations/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.cancel_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.CancelOperationRequest, dict])
def test_cancel_operation_rest(request_type):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/operations/sample3'}
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

def test_delete_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.DeleteOperationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2/operations/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.DeleteOperationRequest, dict])
def test_delete_operation_rest(request_type):
    if False:
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/operations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = '{}'
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_operation(request)
    assert response is None

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2/operations/sample3'}, request)
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
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/operations/sample3'}
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
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2'}, request)
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
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2'}
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

def test_delete_operation(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = operations_pb2.DeleteOperationRequest()
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert response is None

@pytest.mark.asyncio
async def test_delete_operation_async(transport: str='grpc'):
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = operations_pb2.DeleteOperationRequest()
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert response is None

def test_delete_operation_field_headers():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = operations_pb2.DeleteOperationRequest()
    request.name = 'locations'
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        client.delete_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_operation_field_headers_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = operations_pb2.DeleteOperationRequest()
    request.name = 'locations'
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_operation(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=locations') in kw['metadata']

def test_delete_operation_from_dict():
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        return 10
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = LivestreamServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = LivestreamServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(LivestreamServiceClient, transports.LivestreamServiceGrpcTransport), (LivestreamServiceAsyncClient, transports.LivestreamServiceGrpcAsyncIOTransport)])
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