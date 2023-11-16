import os
try:
    from unittest import mock
    from unittest.mock import AsyncMock
except ImportError:
    import mock
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
from google.protobuf import duration_pb2
from google.protobuf import empty_pb2
from google.protobuf import field_mask_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from google.cloud.video.stitcher_v1.services.video_stitcher_service import VideoStitcherServiceAsyncClient, VideoStitcherServiceClient, pagers, transports
from google.cloud.video.stitcher_v1.types import ad_tag_details, cdn_keys, companions, events, live_configs, sessions, slates, stitch_details, video_stitcher_service

def client_cert_source_callback():
    if False:
        for i in range(10):
            print('nop')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        return 10
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        return 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert VideoStitcherServiceClient._get_default_mtls_endpoint(None) is None
    assert VideoStitcherServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert VideoStitcherServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert VideoStitcherServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert VideoStitcherServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert VideoStitcherServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(VideoStitcherServiceClient, 'grpc'), (VideoStitcherServiceAsyncClient, 'grpc_asyncio')])
def test_video_stitcher_service_client_from_service_account_info(client_class, transport_name):
    if False:
        print('Hello World!')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == 'videostitcher.googleapis.com:443'

@pytest.mark.parametrize('transport_class,transport_name', [(transports.VideoStitcherServiceGrpcTransport, 'grpc'), (transports.VideoStitcherServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_video_stitcher_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(VideoStitcherServiceClient, 'grpc'), (VideoStitcherServiceAsyncClient, 'grpc_asyncio')])
def test_video_stitcher_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == 'videostitcher.googleapis.com:443'

def test_video_stitcher_service_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = VideoStitcherServiceClient.get_transport_class()
    available_transports = [transports.VideoStitcherServiceGrpcTransport]
    assert transport in available_transports
    transport = VideoStitcherServiceClient.get_transport_class('grpc')
    assert transport == transports.VideoStitcherServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(VideoStitcherServiceClient, transports.VideoStitcherServiceGrpcTransport, 'grpc'), (VideoStitcherServiceAsyncClient, transports.VideoStitcherServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
@mock.patch.object(VideoStitcherServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VideoStitcherServiceClient))
@mock.patch.object(VideoStitcherServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VideoStitcherServiceAsyncClient))
def test_video_stitcher_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(VideoStitcherServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(VideoStitcherServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(VideoStitcherServiceClient, transports.VideoStitcherServiceGrpcTransport, 'grpc', 'true'), (VideoStitcherServiceAsyncClient, transports.VideoStitcherServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (VideoStitcherServiceClient, transports.VideoStitcherServiceGrpcTransport, 'grpc', 'false'), (VideoStitcherServiceAsyncClient, transports.VideoStitcherServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false')])
@mock.patch.object(VideoStitcherServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VideoStitcherServiceClient))
@mock.patch.object(VideoStitcherServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VideoStitcherServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_video_stitcher_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [VideoStitcherServiceClient, VideoStitcherServiceAsyncClient])
@mock.patch.object(VideoStitcherServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VideoStitcherServiceClient))
@mock.patch.object(VideoStitcherServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VideoStitcherServiceAsyncClient))
def test_video_stitcher_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(VideoStitcherServiceClient, transports.VideoStitcherServiceGrpcTransport, 'grpc'), (VideoStitcherServiceAsyncClient, transports.VideoStitcherServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_video_stitcher_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(VideoStitcherServiceClient, transports.VideoStitcherServiceGrpcTransport, 'grpc', grpc_helpers), (VideoStitcherServiceAsyncClient, transports.VideoStitcherServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_video_stitcher_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_video_stitcher_service_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.video.stitcher_v1.services.video_stitcher_service.transports.VideoStitcherServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = VideoStitcherServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(VideoStitcherServiceClient, transports.VideoStitcherServiceGrpcTransport, 'grpc', grpc_helpers), (VideoStitcherServiceAsyncClient, transports.VideoStitcherServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_video_stitcher_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('videostitcher.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='videostitcher.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [video_stitcher_service.CreateCdnKeyRequest, dict])
def test_create_cdn_key(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_cdn_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_cdn_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.CreateCdnKeyRequest()
    assert isinstance(response, future.Future)

def test_create_cdn_key_empty_call():
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_cdn_key), '__call__') as call:
        client.create_cdn_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.CreateCdnKeyRequest()

@pytest.mark.asyncio
async def test_create_cdn_key_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.CreateCdnKeyRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_cdn_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_cdn_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.CreateCdnKeyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_cdn_key_async_from_dict():
    await test_create_cdn_key_async(request_type=dict)

def test_create_cdn_key_field_headers():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.CreateCdnKeyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_cdn_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_cdn_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_cdn_key_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.CreateCdnKeyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_cdn_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_cdn_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_cdn_key_flattened():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_cdn_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_cdn_key(parent='parent_value', cdn_key=cdn_keys.CdnKey(google_cdn_key=cdn_keys.GoogleCdnKey(private_key=b'private_key_blob')), cdn_key_id='cdn_key_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].cdn_key
        mock_val = cdn_keys.CdnKey(google_cdn_key=cdn_keys.GoogleCdnKey(private_key=b'private_key_blob'))
        assert arg == mock_val
        arg = args[0].cdn_key_id
        mock_val = 'cdn_key_id_value'
        assert arg == mock_val

def test_create_cdn_key_flattened_error():
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_cdn_key(video_stitcher_service.CreateCdnKeyRequest(), parent='parent_value', cdn_key=cdn_keys.CdnKey(google_cdn_key=cdn_keys.GoogleCdnKey(private_key=b'private_key_blob')), cdn_key_id='cdn_key_id_value')

@pytest.mark.asyncio
async def test_create_cdn_key_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_cdn_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_cdn_key(parent='parent_value', cdn_key=cdn_keys.CdnKey(google_cdn_key=cdn_keys.GoogleCdnKey(private_key=b'private_key_blob')), cdn_key_id='cdn_key_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].cdn_key
        mock_val = cdn_keys.CdnKey(google_cdn_key=cdn_keys.GoogleCdnKey(private_key=b'private_key_blob'))
        assert arg == mock_val
        arg = args[0].cdn_key_id
        mock_val = 'cdn_key_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_cdn_key_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_cdn_key(video_stitcher_service.CreateCdnKeyRequest(), parent='parent_value', cdn_key=cdn_keys.CdnKey(google_cdn_key=cdn_keys.GoogleCdnKey(private_key=b'private_key_blob')), cdn_key_id='cdn_key_id_value')

@pytest.mark.parametrize('request_type', [video_stitcher_service.ListCdnKeysRequest, dict])
def test_list_cdn_keys(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_cdn_keys), '__call__') as call:
        call.return_value = video_stitcher_service.ListCdnKeysResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_cdn_keys(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.ListCdnKeysRequest()
    assert isinstance(response, pagers.ListCdnKeysPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_cdn_keys_empty_call():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_cdn_keys), '__call__') as call:
        client.list_cdn_keys()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.ListCdnKeysRequest()

@pytest.mark.asyncio
async def test_list_cdn_keys_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.ListCdnKeysRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_cdn_keys), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(video_stitcher_service.ListCdnKeysResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_cdn_keys(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.ListCdnKeysRequest()
    assert isinstance(response, pagers.ListCdnKeysAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_cdn_keys_async_from_dict():
    await test_list_cdn_keys_async(request_type=dict)

def test_list_cdn_keys_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.ListCdnKeysRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_cdn_keys), '__call__') as call:
        call.return_value = video_stitcher_service.ListCdnKeysResponse()
        client.list_cdn_keys(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_cdn_keys_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.ListCdnKeysRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_cdn_keys), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(video_stitcher_service.ListCdnKeysResponse())
        await client.list_cdn_keys(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_cdn_keys_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_cdn_keys), '__call__') as call:
        call.return_value = video_stitcher_service.ListCdnKeysResponse()
        client.list_cdn_keys(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_cdn_keys_flattened_error():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_cdn_keys(video_stitcher_service.ListCdnKeysRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_cdn_keys_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_cdn_keys), '__call__') as call:
        call.return_value = video_stitcher_service.ListCdnKeysResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(video_stitcher_service.ListCdnKeysResponse())
        response = await client.list_cdn_keys(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_cdn_keys_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_cdn_keys(video_stitcher_service.ListCdnKeysRequest(), parent='parent_value')

def test_list_cdn_keys_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_cdn_keys), '__call__') as call:
        call.side_effect = (video_stitcher_service.ListCdnKeysResponse(cdn_keys=[cdn_keys.CdnKey(), cdn_keys.CdnKey(), cdn_keys.CdnKey()], next_page_token='abc'), video_stitcher_service.ListCdnKeysResponse(cdn_keys=[], next_page_token='def'), video_stitcher_service.ListCdnKeysResponse(cdn_keys=[cdn_keys.CdnKey()], next_page_token='ghi'), video_stitcher_service.ListCdnKeysResponse(cdn_keys=[cdn_keys.CdnKey(), cdn_keys.CdnKey()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_cdn_keys(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cdn_keys.CdnKey) for i in results))

def test_list_cdn_keys_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_cdn_keys), '__call__') as call:
        call.side_effect = (video_stitcher_service.ListCdnKeysResponse(cdn_keys=[cdn_keys.CdnKey(), cdn_keys.CdnKey(), cdn_keys.CdnKey()], next_page_token='abc'), video_stitcher_service.ListCdnKeysResponse(cdn_keys=[], next_page_token='def'), video_stitcher_service.ListCdnKeysResponse(cdn_keys=[cdn_keys.CdnKey()], next_page_token='ghi'), video_stitcher_service.ListCdnKeysResponse(cdn_keys=[cdn_keys.CdnKey(), cdn_keys.CdnKey()]), RuntimeError)
        pages = list(client.list_cdn_keys(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_cdn_keys_async_pager():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_cdn_keys), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (video_stitcher_service.ListCdnKeysResponse(cdn_keys=[cdn_keys.CdnKey(), cdn_keys.CdnKey(), cdn_keys.CdnKey()], next_page_token='abc'), video_stitcher_service.ListCdnKeysResponse(cdn_keys=[], next_page_token='def'), video_stitcher_service.ListCdnKeysResponse(cdn_keys=[cdn_keys.CdnKey()], next_page_token='ghi'), video_stitcher_service.ListCdnKeysResponse(cdn_keys=[cdn_keys.CdnKey(), cdn_keys.CdnKey()]), RuntimeError)
        async_pager = await client.list_cdn_keys(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, cdn_keys.CdnKey) for i in responses))

@pytest.mark.asyncio
async def test_list_cdn_keys_async_pages():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_cdn_keys), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (video_stitcher_service.ListCdnKeysResponse(cdn_keys=[cdn_keys.CdnKey(), cdn_keys.CdnKey(), cdn_keys.CdnKey()], next_page_token='abc'), video_stitcher_service.ListCdnKeysResponse(cdn_keys=[], next_page_token='def'), video_stitcher_service.ListCdnKeysResponse(cdn_keys=[cdn_keys.CdnKey()], next_page_token='ghi'), video_stitcher_service.ListCdnKeysResponse(cdn_keys=[cdn_keys.CdnKey(), cdn_keys.CdnKey()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_cdn_keys(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [video_stitcher_service.GetCdnKeyRequest, dict])
def test_get_cdn_key(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_cdn_key), '__call__') as call:
        call.return_value = cdn_keys.CdnKey(name='name_value', hostname='hostname_value')
        response = client.get_cdn_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetCdnKeyRequest()
    assert isinstance(response, cdn_keys.CdnKey)
    assert response.name == 'name_value'
    assert response.hostname == 'hostname_value'

def test_get_cdn_key_empty_call():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_cdn_key), '__call__') as call:
        client.get_cdn_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetCdnKeyRequest()

@pytest.mark.asyncio
async def test_get_cdn_key_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.GetCdnKeyRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_cdn_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cdn_keys.CdnKey(name='name_value', hostname='hostname_value'))
        response = await client.get_cdn_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetCdnKeyRequest()
    assert isinstance(response, cdn_keys.CdnKey)
    assert response.name == 'name_value'
    assert response.hostname == 'hostname_value'

@pytest.mark.asyncio
async def test_get_cdn_key_async_from_dict():
    await test_get_cdn_key_async(request_type=dict)

def test_get_cdn_key_field_headers():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.GetCdnKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_cdn_key), '__call__') as call:
        call.return_value = cdn_keys.CdnKey()
        client.get_cdn_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_cdn_key_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.GetCdnKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_cdn_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cdn_keys.CdnKey())
        await client.get_cdn_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_cdn_key_flattened():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_cdn_key), '__call__') as call:
        call.return_value = cdn_keys.CdnKey()
        client.get_cdn_key(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_cdn_key_flattened_error():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_cdn_key(video_stitcher_service.GetCdnKeyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_cdn_key_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_cdn_key), '__call__') as call:
        call.return_value = cdn_keys.CdnKey()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cdn_keys.CdnKey())
        response = await client.get_cdn_key(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_cdn_key_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_cdn_key(video_stitcher_service.GetCdnKeyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [video_stitcher_service.DeleteCdnKeyRequest, dict])
def test_delete_cdn_key(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_cdn_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_cdn_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.DeleteCdnKeyRequest()
    assert isinstance(response, future.Future)

def test_delete_cdn_key_empty_call():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_cdn_key), '__call__') as call:
        client.delete_cdn_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.DeleteCdnKeyRequest()

@pytest.mark.asyncio
async def test_delete_cdn_key_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.DeleteCdnKeyRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_cdn_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_cdn_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.DeleteCdnKeyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_cdn_key_async_from_dict():
    await test_delete_cdn_key_async(request_type=dict)

def test_delete_cdn_key_field_headers():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.DeleteCdnKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_cdn_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_cdn_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_cdn_key_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.DeleteCdnKeyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_cdn_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_cdn_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_cdn_key_flattened():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_cdn_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_cdn_key(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_cdn_key_flattened_error():
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_cdn_key(video_stitcher_service.DeleteCdnKeyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_cdn_key_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_cdn_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_cdn_key(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_cdn_key_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_cdn_key(video_stitcher_service.DeleteCdnKeyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [video_stitcher_service.UpdateCdnKeyRequest, dict])
def test_update_cdn_key(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_cdn_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_cdn_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.UpdateCdnKeyRequest()
    assert isinstance(response, future.Future)

def test_update_cdn_key_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_cdn_key), '__call__') as call:
        client.update_cdn_key()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.UpdateCdnKeyRequest()

@pytest.mark.asyncio
async def test_update_cdn_key_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.UpdateCdnKeyRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_cdn_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_cdn_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.UpdateCdnKeyRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_cdn_key_async_from_dict():
    await test_update_cdn_key_async(request_type=dict)

def test_update_cdn_key_field_headers():
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.UpdateCdnKeyRequest()
    request.cdn_key.name = 'name_value'
    with mock.patch.object(type(client.transport.update_cdn_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_cdn_key(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'cdn_key.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_cdn_key_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.UpdateCdnKeyRequest()
    request.cdn_key.name = 'name_value'
    with mock.patch.object(type(client.transport.update_cdn_key), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_cdn_key(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'cdn_key.name=name_value') in kw['metadata']

def test_update_cdn_key_flattened():
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_cdn_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_cdn_key(cdn_key=cdn_keys.CdnKey(google_cdn_key=cdn_keys.GoogleCdnKey(private_key=b'private_key_blob')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].cdn_key
        mock_val = cdn_keys.CdnKey(google_cdn_key=cdn_keys.GoogleCdnKey(private_key=b'private_key_blob'))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_cdn_key_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_cdn_key(video_stitcher_service.UpdateCdnKeyRequest(), cdn_key=cdn_keys.CdnKey(google_cdn_key=cdn_keys.GoogleCdnKey(private_key=b'private_key_blob')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_cdn_key_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_cdn_key), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_cdn_key(cdn_key=cdn_keys.CdnKey(google_cdn_key=cdn_keys.GoogleCdnKey(private_key=b'private_key_blob')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].cdn_key
        mock_val = cdn_keys.CdnKey(google_cdn_key=cdn_keys.GoogleCdnKey(private_key=b'private_key_blob'))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_cdn_key_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_cdn_key(video_stitcher_service.UpdateCdnKeyRequest(), cdn_key=cdn_keys.CdnKey(google_cdn_key=cdn_keys.GoogleCdnKey(private_key=b'private_key_blob')), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [video_stitcher_service.CreateVodSessionRequest, dict])
def test_create_vod_session(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_vod_session), '__call__') as call:
        call.return_value = sessions.VodSession(name='name_value', play_uri='play_uri_value', source_uri='source_uri_value', ad_tag_uri='ad_tag_uri_value', asset_id='asset_id_value', ad_tracking=live_configs.AdTracking.CLIENT)
        response = client.create_vod_session(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.CreateVodSessionRequest()
    assert isinstance(response, sessions.VodSession)
    assert response.name == 'name_value'
    assert response.play_uri == 'play_uri_value'
    assert response.source_uri == 'source_uri_value'
    assert response.ad_tag_uri == 'ad_tag_uri_value'
    assert response.asset_id == 'asset_id_value'
    assert response.ad_tracking == live_configs.AdTracking.CLIENT

def test_create_vod_session_empty_call():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_vod_session), '__call__') as call:
        client.create_vod_session()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.CreateVodSessionRequest()

@pytest.mark.asyncio
async def test_create_vod_session_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.CreateVodSessionRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_vod_session), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(sessions.VodSession(name='name_value', play_uri='play_uri_value', source_uri='source_uri_value', ad_tag_uri='ad_tag_uri_value', asset_id='asset_id_value', ad_tracking=live_configs.AdTracking.CLIENT))
        response = await client.create_vod_session(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.CreateVodSessionRequest()
    assert isinstance(response, sessions.VodSession)
    assert response.name == 'name_value'
    assert response.play_uri == 'play_uri_value'
    assert response.source_uri == 'source_uri_value'
    assert response.ad_tag_uri == 'ad_tag_uri_value'
    assert response.asset_id == 'asset_id_value'
    assert response.ad_tracking == live_configs.AdTracking.CLIENT

@pytest.mark.asyncio
async def test_create_vod_session_async_from_dict():
    await test_create_vod_session_async(request_type=dict)

def test_create_vod_session_field_headers():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.CreateVodSessionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_vod_session), '__call__') as call:
        call.return_value = sessions.VodSession()
        client.create_vod_session(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_vod_session_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.CreateVodSessionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_vod_session), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(sessions.VodSession())
        await client.create_vod_session(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_vod_session_flattened():
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_vod_session), '__call__') as call:
        call.return_value = sessions.VodSession()
        client.create_vod_session(parent='parent_value', vod_session=sessions.VodSession(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].vod_session
        mock_val = sessions.VodSession(name='name_value')
        assert arg == mock_val

def test_create_vod_session_flattened_error():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_vod_session(video_stitcher_service.CreateVodSessionRequest(), parent='parent_value', vod_session=sessions.VodSession(name='name_value'))

@pytest.mark.asyncio
async def test_create_vod_session_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_vod_session), '__call__') as call:
        call.return_value = sessions.VodSession()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(sessions.VodSession())
        response = await client.create_vod_session(parent='parent_value', vod_session=sessions.VodSession(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].vod_session
        mock_val = sessions.VodSession(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_vod_session_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_vod_session(video_stitcher_service.CreateVodSessionRequest(), parent='parent_value', vod_session=sessions.VodSession(name='name_value'))

@pytest.mark.parametrize('request_type', [video_stitcher_service.GetVodSessionRequest, dict])
def test_get_vod_session(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_vod_session), '__call__') as call:
        call.return_value = sessions.VodSession(name='name_value', play_uri='play_uri_value', source_uri='source_uri_value', ad_tag_uri='ad_tag_uri_value', asset_id='asset_id_value', ad_tracking=live_configs.AdTracking.CLIENT)
        response = client.get_vod_session(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetVodSessionRequest()
    assert isinstance(response, sessions.VodSession)
    assert response.name == 'name_value'
    assert response.play_uri == 'play_uri_value'
    assert response.source_uri == 'source_uri_value'
    assert response.ad_tag_uri == 'ad_tag_uri_value'
    assert response.asset_id == 'asset_id_value'
    assert response.ad_tracking == live_configs.AdTracking.CLIENT

def test_get_vod_session_empty_call():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_vod_session), '__call__') as call:
        client.get_vod_session()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetVodSessionRequest()

@pytest.mark.asyncio
async def test_get_vod_session_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.GetVodSessionRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_vod_session), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(sessions.VodSession(name='name_value', play_uri='play_uri_value', source_uri='source_uri_value', ad_tag_uri='ad_tag_uri_value', asset_id='asset_id_value', ad_tracking=live_configs.AdTracking.CLIENT))
        response = await client.get_vod_session(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetVodSessionRequest()
    assert isinstance(response, sessions.VodSession)
    assert response.name == 'name_value'
    assert response.play_uri == 'play_uri_value'
    assert response.source_uri == 'source_uri_value'
    assert response.ad_tag_uri == 'ad_tag_uri_value'
    assert response.asset_id == 'asset_id_value'
    assert response.ad_tracking == live_configs.AdTracking.CLIENT

@pytest.mark.asyncio
async def test_get_vod_session_async_from_dict():
    await test_get_vod_session_async(request_type=dict)

def test_get_vod_session_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.GetVodSessionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_vod_session), '__call__') as call:
        call.return_value = sessions.VodSession()
        client.get_vod_session(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_vod_session_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.GetVodSessionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_vod_session), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(sessions.VodSession())
        await client.get_vod_session(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_vod_session_flattened():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_vod_session), '__call__') as call:
        call.return_value = sessions.VodSession()
        client.get_vod_session(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_vod_session_flattened_error():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_vod_session(video_stitcher_service.GetVodSessionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_vod_session_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_vod_session), '__call__') as call:
        call.return_value = sessions.VodSession()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(sessions.VodSession())
        response = await client.get_vod_session(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_vod_session_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_vod_session(video_stitcher_service.GetVodSessionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [video_stitcher_service.ListVodStitchDetailsRequest, dict])
def test_list_vod_stitch_details(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_vod_stitch_details), '__call__') as call:
        call.return_value = video_stitcher_service.ListVodStitchDetailsResponse(next_page_token='next_page_token_value')
        response = client.list_vod_stitch_details(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.ListVodStitchDetailsRequest()
    assert isinstance(response, pagers.ListVodStitchDetailsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_vod_stitch_details_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_vod_stitch_details), '__call__') as call:
        client.list_vod_stitch_details()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.ListVodStitchDetailsRequest()

@pytest.mark.asyncio
async def test_list_vod_stitch_details_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.ListVodStitchDetailsRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_vod_stitch_details), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(video_stitcher_service.ListVodStitchDetailsResponse(next_page_token='next_page_token_value'))
        response = await client.list_vod_stitch_details(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.ListVodStitchDetailsRequest()
    assert isinstance(response, pagers.ListVodStitchDetailsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_vod_stitch_details_async_from_dict():
    await test_list_vod_stitch_details_async(request_type=dict)

def test_list_vod_stitch_details_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.ListVodStitchDetailsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_vod_stitch_details), '__call__') as call:
        call.return_value = video_stitcher_service.ListVodStitchDetailsResponse()
        client.list_vod_stitch_details(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_vod_stitch_details_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.ListVodStitchDetailsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_vod_stitch_details), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(video_stitcher_service.ListVodStitchDetailsResponse())
        await client.list_vod_stitch_details(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_vod_stitch_details_flattened():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_vod_stitch_details), '__call__') as call:
        call.return_value = video_stitcher_service.ListVodStitchDetailsResponse()
        client.list_vod_stitch_details(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_vod_stitch_details_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_vod_stitch_details(video_stitcher_service.ListVodStitchDetailsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_vod_stitch_details_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_vod_stitch_details), '__call__') as call:
        call.return_value = video_stitcher_service.ListVodStitchDetailsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(video_stitcher_service.ListVodStitchDetailsResponse())
        response = await client.list_vod_stitch_details(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_vod_stitch_details_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_vod_stitch_details(video_stitcher_service.ListVodStitchDetailsRequest(), parent='parent_value')

def test_list_vod_stitch_details_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_vod_stitch_details), '__call__') as call:
        call.side_effect = (video_stitcher_service.ListVodStitchDetailsResponse(vod_stitch_details=[stitch_details.VodStitchDetail(), stitch_details.VodStitchDetail(), stitch_details.VodStitchDetail()], next_page_token='abc'), video_stitcher_service.ListVodStitchDetailsResponse(vod_stitch_details=[], next_page_token='def'), video_stitcher_service.ListVodStitchDetailsResponse(vod_stitch_details=[stitch_details.VodStitchDetail()], next_page_token='ghi'), video_stitcher_service.ListVodStitchDetailsResponse(vod_stitch_details=[stitch_details.VodStitchDetail(), stitch_details.VodStitchDetail()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_vod_stitch_details(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, stitch_details.VodStitchDetail) for i in results))

def test_list_vod_stitch_details_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_vod_stitch_details), '__call__') as call:
        call.side_effect = (video_stitcher_service.ListVodStitchDetailsResponse(vod_stitch_details=[stitch_details.VodStitchDetail(), stitch_details.VodStitchDetail(), stitch_details.VodStitchDetail()], next_page_token='abc'), video_stitcher_service.ListVodStitchDetailsResponse(vod_stitch_details=[], next_page_token='def'), video_stitcher_service.ListVodStitchDetailsResponse(vod_stitch_details=[stitch_details.VodStitchDetail()], next_page_token='ghi'), video_stitcher_service.ListVodStitchDetailsResponse(vod_stitch_details=[stitch_details.VodStitchDetail(), stitch_details.VodStitchDetail()]), RuntimeError)
        pages = list(client.list_vod_stitch_details(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_vod_stitch_details_async_pager():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_vod_stitch_details), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (video_stitcher_service.ListVodStitchDetailsResponse(vod_stitch_details=[stitch_details.VodStitchDetail(), stitch_details.VodStitchDetail(), stitch_details.VodStitchDetail()], next_page_token='abc'), video_stitcher_service.ListVodStitchDetailsResponse(vod_stitch_details=[], next_page_token='def'), video_stitcher_service.ListVodStitchDetailsResponse(vod_stitch_details=[stitch_details.VodStitchDetail()], next_page_token='ghi'), video_stitcher_service.ListVodStitchDetailsResponse(vod_stitch_details=[stitch_details.VodStitchDetail(), stitch_details.VodStitchDetail()]), RuntimeError)
        async_pager = await client.list_vod_stitch_details(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, stitch_details.VodStitchDetail) for i in responses))

@pytest.mark.asyncio
async def test_list_vod_stitch_details_async_pages():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_vod_stitch_details), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (video_stitcher_service.ListVodStitchDetailsResponse(vod_stitch_details=[stitch_details.VodStitchDetail(), stitch_details.VodStitchDetail(), stitch_details.VodStitchDetail()], next_page_token='abc'), video_stitcher_service.ListVodStitchDetailsResponse(vod_stitch_details=[], next_page_token='def'), video_stitcher_service.ListVodStitchDetailsResponse(vod_stitch_details=[stitch_details.VodStitchDetail()], next_page_token='ghi'), video_stitcher_service.ListVodStitchDetailsResponse(vod_stitch_details=[stitch_details.VodStitchDetail(), stitch_details.VodStitchDetail()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_vod_stitch_details(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [video_stitcher_service.GetVodStitchDetailRequest, dict])
def test_get_vod_stitch_detail(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_vod_stitch_detail), '__call__') as call:
        call.return_value = stitch_details.VodStitchDetail(name='name_value')
        response = client.get_vod_stitch_detail(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetVodStitchDetailRequest()
    assert isinstance(response, stitch_details.VodStitchDetail)
    assert response.name == 'name_value'

def test_get_vod_stitch_detail_empty_call():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_vod_stitch_detail), '__call__') as call:
        client.get_vod_stitch_detail()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetVodStitchDetailRequest()

@pytest.mark.asyncio
async def test_get_vod_stitch_detail_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.GetVodStitchDetailRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_vod_stitch_detail), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(stitch_details.VodStitchDetail(name='name_value'))
        response = await client.get_vod_stitch_detail(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetVodStitchDetailRequest()
    assert isinstance(response, stitch_details.VodStitchDetail)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_vod_stitch_detail_async_from_dict():
    await test_get_vod_stitch_detail_async(request_type=dict)

def test_get_vod_stitch_detail_field_headers():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.GetVodStitchDetailRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_vod_stitch_detail), '__call__') as call:
        call.return_value = stitch_details.VodStitchDetail()
        client.get_vod_stitch_detail(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_vod_stitch_detail_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.GetVodStitchDetailRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_vod_stitch_detail), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(stitch_details.VodStitchDetail())
        await client.get_vod_stitch_detail(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_vod_stitch_detail_flattened():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_vod_stitch_detail), '__call__') as call:
        call.return_value = stitch_details.VodStitchDetail()
        client.get_vod_stitch_detail(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_vod_stitch_detail_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_vod_stitch_detail(video_stitcher_service.GetVodStitchDetailRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_vod_stitch_detail_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_vod_stitch_detail), '__call__') as call:
        call.return_value = stitch_details.VodStitchDetail()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(stitch_details.VodStitchDetail())
        response = await client.get_vod_stitch_detail(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_vod_stitch_detail_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_vod_stitch_detail(video_stitcher_service.GetVodStitchDetailRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [video_stitcher_service.ListVodAdTagDetailsRequest, dict])
def test_list_vod_ad_tag_details(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_vod_ad_tag_details), '__call__') as call:
        call.return_value = video_stitcher_service.ListVodAdTagDetailsResponse(next_page_token='next_page_token_value')
        response = client.list_vod_ad_tag_details(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.ListVodAdTagDetailsRequest()
    assert isinstance(response, pagers.ListVodAdTagDetailsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_vod_ad_tag_details_empty_call():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_vod_ad_tag_details), '__call__') as call:
        client.list_vod_ad_tag_details()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.ListVodAdTagDetailsRequest()

@pytest.mark.asyncio
async def test_list_vod_ad_tag_details_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.ListVodAdTagDetailsRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_vod_ad_tag_details), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(video_stitcher_service.ListVodAdTagDetailsResponse(next_page_token='next_page_token_value'))
        response = await client.list_vod_ad_tag_details(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.ListVodAdTagDetailsRequest()
    assert isinstance(response, pagers.ListVodAdTagDetailsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_vod_ad_tag_details_async_from_dict():
    await test_list_vod_ad_tag_details_async(request_type=dict)

def test_list_vod_ad_tag_details_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.ListVodAdTagDetailsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_vod_ad_tag_details), '__call__') as call:
        call.return_value = video_stitcher_service.ListVodAdTagDetailsResponse()
        client.list_vod_ad_tag_details(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_vod_ad_tag_details_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.ListVodAdTagDetailsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_vod_ad_tag_details), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(video_stitcher_service.ListVodAdTagDetailsResponse())
        await client.list_vod_ad_tag_details(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_vod_ad_tag_details_flattened():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_vod_ad_tag_details), '__call__') as call:
        call.return_value = video_stitcher_service.ListVodAdTagDetailsResponse()
        client.list_vod_ad_tag_details(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_vod_ad_tag_details_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_vod_ad_tag_details(video_stitcher_service.ListVodAdTagDetailsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_vod_ad_tag_details_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_vod_ad_tag_details), '__call__') as call:
        call.return_value = video_stitcher_service.ListVodAdTagDetailsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(video_stitcher_service.ListVodAdTagDetailsResponse())
        response = await client.list_vod_ad_tag_details(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_vod_ad_tag_details_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_vod_ad_tag_details(video_stitcher_service.ListVodAdTagDetailsRequest(), parent='parent_value')

def test_list_vod_ad_tag_details_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_vod_ad_tag_details), '__call__') as call:
        call.side_effect = (video_stitcher_service.ListVodAdTagDetailsResponse(vod_ad_tag_details=[ad_tag_details.VodAdTagDetail(), ad_tag_details.VodAdTagDetail(), ad_tag_details.VodAdTagDetail()], next_page_token='abc'), video_stitcher_service.ListVodAdTagDetailsResponse(vod_ad_tag_details=[], next_page_token='def'), video_stitcher_service.ListVodAdTagDetailsResponse(vod_ad_tag_details=[ad_tag_details.VodAdTagDetail()], next_page_token='ghi'), video_stitcher_service.ListVodAdTagDetailsResponse(vod_ad_tag_details=[ad_tag_details.VodAdTagDetail(), ad_tag_details.VodAdTagDetail()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_vod_ad_tag_details(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, ad_tag_details.VodAdTagDetail) for i in results))

def test_list_vod_ad_tag_details_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_vod_ad_tag_details), '__call__') as call:
        call.side_effect = (video_stitcher_service.ListVodAdTagDetailsResponse(vod_ad_tag_details=[ad_tag_details.VodAdTagDetail(), ad_tag_details.VodAdTagDetail(), ad_tag_details.VodAdTagDetail()], next_page_token='abc'), video_stitcher_service.ListVodAdTagDetailsResponse(vod_ad_tag_details=[], next_page_token='def'), video_stitcher_service.ListVodAdTagDetailsResponse(vod_ad_tag_details=[ad_tag_details.VodAdTagDetail()], next_page_token='ghi'), video_stitcher_service.ListVodAdTagDetailsResponse(vod_ad_tag_details=[ad_tag_details.VodAdTagDetail(), ad_tag_details.VodAdTagDetail()]), RuntimeError)
        pages = list(client.list_vod_ad_tag_details(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_vod_ad_tag_details_async_pager():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_vod_ad_tag_details), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (video_stitcher_service.ListVodAdTagDetailsResponse(vod_ad_tag_details=[ad_tag_details.VodAdTagDetail(), ad_tag_details.VodAdTagDetail(), ad_tag_details.VodAdTagDetail()], next_page_token='abc'), video_stitcher_service.ListVodAdTagDetailsResponse(vod_ad_tag_details=[], next_page_token='def'), video_stitcher_service.ListVodAdTagDetailsResponse(vod_ad_tag_details=[ad_tag_details.VodAdTagDetail()], next_page_token='ghi'), video_stitcher_service.ListVodAdTagDetailsResponse(vod_ad_tag_details=[ad_tag_details.VodAdTagDetail(), ad_tag_details.VodAdTagDetail()]), RuntimeError)
        async_pager = await client.list_vod_ad_tag_details(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, ad_tag_details.VodAdTagDetail) for i in responses))

@pytest.mark.asyncio
async def test_list_vod_ad_tag_details_async_pages():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_vod_ad_tag_details), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (video_stitcher_service.ListVodAdTagDetailsResponse(vod_ad_tag_details=[ad_tag_details.VodAdTagDetail(), ad_tag_details.VodAdTagDetail(), ad_tag_details.VodAdTagDetail()], next_page_token='abc'), video_stitcher_service.ListVodAdTagDetailsResponse(vod_ad_tag_details=[], next_page_token='def'), video_stitcher_service.ListVodAdTagDetailsResponse(vod_ad_tag_details=[ad_tag_details.VodAdTagDetail()], next_page_token='ghi'), video_stitcher_service.ListVodAdTagDetailsResponse(vod_ad_tag_details=[ad_tag_details.VodAdTagDetail(), ad_tag_details.VodAdTagDetail()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_vod_ad_tag_details(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [video_stitcher_service.GetVodAdTagDetailRequest, dict])
def test_get_vod_ad_tag_detail(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_vod_ad_tag_detail), '__call__') as call:
        call.return_value = ad_tag_details.VodAdTagDetail(name='name_value')
        response = client.get_vod_ad_tag_detail(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetVodAdTagDetailRequest()
    assert isinstance(response, ad_tag_details.VodAdTagDetail)
    assert response.name == 'name_value'

def test_get_vod_ad_tag_detail_empty_call():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_vod_ad_tag_detail), '__call__') as call:
        client.get_vod_ad_tag_detail()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetVodAdTagDetailRequest()

@pytest.mark.asyncio
async def test_get_vod_ad_tag_detail_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.GetVodAdTagDetailRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_vod_ad_tag_detail), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ad_tag_details.VodAdTagDetail(name='name_value'))
        response = await client.get_vod_ad_tag_detail(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetVodAdTagDetailRequest()
    assert isinstance(response, ad_tag_details.VodAdTagDetail)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_vod_ad_tag_detail_async_from_dict():
    await test_get_vod_ad_tag_detail_async(request_type=dict)

def test_get_vod_ad_tag_detail_field_headers():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.GetVodAdTagDetailRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_vod_ad_tag_detail), '__call__') as call:
        call.return_value = ad_tag_details.VodAdTagDetail()
        client.get_vod_ad_tag_detail(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_vod_ad_tag_detail_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.GetVodAdTagDetailRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_vod_ad_tag_detail), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ad_tag_details.VodAdTagDetail())
        await client.get_vod_ad_tag_detail(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_vod_ad_tag_detail_flattened():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_vod_ad_tag_detail), '__call__') as call:
        call.return_value = ad_tag_details.VodAdTagDetail()
        client.get_vod_ad_tag_detail(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_vod_ad_tag_detail_flattened_error():
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_vod_ad_tag_detail(video_stitcher_service.GetVodAdTagDetailRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_vod_ad_tag_detail_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_vod_ad_tag_detail), '__call__') as call:
        call.return_value = ad_tag_details.VodAdTagDetail()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ad_tag_details.VodAdTagDetail())
        response = await client.get_vod_ad_tag_detail(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_vod_ad_tag_detail_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_vod_ad_tag_detail(video_stitcher_service.GetVodAdTagDetailRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [video_stitcher_service.ListLiveAdTagDetailsRequest, dict])
def test_list_live_ad_tag_details(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_live_ad_tag_details), '__call__') as call:
        call.return_value = video_stitcher_service.ListLiveAdTagDetailsResponse(next_page_token='next_page_token_value')
        response = client.list_live_ad_tag_details(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.ListLiveAdTagDetailsRequest()
    assert isinstance(response, pagers.ListLiveAdTagDetailsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_live_ad_tag_details_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_live_ad_tag_details), '__call__') as call:
        client.list_live_ad_tag_details()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.ListLiveAdTagDetailsRequest()

@pytest.mark.asyncio
async def test_list_live_ad_tag_details_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.ListLiveAdTagDetailsRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_live_ad_tag_details), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(video_stitcher_service.ListLiveAdTagDetailsResponse(next_page_token='next_page_token_value'))
        response = await client.list_live_ad_tag_details(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.ListLiveAdTagDetailsRequest()
    assert isinstance(response, pagers.ListLiveAdTagDetailsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_live_ad_tag_details_async_from_dict():
    await test_list_live_ad_tag_details_async(request_type=dict)

def test_list_live_ad_tag_details_field_headers():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.ListLiveAdTagDetailsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_live_ad_tag_details), '__call__') as call:
        call.return_value = video_stitcher_service.ListLiveAdTagDetailsResponse()
        client.list_live_ad_tag_details(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_live_ad_tag_details_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.ListLiveAdTagDetailsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_live_ad_tag_details), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(video_stitcher_service.ListLiveAdTagDetailsResponse())
        await client.list_live_ad_tag_details(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_live_ad_tag_details_flattened():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_live_ad_tag_details), '__call__') as call:
        call.return_value = video_stitcher_service.ListLiveAdTagDetailsResponse()
        client.list_live_ad_tag_details(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_live_ad_tag_details_flattened_error():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_live_ad_tag_details(video_stitcher_service.ListLiveAdTagDetailsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_live_ad_tag_details_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_live_ad_tag_details), '__call__') as call:
        call.return_value = video_stitcher_service.ListLiveAdTagDetailsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(video_stitcher_service.ListLiveAdTagDetailsResponse())
        response = await client.list_live_ad_tag_details(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_live_ad_tag_details_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_live_ad_tag_details(video_stitcher_service.ListLiveAdTagDetailsRequest(), parent='parent_value')

def test_list_live_ad_tag_details_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_live_ad_tag_details), '__call__') as call:
        call.side_effect = (video_stitcher_service.ListLiveAdTagDetailsResponse(live_ad_tag_details=[ad_tag_details.LiveAdTagDetail(), ad_tag_details.LiveAdTagDetail(), ad_tag_details.LiveAdTagDetail()], next_page_token='abc'), video_stitcher_service.ListLiveAdTagDetailsResponse(live_ad_tag_details=[], next_page_token='def'), video_stitcher_service.ListLiveAdTagDetailsResponse(live_ad_tag_details=[ad_tag_details.LiveAdTagDetail()], next_page_token='ghi'), video_stitcher_service.ListLiveAdTagDetailsResponse(live_ad_tag_details=[ad_tag_details.LiveAdTagDetail(), ad_tag_details.LiveAdTagDetail()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_live_ad_tag_details(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, ad_tag_details.LiveAdTagDetail) for i in results))

def test_list_live_ad_tag_details_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_live_ad_tag_details), '__call__') as call:
        call.side_effect = (video_stitcher_service.ListLiveAdTagDetailsResponse(live_ad_tag_details=[ad_tag_details.LiveAdTagDetail(), ad_tag_details.LiveAdTagDetail(), ad_tag_details.LiveAdTagDetail()], next_page_token='abc'), video_stitcher_service.ListLiveAdTagDetailsResponse(live_ad_tag_details=[], next_page_token='def'), video_stitcher_service.ListLiveAdTagDetailsResponse(live_ad_tag_details=[ad_tag_details.LiveAdTagDetail()], next_page_token='ghi'), video_stitcher_service.ListLiveAdTagDetailsResponse(live_ad_tag_details=[ad_tag_details.LiveAdTagDetail(), ad_tag_details.LiveAdTagDetail()]), RuntimeError)
        pages = list(client.list_live_ad_tag_details(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_live_ad_tag_details_async_pager():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_live_ad_tag_details), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (video_stitcher_service.ListLiveAdTagDetailsResponse(live_ad_tag_details=[ad_tag_details.LiveAdTagDetail(), ad_tag_details.LiveAdTagDetail(), ad_tag_details.LiveAdTagDetail()], next_page_token='abc'), video_stitcher_service.ListLiveAdTagDetailsResponse(live_ad_tag_details=[], next_page_token='def'), video_stitcher_service.ListLiveAdTagDetailsResponse(live_ad_tag_details=[ad_tag_details.LiveAdTagDetail()], next_page_token='ghi'), video_stitcher_service.ListLiveAdTagDetailsResponse(live_ad_tag_details=[ad_tag_details.LiveAdTagDetail(), ad_tag_details.LiveAdTagDetail()]), RuntimeError)
        async_pager = await client.list_live_ad_tag_details(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, ad_tag_details.LiveAdTagDetail) for i in responses))

@pytest.mark.asyncio
async def test_list_live_ad_tag_details_async_pages():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_live_ad_tag_details), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (video_stitcher_service.ListLiveAdTagDetailsResponse(live_ad_tag_details=[ad_tag_details.LiveAdTagDetail(), ad_tag_details.LiveAdTagDetail(), ad_tag_details.LiveAdTagDetail()], next_page_token='abc'), video_stitcher_service.ListLiveAdTagDetailsResponse(live_ad_tag_details=[], next_page_token='def'), video_stitcher_service.ListLiveAdTagDetailsResponse(live_ad_tag_details=[ad_tag_details.LiveAdTagDetail()], next_page_token='ghi'), video_stitcher_service.ListLiveAdTagDetailsResponse(live_ad_tag_details=[ad_tag_details.LiveAdTagDetail(), ad_tag_details.LiveAdTagDetail()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_live_ad_tag_details(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [video_stitcher_service.GetLiveAdTagDetailRequest, dict])
def test_get_live_ad_tag_detail(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_live_ad_tag_detail), '__call__') as call:
        call.return_value = ad_tag_details.LiveAdTagDetail(name='name_value')
        response = client.get_live_ad_tag_detail(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetLiveAdTagDetailRequest()
    assert isinstance(response, ad_tag_details.LiveAdTagDetail)
    assert response.name == 'name_value'

def test_get_live_ad_tag_detail_empty_call():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_live_ad_tag_detail), '__call__') as call:
        client.get_live_ad_tag_detail()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetLiveAdTagDetailRequest()

@pytest.mark.asyncio
async def test_get_live_ad_tag_detail_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.GetLiveAdTagDetailRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_live_ad_tag_detail), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ad_tag_details.LiveAdTagDetail(name='name_value'))
        response = await client.get_live_ad_tag_detail(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetLiveAdTagDetailRequest()
    assert isinstance(response, ad_tag_details.LiveAdTagDetail)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_live_ad_tag_detail_async_from_dict():
    await test_get_live_ad_tag_detail_async(request_type=dict)

def test_get_live_ad_tag_detail_field_headers():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.GetLiveAdTagDetailRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_live_ad_tag_detail), '__call__') as call:
        call.return_value = ad_tag_details.LiveAdTagDetail()
        client.get_live_ad_tag_detail(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_live_ad_tag_detail_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.GetLiveAdTagDetailRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_live_ad_tag_detail), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ad_tag_details.LiveAdTagDetail())
        await client.get_live_ad_tag_detail(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_live_ad_tag_detail_flattened():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_live_ad_tag_detail), '__call__') as call:
        call.return_value = ad_tag_details.LiveAdTagDetail()
        client.get_live_ad_tag_detail(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_live_ad_tag_detail_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_live_ad_tag_detail(video_stitcher_service.GetLiveAdTagDetailRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_live_ad_tag_detail_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_live_ad_tag_detail), '__call__') as call:
        call.return_value = ad_tag_details.LiveAdTagDetail()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(ad_tag_details.LiveAdTagDetail())
        response = await client.get_live_ad_tag_detail(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_live_ad_tag_detail_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_live_ad_tag_detail(video_stitcher_service.GetLiveAdTagDetailRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [video_stitcher_service.CreateSlateRequest, dict])
def test_create_slate(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_slate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_slate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.CreateSlateRequest()
    assert isinstance(response, future.Future)

def test_create_slate_empty_call():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_slate), '__call__') as call:
        client.create_slate()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.CreateSlateRequest()

@pytest.mark.asyncio
async def test_create_slate_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.CreateSlateRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_slate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_slate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.CreateSlateRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_slate_async_from_dict():
    await test_create_slate_async(request_type=dict)

def test_create_slate_field_headers():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.CreateSlateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_slate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_slate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_slate_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.CreateSlateRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_slate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_slate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_slate_flattened():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_slate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_slate(parent='parent_value', slate=slates.Slate(name='name_value'), slate_id='slate_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].slate
        mock_val = slates.Slate(name='name_value')
        assert arg == mock_val
        arg = args[0].slate_id
        mock_val = 'slate_id_value'
        assert arg == mock_val

def test_create_slate_flattened_error():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_slate(video_stitcher_service.CreateSlateRequest(), parent='parent_value', slate=slates.Slate(name='name_value'), slate_id='slate_id_value')

@pytest.mark.asyncio
async def test_create_slate_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_slate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_slate(parent='parent_value', slate=slates.Slate(name='name_value'), slate_id='slate_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].slate
        mock_val = slates.Slate(name='name_value')
        assert arg == mock_val
        arg = args[0].slate_id
        mock_val = 'slate_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_slate_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_slate(video_stitcher_service.CreateSlateRequest(), parent='parent_value', slate=slates.Slate(name='name_value'), slate_id='slate_id_value')

@pytest.mark.parametrize('request_type', [video_stitcher_service.ListSlatesRequest, dict])
def test_list_slates(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_slates), '__call__') as call:
        call.return_value = video_stitcher_service.ListSlatesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_slates(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.ListSlatesRequest()
    assert isinstance(response, pagers.ListSlatesPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_slates_empty_call():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_slates), '__call__') as call:
        client.list_slates()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.ListSlatesRequest()

@pytest.mark.asyncio
async def test_list_slates_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.ListSlatesRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_slates), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(video_stitcher_service.ListSlatesResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_slates(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.ListSlatesRequest()
    assert isinstance(response, pagers.ListSlatesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_slates_async_from_dict():
    await test_list_slates_async(request_type=dict)

def test_list_slates_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.ListSlatesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_slates), '__call__') as call:
        call.return_value = video_stitcher_service.ListSlatesResponse()
        client.list_slates(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_slates_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.ListSlatesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_slates), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(video_stitcher_service.ListSlatesResponse())
        await client.list_slates(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_slates_flattened():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_slates), '__call__') as call:
        call.return_value = video_stitcher_service.ListSlatesResponse()
        client.list_slates(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_slates_flattened_error():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_slates(video_stitcher_service.ListSlatesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_slates_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_slates), '__call__') as call:
        call.return_value = video_stitcher_service.ListSlatesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(video_stitcher_service.ListSlatesResponse())
        response = await client.list_slates(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_slates_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_slates(video_stitcher_service.ListSlatesRequest(), parent='parent_value')

def test_list_slates_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_slates), '__call__') as call:
        call.side_effect = (video_stitcher_service.ListSlatesResponse(slates=[slates.Slate(), slates.Slate(), slates.Slate()], next_page_token='abc'), video_stitcher_service.ListSlatesResponse(slates=[], next_page_token='def'), video_stitcher_service.ListSlatesResponse(slates=[slates.Slate()], next_page_token='ghi'), video_stitcher_service.ListSlatesResponse(slates=[slates.Slate(), slates.Slate()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_slates(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, slates.Slate) for i in results))

def test_list_slates_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_slates), '__call__') as call:
        call.side_effect = (video_stitcher_service.ListSlatesResponse(slates=[slates.Slate(), slates.Slate(), slates.Slate()], next_page_token='abc'), video_stitcher_service.ListSlatesResponse(slates=[], next_page_token='def'), video_stitcher_service.ListSlatesResponse(slates=[slates.Slate()], next_page_token='ghi'), video_stitcher_service.ListSlatesResponse(slates=[slates.Slate(), slates.Slate()]), RuntimeError)
        pages = list(client.list_slates(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_slates_async_pager():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_slates), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (video_stitcher_service.ListSlatesResponse(slates=[slates.Slate(), slates.Slate(), slates.Slate()], next_page_token='abc'), video_stitcher_service.ListSlatesResponse(slates=[], next_page_token='def'), video_stitcher_service.ListSlatesResponse(slates=[slates.Slate()], next_page_token='ghi'), video_stitcher_service.ListSlatesResponse(slates=[slates.Slate(), slates.Slate()]), RuntimeError)
        async_pager = await client.list_slates(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, slates.Slate) for i in responses))

@pytest.mark.asyncio
async def test_list_slates_async_pages():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_slates), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (video_stitcher_service.ListSlatesResponse(slates=[slates.Slate(), slates.Slate(), slates.Slate()], next_page_token='abc'), video_stitcher_service.ListSlatesResponse(slates=[], next_page_token='def'), video_stitcher_service.ListSlatesResponse(slates=[slates.Slate()], next_page_token='ghi'), video_stitcher_service.ListSlatesResponse(slates=[slates.Slate(), slates.Slate()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_slates(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [video_stitcher_service.GetSlateRequest, dict])
def test_get_slate(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_slate), '__call__') as call:
        call.return_value = slates.Slate(name='name_value', uri='uri_value')
        response = client.get_slate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetSlateRequest()
    assert isinstance(response, slates.Slate)
    assert response.name == 'name_value'
    assert response.uri == 'uri_value'

def test_get_slate_empty_call():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_slate), '__call__') as call:
        client.get_slate()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetSlateRequest()

@pytest.mark.asyncio
async def test_get_slate_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.GetSlateRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_slate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(slates.Slate(name='name_value', uri='uri_value'))
        response = await client.get_slate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetSlateRequest()
    assert isinstance(response, slates.Slate)
    assert response.name == 'name_value'
    assert response.uri == 'uri_value'

@pytest.mark.asyncio
async def test_get_slate_async_from_dict():
    await test_get_slate_async(request_type=dict)

def test_get_slate_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.GetSlateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_slate), '__call__') as call:
        call.return_value = slates.Slate()
        client.get_slate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_slate_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.GetSlateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_slate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(slates.Slate())
        await client.get_slate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_slate_flattened():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_slate), '__call__') as call:
        call.return_value = slates.Slate()
        client.get_slate(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_slate_flattened_error():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_slate(video_stitcher_service.GetSlateRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_slate_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_slate), '__call__') as call:
        call.return_value = slates.Slate()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(slates.Slate())
        response = await client.get_slate(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_slate_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_slate(video_stitcher_service.GetSlateRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [video_stitcher_service.UpdateSlateRequest, dict])
def test_update_slate(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_slate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_slate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.UpdateSlateRequest()
    assert isinstance(response, future.Future)

def test_update_slate_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_slate), '__call__') as call:
        client.update_slate()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.UpdateSlateRequest()

@pytest.mark.asyncio
async def test_update_slate_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.UpdateSlateRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_slate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_slate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.UpdateSlateRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_slate_async_from_dict():
    await test_update_slate_async(request_type=dict)

def test_update_slate_field_headers():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.UpdateSlateRequest()
    request.slate.name = 'name_value'
    with mock.patch.object(type(client.transport.update_slate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_slate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'slate.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_slate_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.UpdateSlateRequest()
    request.slate.name = 'name_value'
    with mock.patch.object(type(client.transport.update_slate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_slate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'slate.name=name_value') in kw['metadata']

def test_update_slate_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_slate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_slate(slate=slates.Slate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].slate
        mock_val = slates.Slate(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_slate_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_slate(video_stitcher_service.UpdateSlateRequest(), slate=slates.Slate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_slate_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_slate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_slate(slate=slates.Slate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].slate
        mock_val = slates.Slate(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_slate_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_slate(video_stitcher_service.UpdateSlateRequest(), slate=slates.Slate(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [video_stitcher_service.DeleteSlateRequest, dict])
def test_delete_slate(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_slate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_slate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.DeleteSlateRequest()
    assert isinstance(response, future.Future)

def test_delete_slate_empty_call():
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_slate), '__call__') as call:
        client.delete_slate()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.DeleteSlateRequest()

@pytest.mark.asyncio
async def test_delete_slate_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.DeleteSlateRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_slate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_slate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.DeleteSlateRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_slate_async_from_dict():
    await test_delete_slate_async(request_type=dict)

def test_delete_slate_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.DeleteSlateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_slate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_slate(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_slate_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.DeleteSlateRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_slate), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_slate(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_slate_flattened():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_slate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_slate(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_slate_flattened_error():
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_slate(video_stitcher_service.DeleteSlateRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_slate_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_slate), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_slate(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_slate_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_slate(video_stitcher_service.DeleteSlateRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [video_stitcher_service.CreateLiveSessionRequest, dict])
def test_create_live_session(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_live_session), '__call__') as call:
        call.return_value = sessions.LiveSession(name='name_value', play_uri='play_uri_value', live_config='live_config_value')
        response = client.create_live_session(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.CreateLiveSessionRequest()
    assert isinstance(response, sessions.LiveSession)
    assert response.name == 'name_value'
    assert response.play_uri == 'play_uri_value'
    assert response.live_config == 'live_config_value'

def test_create_live_session_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_live_session), '__call__') as call:
        client.create_live_session()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.CreateLiveSessionRequest()

@pytest.mark.asyncio
async def test_create_live_session_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.CreateLiveSessionRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_live_session), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(sessions.LiveSession(name='name_value', play_uri='play_uri_value', live_config='live_config_value'))
        response = await client.create_live_session(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.CreateLiveSessionRequest()
    assert isinstance(response, sessions.LiveSession)
    assert response.name == 'name_value'
    assert response.play_uri == 'play_uri_value'
    assert response.live_config == 'live_config_value'

@pytest.mark.asyncio
async def test_create_live_session_async_from_dict():
    await test_create_live_session_async(request_type=dict)

def test_create_live_session_field_headers():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.CreateLiveSessionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_live_session), '__call__') as call:
        call.return_value = sessions.LiveSession()
        client.create_live_session(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_live_session_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.CreateLiveSessionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_live_session), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(sessions.LiveSession())
        await client.create_live_session(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_live_session_flattened():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_live_session), '__call__') as call:
        call.return_value = sessions.LiveSession()
        client.create_live_session(parent='parent_value', live_session=sessions.LiveSession(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].live_session
        mock_val = sessions.LiveSession(name='name_value')
        assert arg == mock_val

def test_create_live_session_flattened_error():
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_live_session(video_stitcher_service.CreateLiveSessionRequest(), parent='parent_value', live_session=sessions.LiveSession(name='name_value'))

@pytest.mark.asyncio
async def test_create_live_session_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_live_session), '__call__') as call:
        call.return_value = sessions.LiveSession()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(sessions.LiveSession())
        response = await client.create_live_session(parent='parent_value', live_session=sessions.LiveSession(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].live_session
        mock_val = sessions.LiveSession(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_live_session_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_live_session(video_stitcher_service.CreateLiveSessionRequest(), parent='parent_value', live_session=sessions.LiveSession(name='name_value'))

@pytest.mark.parametrize('request_type', [video_stitcher_service.GetLiveSessionRequest, dict])
def test_get_live_session(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_live_session), '__call__') as call:
        call.return_value = sessions.LiveSession(name='name_value', play_uri='play_uri_value', live_config='live_config_value')
        response = client.get_live_session(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetLiveSessionRequest()
    assert isinstance(response, sessions.LiveSession)
    assert response.name == 'name_value'
    assert response.play_uri == 'play_uri_value'
    assert response.live_config == 'live_config_value'

def test_get_live_session_empty_call():
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_live_session), '__call__') as call:
        client.get_live_session()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetLiveSessionRequest()

@pytest.mark.asyncio
async def test_get_live_session_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.GetLiveSessionRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_live_session), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(sessions.LiveSession(name='name_value', play_uri='play_uri_value', live_config='live_config_value'))
        response = await client.get_live_session(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetLiveSessionRequest()
    assert isinstance(response, sessions.LiveSession)
    assert response.name == 'name_value'
    assert response.play_uri == 'play_uri_value'
    assert response.live_config == 'live_config_value'

@pytest.mark.asyncio
async def test_get_live_session_async_from_dict():
    await test_get_live_session_async(request_type=dict)

def test_get_live_session_field_headers():
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.GetLiveSessionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_live_session), '__call__') as call:
        call.return_value = sessions.LiveSession()
        client.get_live_session(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_live_session_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.GetLiveSessionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_live_session), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(sessions.LiveSession())
        await client.get_live_session(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_live_session_flattened():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_live_session), '__call__') as call:
        call.return_value = sessions.LiveSession()
        client.get_live_session(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_live_session_flattened_error():
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_live_session(video_stitcher_service.GetLiveSessionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_live_session_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_live_session), '__call__') as call:
        call.return_value = sessions.LiveSession()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(sessions.LiveSession())
        response = await client.get_live_session(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_live_session_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_live_session(video_stitcher_service.GetLiveSessionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [video_stitcher_service.CreateLiveConfigRequest, dict])
def test_create_live_config(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_live_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_live_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.CreateLiveConfigRequest()
    assert isinstance(response, future.Future)

def test_create_live_config_empty_call():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_live_config), '__call__') as call:
        client.create_live_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.CreateLiveConfigRequest()

@pytest.mark.asyncio
async def test_create_live_config_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.CreateLiveConfigRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_live_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_live_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.CreateLiveConfigRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_live_config_async_from_dict():
    await test_create_live_config_async(request_type=dict)

def test_create_live_config_field_headers():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.CreateLiveConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_live_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_live_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_live_config_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.CreateLiveConfigRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_live_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_live_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_live_config_flattened():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_live_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_live_config(parent='parent_value', live_config=live_configs.LiveConfig(name='name_value'), live_config_id='live_config_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].live_config
        mock_val = live_configs.LiveConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].live_config_id
        mock_val = 'live_config_id_value'
        assert arg == mock_val

def test_create_live_config_flattened_error():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_live_config(video_stitcher_service.CreateLiveConfigRequest(), parent='parent_value', live_config=live_configs.LiveConfig(name='name_value'), live_config_id='live_config_id_value')

@pytest.mark.asyncio
async def test_create_live_config_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_live_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_live_config(parent='parent_value', live_config=live_configs.LiveConfig(name='name_value'), live_config_id='live_config_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].live_config
        mock_val = live_configs.LiveConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].live_config_id
        mock_val = 'live_config_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_live_config_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_live_config(video_stitcher_service.CreateLiveConfigRequest(), parent='parent_value', live_config=live_configs.LiveConfig(name='name_value'), live_config_id='live_config_id_value')

@pytest.mark.parametrize('request_type', [video_stitcher_service.ListLiveConfigsRequest, dict])
def test_list_live_configs(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_live_configs), '__call__') as call:
        call.return_value = video_stitcher_service.ListLiveConfigsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_live_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.ListLiveConfigsRequest()
    assert isinstance(response, pagers.ListLiveConfigsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_live_configs_empty_call():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_live_configs), '__call__') as call:
        client.list_live_configs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.ListLiveConfigsRequest()

@pytest.mark.asyncio
async def test_list_live_configs_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.ListLiveConfigsRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_live_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(video_stitcher_service.ListLiveConfigsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_live_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.ListLiveConfigsRequest()
    assert isinstance(response, pagers.ListLiveConfigsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_live_configs_async_from_dict():
    await test_list_live_configs_async(request_type=dict)

def test_list_live_configs_field_headers():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.ListLiveConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_live_configs), '__call__') as call:
        call.return_value = video_stitcher_service.ListLiveConfigsResponse()
        client.list_live_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_live_configs_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.ListLiveConfigsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_live_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(video_stitcher_service.ListLiveConfigsResponse())
        await client.list_live_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_live_configs_flattened():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_live_configs), '__call__') as call:
        call.return_value = video_stitcher_service.ListLiveConfigsResponse()
        client.list_live_configs(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_live_configs_flattened_error():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_live_configs(video_stitcher_service.ListLiveConfigsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_live_configs_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_live_configs), '__call__') as call:
        call.return_value = video_stitcher_service.ListLiveConfigsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(video_stitcher_service.ListLiveConfigsResponse())
        response = await client.list_live_configs(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_live_configs_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_live_configs(video_stitcher_service.ListLiveConfigsRequest(), parent='parent_value')

def test_list_live_configs_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_live_configs), '__call__') as call:
        call.side_effect = (video_stitcher_service.ListLiveConfigsResponse(live_configs=[live_configs.LiveConfig(), live_configs.LiveConfig(), live_configs.LiveConfig()], next_page_token='abc'), video_stitcher_service.ListLiveConfigsResponse(live_configs=[], next_page_token='def'), video_stitcher_service.ListLiveConfigsResponse(live_configs=[live_configs.LiveConfig()], next_page_token='ghi'), video_stitcher_service.ListLiveConfigsResponse(live_configs=[live_configs.LiveConfig(), live_configs.LiveConfig()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_live_configs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, live_configs.LiveConfig) for i in results))

def test_list_live_configs_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_live_configs), '__call__') as call:
        call.side_effect = (video_stitcher_service.ListLiveConfigsResponse(live_configs=[live_configs.LiveConfig(), live_configs.LiveConfig(), live_configs.LiveConfig()], next_page_token='abc'), video_stitcher_service.ListLiveConfigsResponse(live_configs=[], next_page_token='def'), video_stitcher_service.ListLiveConfigsResponse(live_configs=[live_configs.LiveConfig()], next_page_token='ghi'), video_stitcher_service.ListLiveConfigsResponse(live_configs=[live_configs.LiveConfig(), live_configs.LiveConfig()]), RuntimeError)
        pages = list(client.list_live_configs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_live_configs_async_pager():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_live_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (video_stitcher_service.ListLiveConfigsResponse(live_configs=[live_configs.LiveConfig(), live_configs.LiveConfig(), live_configs.LiveConfig()], next_page_token='abc'), video_stitcher_service.ListLiveConfigsResponse(live_configs=[], next_page_token='def'), video_stitcher_service.ListLiveConfigsResponse(live_configs=[live_configs.LiveConfig()], next_page_token='ghi'), video_stitcher_service.ListLiveConfigsResponse(live_configs=[live_configs.LiveConfig(), live_configs.LiveConfig()]), RuntimeError)
        async_pager = await client.list_live_configs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, live_configs.LiveConfig) for i in responses))

@pytest.mark.asyncio
async def test_list_live_configs_async_pages():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_live_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (video_stitcher_service.ListLiveConfigsResponse(live_configs=[live_configs.LiveConfig(), live_configs.LiveConfig(), live_configs.LiveConfig()], next_page_token='abc'), video_stitcher_service.ListLiveConfigsResponse(live_configs=[], next_page_token='def'), video_stitcher_service.ListLiveConfigsResponse(live_configs=[live_configs.LiveConfig()], next_page_token='ghi'), video_stitcher_service.ListLiveConfigsResponse(live_configs=[live_configs.LiveConfig(), live_configs.LiveConfig()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_live_configs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [video_stitcher_service.GetLiveConfigRequest, dict])
def test_get_live_config(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_live_config), '__call__') as call:
        call.return_value = live_configs.LiveConfig(name='name_value', source_uri='source_uri_value', ad_tag_uri='ad_tag_uri_value', state=live_configs.LiveConfig.State.CREATING, ad_tracking=live_configs.AdTracking.CLIENT, default_slate='default_slate_value', stitching_policy=live_configs.LiveConfig.StitchingPolicy.CUT_CURRENT)
        response = client.get_live_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetLiveConfigRequest()
    assert isinstance(response, live_configs.LiveConfig)
    assert response.name == 'name_value'
    assert response.source_uri == 'source_uri_value'
    assert response.ad_tag_uri == 'ad_tag_uri_value'
    assert response.state == live_configs.LiveConfig.State.CREATING
    assert response.ad_tracking == live_configs.AdTracking.CLIENT
    assert response.default_slate == 'default_slate_value'
    assert response.stitching_policy == live_configs.LiveConfig.StitchingPolicy.CUT_CURRENT

def test_get_live_config_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_live_config), '__call__') as call:
        client.get_live_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetLiveConfigRequest()

@pytest.mark.asyncio
async def test_get_live_config_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.GetLiveConfigRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_live_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(live_configs.LiveConfig(name='name_value', source_uri='source_uri_value', ad_tag_uri='ad_tag_uri_value', state=live_configs.LiveConfig.State.CREATING, ad_tracking=live_configs.AdTracking.CLIENT, default_slate='default_slate_value', stitching_policy=live_configs.LiveConfig.StitchingPolicy.CUT_CURRENT))
        response = await client.get_live_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.GetLiveConfigRequest()
    assert isinstance(response, live_configs.LiveConfig)
    assert response.name == 'name_value'
    assert response.source_uri == 'source_uri_value'
    assert response.ad_tag_uri == 'ad_tag_uri_value'
    assert response.state == live_configs.LiveConfig.State.CREATING
    assert response.ad_tracking == live_configs.AdTracking.CLIENT
    assert response.default_slate == 'default_slate_value'
    assert response.stitching_policy == live_configs.LiveConfig.StitchingPolicy.CUT_CURRENT

@pytest.mark.asyncio
async def test_get_live_config_async_from_dict():
    await test_get_live_config_async(request_type=dict)

def test_get_live_config_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.GetLiveConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_live_config), '__call__') as call:
        call.return_value = live_configs.LiveConfig()
        client.get_live_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_live_config_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.GetLiveConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_live_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(live_configs.LiveConfig())
        await client.get_live_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_live_config_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_live_config), '__call__') as call:
        call.return_value = live_configs.LiveConfig()
        client.get_live_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_live_config_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_live_config(video_stitcher_service.GetLiveConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_live_config_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_live_config), '__call__') as call:
        call.return_value = live_configs.LiveConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(live_configs.LiveConfig())
        response = await client.get_live_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_live_config_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_live_config(video_stitcher_service.GetLiveConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [video_stitcher_service.DeleteLiveConfigRequest, dict])
def test_delete_live_config(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_live_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_live_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.DeleteLiveConfigRequest()
    assert isinstance(response, future.Future)

def test_delete_live_config_empty_call():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_live_config), '__call__') as call:
        client.delete_live_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.DeleteLiveConfigRequest()

@pytest.mark.asyncio
async def test_delete_live_config_async(transport: str='grpc_asyncio', request_type=video_stitcher_service.DeleteLiveConfigRequest):
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_live_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_live_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == video_stitcher_service.DeleteLiveConfigRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_live_config_async_from_dict():
    await test_delete_live_config_async(request_type=dict)

def test_delete_live_config_field_headers():
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.DeleteLiveConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_live_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_live_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_live_config_field_headers_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = video_stitcher_service.DeleteLiveConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_live_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_live_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_live_config_flattened():
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_live_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_live_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_live_config_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_live_config(video_stitcher_service.DeleteLiveConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_live_config_flattened_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_live_config), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_live_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_live_config_flattened_error_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_live_config(video_stitcher_service.DeleteLiveConfigRequest(), name='name_value')

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.VideoStitcherServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.VideoStitcherServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = VideoStitcherServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.VideoStitcherServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = VideoStitcherServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = VideoStitcherServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.VideoStitcherServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = VideoStitcherServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VideoStitcherServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = VideoStitcherServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        print('Hello World!')
    transport = transports.VideoStitcherServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.VideoStitcherServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.VideoStitcherServiceGrpcTransport, transports.VideoStitcherServiceGrpcAsyncIOTransport])
def test_transport_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc'])
def test_transport_kind(transport_name):
    if False:
        print('Hello World!')
    transport = VideoStitcherServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.VideoStitcherServiceGrpcTransport)

def test_video_stitcher_service_base_transport_error():
    if False:
        return 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.VideoStitcherServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_video_stitcher_service_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.video.stitcher_v1.services.video_stitcher_service.transports.VideoStitcherServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.VideoStitcherServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_cdn_key', 'list_cdn_keys', 'get_cdn_key', 'delete_cdn_key', 'update_cdn_key', 'create_vod_session', 'get_vod_session', 'list_vod_stitch_details', 'get_vod_stitch_detail', 'list_vod_ad_tag_details', 'get_vod_ad_tag_detail', 'list_live_ad_tag_details', 'get_live_ad_tag_detail', 'create_slate', 'list_slates', 'get_slate', 'update_slate', 'delete_slate', 'create_live_session', 'get_live_session', 'create_live_config', 'list_live_configs', 'get_live_config', 'delete_live_config', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_video_stitcher_service_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.video.stitcher_v1.services.video_stitcher_service.transports.VideoStitcherServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.VideoStitcherServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_video_stitcher_service_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.video.stitcher_v1.services.video_stitcher_service.transports.VideoStitcherServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.VideoStitcherServiceTransport()
        adc.assert_called_once()

def test_video_stitcher_service_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        VideoStitcherServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.VideoStitcherServiceGrpcTransport, transports.VideoStitcherServiceGrpcAsyncIOTransport])
def test_video_stitcher_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.VideoStitcherServiceGrpcTransport, transports.VideoStitcherServiceGrpcAsyncIOTransport])
def test_video_stitcher_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.VideoStitcherServiceGrpcTransport, grpc_helpers), (transports.VideoStitcherServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_video_stitcher_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('videostitcher.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='videostitcher.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.VideoStitcherServiceGrpcTransport, transports.VideoStitcherServiceGrpcAsyncIOTransport])
def test_video_stitcher_service_grpc_transport_client_cert_source_for_mtls(transport_class):
    if False:
        print('Hello World!')
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

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_video_stitcher_service_host_no_port(transport_name):
    if False:
        return 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='videostitcher.googleapis.com'), transport=transport_name)
    assert client.transport._host == 'videostitcher.googleapis.com:443'

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_video_stitcher_service_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='videostitcher.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == 'videostitcher.googleapis.com:8000'

def test_video_stitcher_service_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.VideoStitcherServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_video_stitcher_service_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.VideoStitcherServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.VideoStitcherServiceGrpcTransport, transports.VideoStitcherServiceGrpcAsyncIOTransport])
def test_video_stitcher_service_transport_channel_mtls_with_client_cert_source(transport_class):
    if False:
        print('Hello World!')
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

@pytest.mark.parametrize('transport_class', [transports.VideoStitcherServiceGrpcTransport, transports.VideoStitcherServiceGrpcAsyncIOTransport])
def test_video_stitcher_service_transport_channel_mtls_with_adc(transport_class):
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

def test_video_stitcher_service_grpc_lro_client():
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_video_stitcher_service_grpc_lro_async_client():
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_cdn_key_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    cdn_key = 'whelk'
    expected = 'projects/{project}/locations/{location}/cdnKeys/{cdn_key}'.format(project=project, location=location, cdn_key=cdn_key)
    actual = VideoStitcherServiceClient.cdn_key_path(project, location, cdn_key)
    assert expected == actual

def test_parse_cdn_key_path():
    if False:
        return 10
    expected = {'project': 'octopus', 'location': 'oyster', 'cdn_key': 'nudibranch'}
    path = VideoStitcherServiceClient.cdn_key_path(**expected)
    actual = VideoStitcherServiceClient.parse_cdn_key_path(path)
    assert expected == actual

def test_live_ad_tag_detail_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    location = 'mussel'
    live_session = 'winkle'
    live_ad_tag_detail = 'nautilus'
    expected = 'projects/{project}/locations/{location}/liveSessions/{live_session}/liveAdTagDetails/{live_ad_tag_detail}'.format(project=project, location=location, live_session=live_session, live_ad_tag_detail=live_ad_tag_detail)
    actual = VideoStitcherServiceClient.live_ad_tag_detail_path(project, location, live_session, live_ad_tag_detail)
    assert expected == actual

def test_parse_live_ad_tag_detail_path():
    if False:
        return 10
    expected = {'project': 'scallop', 'location': 'abalone', 'live_session': 'squid', 'live_ad_tag_detail': 'clam'}
    path = VideoStitcherServiceClient.live_ad_tag_detail_path(**expected)
    actual = VideoStitcherServiceClient.parse_live_ad_tag_detail_path(path)
    assert expected == actual

def test_live_config_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'whelk'
    location = 'octopus'
    live_config = 'oyster'
    expected = 'projects/{project}/locations/{location}/liveConfigs/{live_config}'.format(project=project, location=location, live_config=live_config)
    actual = VideoStitcherServiceClient.live_config_path(project, location, live_config)
    assert expected == actual

def test_parse_live_config_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'live_config': 'mussel'}
    path = VideoStitcherServiceClient.live_config_path(**expected)
    actual = VideoStitcherServiceClient.parse_live_config_path(path)
    assert expected == actual

def test_live_session_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    location = 'nautilus'
    live_session = 'scallop'
    expected = 'projects/{project}/locations/{location}/liveSessions/{live_session}'.format(project=project, location=location, live_session=live_session)
    actual = VideoStitcherServiceClient.live_session_path(project, location, live_session)
    assert expected == actual

def test_parse_live_session_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'abalone', 'location': 'squid', 'live_session': 'clam'}
    path = VideoStitcherServiceClient.live_session_path(**expected)
    actual = VideoStitcherServiceClient.parse_live_session_path(path)
    assert expected == actual

def test_slate_path():
    if False:
        i = 10
        return i + 15
    project = 'whelk'
    location = 'octopus'
    slate = 'oyster'
    expected = 'projects/{project}/locations/{location}/slates/{slate}'.format(project=project, location=location, slate=slate)
    actual = VideoStitcherServiceClient.slate_path(project, location, slate)
    assert expected == actual

def test_parse_slate_path():
    if False:
        return 10
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'slate': 'mussel'}
    path = VideoStitcherServiceClient.slate_path(**expected)
    actual = VideoStitcherServiceClient.parse_slate_path(path)
    assert expected == actual

def test_vod_ad_tag_detail_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    location = 'nautilus'
    vod_session = 'scallop'
    vod_ad_tag_detail = 'abalone'
    expected = 'projects/{project}/locations/{location}/vodSessions/{vod_session}/vodAdTagDetails/{vod_ad_tag_detail}'.format(project=project, location=location, vod_session=vod_session, vod_ad_tag_detail=vod_ad_tag_detail)
    actual = VideoStitcherServiceClient.vod_ad_tag_detail_path(project, location, vod_session, vod_ad_tag_detail)
    assert expected == actual

def test_parse_vod_ad_tag_detail_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'squid', 'location': 'clam', 'vod_session': 'whelk', 'vod_ad_tag_detail': 'octopus'}
    path = VideoStitcherServiceClient.vod_ad_tag_detail_path(**expected)
    actual = VideoStitcherServiceClient.parse_vod_ad_tag_detail_path(path)
    assert expected == actual

def test_vod_session_path():
    if False:
        return 10
    project = 'oyster'
    location = 'nudibranch'
    vod_session = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/vodSessions/{vod_session}'.format(project=project, location=location, vod_session=vod_session)
    actual = VideoStitcherServiceClient.vod_session_path(project, location, vod_session)
    assert expected == actual

def test_parse_vod_session_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'mussel', 'location': 'winkle', 'vod_session': 'nautilus'}
    path = VideoStitcherServiceClient.vod_session_path(**expected)
    actual = VideoStitcherServiceClient.parse_vod_session_path(path)
    assert expected == actual

def test_vod_stitch_detail_path():
    if False:
        i = 10
        return i + 15
    project = 'scallop'
    location = 'abalone'
    vod_session = 'squid'
    vod_stitch_detail = 'clam'
    expected = 'projects/{project}/locations/{location}/vodSessions/{vod_session}/vodStitchDetails/{vod_stitch_detail}'.format(project=project, location=location, vod_session=vod_session, vod_stitch_detail=vod_stitch_detail)
    actual = VideoStitcherServiceClient.vod_stitch_detail_path(project, location, vod_session, vod_stitch_detail)
    assert expected == actual

def test_parse_vod_stitch_detail_path():
    if False:
        print('Hello World!')
    expected = {'project': 'whelk', 'location': 'octopus', 'vod_session': 'oyster', 'vod_stitch_detail': 'nudibranch'}
    path = VideoStitcherServiceClient.vod_stitch_detail_path(**expected)
    actual = VideoStitcherServiceClient.parse_vod_stitch_detail_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = VideoStitcherServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'mussel'}
    path = VideoStitcherServiceClient.common_billing_account_path(**expected)
    actual = VideoStitcherServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = VideoStitcherServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'nautilus'}
    path = VideoStitcherServiceClient.common_folder_path(**expected)
    actual = VideoStitcherServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = VideoStitcherServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'abalone'}
    path = VideoStitcherServiceClient.common_organization_path(**expected)
    actual = VideoStitcherServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = VideoStitcherServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'clam'}
    path = VideoStitcherServiceClient.common_project_path(**expected)
    actual = VideoStitcherServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = VideoStitcherServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = VideoStitcherServiceClient.common_location_path(**expected)
    actual = VideoStitcherServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.VideoStitcherServiceTransport, '_prep_wrapped_messages') as prep:
        client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.VideoStitcherServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = VideoStitcherServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_delete_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = VideoStitcherServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        for i in range(10):
            print('nop')
    transports = {'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['grpc']
    for transport in transports:
        client = VideoStitcherServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(VideoStitcherServiceClient, transports.VideoStitcherServiceGrpcTransport), (VideoStitcherServiceAsyncClient, transports.VideoStitcherServiceGrpcAsyncIOTransport)])
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