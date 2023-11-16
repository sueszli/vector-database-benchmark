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
from google.protobuf import json_format
from google.type import latlng_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.vision_v1p3beta1.services.image_annotator import ImageAnnotatorAsyncClient, ImageAnnotatorClient, transports
from google.cloud.vision_v1p3beta1.types import geometry, image_annotator, product_search

def client_cert_source_callback():
    if False:
        while True:
            i = 10
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        while True:
            i = 10
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
    assert ImageAnnotatorClient._get_default_mtls_endpoint(None) is None
    assert ImageAnnotatorClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert ImageAnnotatorClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert ImageAnnotatorClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert ImageAnnotatorClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert ImageAnnotatorClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(ImageAnnotatorClient, 'grpc'), (ImageAnnotatorAsyncClient, 'grpc_asyncio'), (ImageAnnotatorClient, 'rest')])
def test_image_annotator_client_from_service_account_info(client_class, transport_name):
    if False:
        while True:
            i = 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('vision.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://vision.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.ImageAnnotatorGrpcTransport, 'grpc'), (transports.ImageAnnotatorGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.ImageAnnotatorRestTransport, 'rest')])
def test_image_annotator_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(ImageAnnotatorClient, 'grpc'), (ImageAnnotatorAsyncClient, 'grpc_asyncio'), (ImageAnnotatorClient, 'rest')])
def test_image_annotator_client_from_service_account_file(client_class, transport_name):
    if False:
        print('Hello World!')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_file') as factory:
        factory.return_value = creds
        client = client_class.from_service_account_file('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        client = client_class.from_service_account_json('dummy/file/path.json', transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('vision.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://vision.googleapis.com')

def test_image_annotator_client_get_transport_class():
    if False:
        for i in range(10):
            print('nop')
    transport = ImageAnnotatorClient.get_transport_class()
    available_transports = [transports.ImageAnnotatorGrpcTransport, transports.ImageAnnotatorRestTransport]
    assert transport in available_transports
    transport = ImageAnnotatorClient.get_transport_class('grpc')
    assert transport == transports.ImageAnnotatorGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ImageAnnotatorClient, transports.ImageAnnotatorGrpcTransport, 'grpc'), (ImageAnnotatorAsyncClient, transports.ImageAnnotatorGrpcAsyncIOTransport, 'grpc_asyncio'), (ImageAnnotatorClient, transports.ImageAnnotatorRestTransport, 'rest')])
@mock.patch.object(ImageAnnotatorClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ImageAnnotatorClient))
@mock.patch.object(ImageAnnotatorAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ImageAnnotatorAsyncClient))
def test_image_annotator_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(ImageAnnotatorClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(ImageAnnotatorClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(ImageAnnotatorClient, transports.ImageAnnotatorGrpcTransport, 'grpc', 'true'), (ImageAnnotatorAsyncClient, transports.ImageAnnotatorGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (ImageAnnotatorClient, transports.ImageAnnotatorGrpcTransport, 'grpc', 'false'), (ImageAnnotatorAsyncClient, transports.ImageAnnotatorGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (ImageAnnotatorClient, transports.ImageAnnotatorRestTransport, 'rest', 'true'), (ImageAnnotatorClient, transports.ImageAnnotatorRestTransport, 'rest', 'false')])
@mock.patch.object(ImageAnnotatorClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ImageAnnotatorClient))
@mock.patch.object(ImageAnnotatorAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ImageAnnotatorAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_image_annotator_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
    if False:
        i = 10
        return i + 15
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

@pytest.mark.parametrize('client_class', [ImageAnnotatorClient, ImageAnnotatorAsyncClient])
@mock.patch.object(ImageAnnotatorClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ImageAnnotatorClient))
@mock.patch.object(ImageAnnotatorAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ImageAnnotatorAsyncClient))
def test_image_annotator_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ImageAnnotatorClient, transports.ImageAnnotatorGrpcTransport, 'grpc'), (ImageAnnotatorAsyncClient, transports.ImageAnnotatorGrpcAsyncIOTransport, 'grpc_asyncio'), (ImageAnnotatorClient, transports.ImageAnnotatorRestTransport, 'rest')])
def test_image_annotator_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ImageAnnotatorClient, transports.ImageAnnotatorGrpcTransport, 'grpc', grpc_helpers), (ImageAnnotatorAsyncClient, transports.ImageAnnotatorGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (ImageAnnotatorClient, transports.ImageAnnotatorRestTransport, 'rest', None)])
def test_image_annotator_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_image_annotator_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.vision_v1p3beta1.services.image_annotator.transports.ImageAnnotatorGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = ImageAnnotatorClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ImageAnnotatorClient, transports.ImageAnnotatorGrpcTransport, 'grpc', grpc_helpers), (ImageAnnotatorAsyncClient, transports.ImageAnnotatorGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_image_annotator_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('vision.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-vision'), scopes=None, default_host='vision.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [image_annotator.BatchAnnotateImagesRequest, dict])
def test_batch_annotate_images(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_annotate_images), '__call__') as call:
        call.return_value = image_annotator.BatchAnnotateImagesResponse()
        response = client.batch_annotate_images(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == image_annotator.BatchAnnotateImagesRequest()
    assert isinstance(response, image_annotator.BatchAnnotateImagesResponse)

def test_batch_annotate_images_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.batch_annotate_images), '__call__') as call:
        client.batch_annotate_images()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == image_annotator.BatchAnnotateImagesRequest()

@pytest.mark.asyncio
async def test_batch_annotate_images_async(transport: str='grpc_asyncio', request_type=image_annotator.BatchAnnotateImagesRequest):
    client = ImageAnnotatorAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.batch_annotate_images), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(image_annotator.BatchAnnotateImagesResponse())
        response = await client.batch_annotate_images(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == image_annotator.BatchAnnotateImagesRequest()
    assert isinstance(response, image_annotator.BatchAnnotateImagesResponse)

@pytest.mark.asyncio
async def test_batch_annotate_images_async_from_dict():
    await test_batch_annotate_images_async(request_type=dict)

def test_batch_annotate_images_flattened():
    if False:
        while True:
            i = 10
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_annotate_images), '__call__') as call:
        call.return_value = image_annotator.BatchAnnotateImagesResponse()
        client.batch_annotate_images(requests=[image_annotator.AnnotateImageRequest(image=image_annotator.Image(content=b'content_blob'))])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].requests
        mock_val = [image_annotator.AnnotateImageRequest(image=image_annotator.Image(content=b'content_blob'))]
        assert arg == mock_val

def test_batch_annotate_images_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.batch_annotate_images(image_annotator.BatchAnnotateImagesRequest(), requests=[image_annotator.AnnotateImageRequest(image=image_annotator.Image(content=b'content_blob'))])

@pytest.mark.asyncio
async def test_batch_annotate_images_flattened_async():
    client = ImageAnnotatorAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.batch_annotate_images), '__call__') as call:
        call.return_value = image_annotator.BatchAnnotateImagesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(image_annotator.BatchAnnotateImagesResponse())
        response = await client.batch_annotate_images(requests=[image_annotator.AnnotateImageRequest(image=image_annotator.Image(content=b'content_blob'))])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].requests
        mock_val = [image_annotator.AnnotateImageRequest(image=image_annotator.Image(content=b'content_blob'))]
        assert arg == mock_val

@pytest.mark.asyncio
async def test_batch_annotate_images_flattened_error_async():
    client = ImageAnnotatorAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.batch_annotate_images(image_annotator.BatchAnnotateImagesRequest(), requests=[image_annotator.AnnotateImageRequest(image=image_annotator.Image(content=b'content_blob'))])

@pytest.mark.parametrize('request_type', [image_annotator.AsyncBatchAnnotateFilesRequest, dict])
def test_async_batch_annotate_files(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.async_batch_annotate_files), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.async_batch_annotate_files(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == image_annotator.AsyncBatchAnnotateFilesRequest()
    assert isinstance(response, future.Future)

def test_async_batch_annotate_files_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.async_batch_annotate_files), '__call__') as call:
        client.async_batch_annotate_files()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == image_annotator.AsyncBatchAnnotateFilesRequest()

@pytest.mark.asyncio
async def test_async_batch_annotate_files_async(transport: str='grpc_asyncio', request_type=image_annotator.AsyncBatchAnnotateFilesRequest):
    client = ImageAnnotatorAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.async_batch_annotate_files), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.async_batch_annotate_files(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == image_annotator.AsyncBatchAnnotateFilesRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_async_batch_annotate_files_async_from_dict():
    await test_async_batch_annotate_files_async(request_type=dict)

def test_async_batch_annotate_files_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.async_batch_annotate_files), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.async_batch_annotate_files(requests=[image_annotator.AsyncAnnotateFileRequest(input_config=image_annotator.InputConfig(gcs_source=image_annotator.GcsSource(uri='uri_value')))])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].requests
        mock_val = [image_annotator.AsyncAnnotateFileRequest(input_config=image_annotator.InputConfig(gcs_source=image_annotator.GcsSource(uri='uri_value')))]
        assert arg == mock_val

def test_async_batch_annotate_files_flattened_error():
    if False:
        return 10
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.async_batch_annotate_files(image_annotator.AsyncBatchAnnotateFilesRequest(), requests=[image_annotator.AsyncAnnotateFileRequest(input_config=image_annotator.InputConfig(gcs_source=image_annotator.GcsSource(uri='uri_value')))])

@pytest.mark.asyncio
async def test_async_batch_annotate_files_flattened_async():
    client = ImageAnnotatorAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.async_batch_annotate_files), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.async_batch_annotate_files(requests=[image_annotator.AsyncAnnotateFileRequest(input_config=image_annotator.InputConfig(gcs_source=image_annotator.GcsSource(uri='uri_value')))])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].requests
        mock_val = [image_annotator.AsyncAnnotateFileRequest(input_config=image_annotator.InputConfig(gcs_source=image_annotator.GcsSource(uri='uri_value')))]
        assert arg == mock_val

@pytest.mark.asyncio
async def test_async_batch_annotate_files_flattened_error_async():
    client = ImageAnnotatorAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.async_batch_annotate_files(image_annotator.AsyncBatchAnnotateFilesRequest(), requests=[image_annotator.AsyncAnnotateFileRequest(input_config=image_annotator.InputConfig(gcs_source=image_annotator.GcsSource(uri='uri_value')))])

@pytest.mark.parametrize('request_type', [image_annotator.BatchAnnotateImagesRequest, dict])
def test_batch_annotate_images_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = image_annotator.BatchAnnotateImagesResponse()
        response_value = Response()
        response_value.status_code = 200
        return_value = image_annotator.BatchAnnotateImagesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.batch_annotate_images(request)
    assert isinstance(response, image_annotator.BatchAnnotateImagesResponse)

def test_batch_annotate_images_rest_required_fields(request_type=image_annotator.BatchAnnotateImagesRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ImageAnnotatorRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_annotate_images._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).batch_annotate_images._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = image_annotator.BatchAnnotateImagesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = image_annotator.BatchAnnotateImagesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.batch_annotate_images(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_batch_annotate_images_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ImageAnnotatorRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.batch_annotate_images._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('requests',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_batch_annotate_images_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ImageAnnotatorRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ImageAnnotatorRestInterceptor())
    client = ImageAnnotatorClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ImageAnnotatorRestInterceptor, 'post_batch_annotate_images') as post, mock.patch.object(transports.ImageAnnotatorRestInterceptor, 'pre_batch_annotate_images') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = image_annotator.BatchAnnotateImagesRequest.pb(image_annotator.BatchAnnotateImagesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = image_annotator.BatchAnnotateImagesResponse.to_json(image_annotator.BatchAnnotateImagesResponse())
        request = image_annotator.BatchAnnotateImagesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = image_annotator.BatchAnnotateImagesResponse()
        client.batch_annotate_images(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_batch_annotate_images_rest_bad_request(transport: str='rest', request_type=image_annotator.BatchAnnotateImagesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.batch_annotate_images(request)

def test_batch_annotate_images_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = image_annotator.BatchAnnotateImagesResponse()
        sample_request = {}
        mock_args = dict(requests=[image_annotator.AnnotateImageRequest(image=image_annotator.Image(content=b'content_blob'))])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = image_annotator.BatchAnnotateImagesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.batch_annotate_images(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1p3beta1/images:annotate' % client.transport._host, args[1])

def test_batch_annotate_images_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.batch_annotate_images(image_annotator.BatchAnnotateImagesRequest(), requests=[image_annotator.AnnotateImageRequest(image=image_annotator.Image(content=b'content_blob'))])

def test_batch_annotate_images_rest_error():
    if False:
        print('Hello World!')
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [image_annotator.AsyncBatchAnnotateFilesRequest, dict])
def test_async_batch_annotate_files_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.async_batch_annotate_files(request)
    assert response.operation.name == 'operations/spam'

def test_async_batch_annotate_files_rest_required_fields(request_type=image_annotator.AsyncBatchAnnotateFilesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ImageAnnotatorRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).async_batch_annotate_files._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).async_batch_annotate_files._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.async_batch_annotate_files(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_async_batch_annotate_files_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ImageAnnotatorRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.async_batch_annotate_files._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('requests',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_async_batch_annotate_files_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ImageAnnotatorRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ImageAnnotatorRestInterceptor())
    client = ImageAnnotatorClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ImageAnnotatorRestInterceptor, 'post_async_batch_annotate_files') as post, mock.patch.object(transports.ImageAnnotatorRestInterceptor, 'pre_async_batch_annotate_files') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = image_annotator.AsyncBatchAnnotateFilesRequest.pb(image_annotator.AsyncBatchAnnotateFilesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = image_annotator.AsyncBatchAnnotateFilesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.async_batch_annotate_files(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_async_batch_annotate_files_rest_bad_request(transport: str='rest', request_type=image_annotator.AsyncBatchAnnotateFilesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.async_batch_annotate_files(request)

def test_async_batch_annotate_files_rest_flattened():
    if False:
        print('Hello World!')
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {}
        mock_args = dict(requests=[image_annotator.AsyncAnnotateFileRequest(input_config=image_annotator.InputConfig(gcs_source=image_annotator.GcsSource(uri='uri_value')))])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.async_batch_annotate_files(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1p3beta1/files:asyncBatchAnnotate' % client.transport._host, args[1])

def test_async_batch_annotate_files_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.async_batch_annotate_files(image_annotator.AsyncBatchAnnotateFilesRequest(), requests=[image_annotator.AsyncAnnotateFileRequest(input_config=image_annotator.InputConfig(gcs_source=image_annotator.GcsSource(uri='uri_value')))])

def test_async_batch_annotate_files_rest_error():
    if False:
        print('Hello World!')
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.ImageAnnotatorGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.ImageAnnotatorGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ImageAnnotatorClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.ImageAnnotatorGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ImageAnnotatorClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ImageAnnotatorClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.ImageAnnotatorGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ImageAnnotatorClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.ImageAnnotatorGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = ImageAnnotatorClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ImageAnnotatorGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.ImageAnnotatorGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.ImageAnnotatorGrpcTransport, transports.ImageAnnotatorGrpcAsyncIOTransport, transports.ImageAnnotatorRestTransport])
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
    transport = ImageAnnotatorClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.ImageAnnotatorGrpcTransport)

def test_image_annotator_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.ImageAnnotatorTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_image_annotator_base_transport():
    if False:
        return 10
    with mock.patch('google.cloud.vision_v1p3beta1.services.image_annotator.transports.ImageAnnotatorTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.ImageAnnotatorTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('batch_annotate_images', 'async_batch_annotate_files')
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

def test_image_annotator_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.vision_v1p3beta1.services.image_annotator.transports.ImageAnnotatorTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ImageAnnotatorTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-vision'), quota_project_id='octopus')

def test_image_annotator_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.vision_v1p3beta1.services.image_annotator.transports.ImageAnnotatorTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ImageAnnotatorTransport()
        adc.assert_called_once()

def test_image_annotator_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        ImageAnnotatorClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-vision'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.ImageAnnotatorGrpcTransport, transports.ImageAnnotatorGrpcAsyncIOTransport])
def test_image_annotator_transport_auth_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-vision'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.ImageAnnotatorGrpcTransport, transports.ImageAnnotatorGrpcAsyncIOTransport, transports.ImageAnnotatorRestTransport])
def test_image_annotator_transport_auth_gdch_credentials(transport_class):
    if False:
        return 10
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.ImageAnnotatorGrpcTransport, grpc_helpers), (transports.ImageAnnotatorGrpcAsyncIOTransport, grpc_helpers_async)])
def test_image_annotator_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('vision.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-vision'), scopes=['1', '2'], default_host='vision.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.ImageAnnotatorGrpcTransport, transports.ImageAnnotatorGrpcAsyncIOTransport])
def test_image_annotator_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_image_annotator_http_transport_client_cert_source_for_mtls():
    if False:
        for i in range(10):
            print('nop')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.ImageAnnotatorRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_image_annotator_rest_lro_client():
    if False:
        return 10
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_image_annotator_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='vision.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('vision.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://vision.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_image_annotator_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='vision.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('vision.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://vision.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_image_annotator_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = ImageAnnotatorClient(credentials=creds1, transport=transport_name)
    client2 = ImageAnnotatorClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.batch_annotate_images._session
    session2 = client2.transport.batch_annotate_images._session
    assert session1 != session2
    session1 = client1.transport.async_batch_annotate_files._session
    session2 = client2.transport.async_batch_annotate_files._session
    assert session1 != session2

def test_image_annotator_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ImageAnnotatorGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_image_annotator_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ImageAnnotatorGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.ImageAnnotatorGrpcTransport, transports.ImageAnnotatorGrpcAsyncIOTransport])
def test_image_annotator_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.ImageAnnotatorGrpcTransport, transports.ImageAnnotatorGrpcAsyncIOTransport])
def test_image_annotator_transport_channel_mtls_with_adc(transport_class):
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

def test_image_annotator_grpc_lro_client():
    if False:
        while True:
            i = 10
    client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_image_annotator_grpc_lro_async_client():
    if False:
        while True:
            i = 10
    client = ImageAnnotatorAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_product_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    product = 'whelk'
    expected = 'projects/{project}/locations/{location}/products/{product}'.format(project=project, location=location, product=product)
    actual = ImageAnnotatorClient.product_path(project, location, product)
    assert expected == actual

def test_parse_product_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus', 'location': 'oyster', 'product': 'nudibranch'}
    path = ImageAnnotatorClient.product_path(**expected)
    actual = ImageAnnotatorClient.parse_product_path(path)
    assert expected == actual

def test_product_set_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    location = 'mussel'
    product_set = 'winkle'
    expected = 'projects/{project}/locations/{location}/productSets/{product_set}'.format(project=project, location=location, product_set=product_set)
    actual = ImageAnnotatorClient.product_set_path(project, location, product_set)
    assert expected == actual

def test_parse_product_set_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'nautilus', 'location': 'scallop', 'product_set': 'abalone'}
    path = ImageAnnotatorClient.product_set_path(**expected)
    actual = ImageAnnotatorClient.parse_product_set_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = ImageAnnotatorClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'clam'}
    path = ImageAnnotatorClient.common_billing_account_path(**expected)
    actual = ImageAnnotatorClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = ImageAnnotatorClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'octopus'}
    path = ImageAnnotatorClient.common_folder_path(**expected)
    actual = ImageAnnotatorClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = ImageAnnotatorClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'nudibranch'}
    path = ImageAnnotatorClient.common_organization_path(**expected)
    actual = ImageAnnotatorClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = ImageAnnotatorClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'mussel'}
    path = ImageAnnotatorClient.common_project_path(**expected)
    actual = ImageAnnotatorClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = ImageAnnotatorClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = ImageAnnotatorClient.common_location_path(**expected)
    actual = ImageAnnotatorClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.ImageAnnotatorTransport, '_prep_wrapped_messages') as prep:
        client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.ImageAnnotatorTransport, '_prep_wrapped_messages') as prep:
        transport_class = ImageAnnotatorClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = ImageAnnotatorAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
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
        client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = ImageAnnotatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(ImageAnnotatorClient, transports.ImageAnnotatorGrpcTransport), (ImageAnnotatorAsyncClient, transports.ImageAnnotatorGrpcAsyncIOTransport)])
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