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
from google.iam.v1 import iam_policy_pb2
from google.iam.v1 import options_pb2
from google.iam.v1 import policy_pb2
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
from google.cloud.servicedirectory_v1beta1.services.registration_service import RegistrationServiceAsyncClient, RegistrationServiceClient, pagers, transports
from google.cloud.servicedirectory_v1beta1.types import endpoint as gcs_endpoint
from google.cloud.servicedirectory_v1beta1.types import namespace as gcs_namespace
from google.cloud.servicedirectory_v1beta1.types import endpoint
from google.cloud.servicedirectory_v1beta1.types import namespace
from google.cloud.servicedirectory_v1beta1.types import registration_service
from google.cloud.servicedirectory_v1beta1.types import service
from google.cloud.servicedirectory_v1beta1.types import service as gcs_service

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
        for i in range(10):
            print('nop')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert RegistrationServiceClient._get_default_mtls_endpoint(None) is None
    assert RegistrationServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert RegistrationServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert RegistrationServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert RegistrationServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert RegistrationServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(RegistrationServiceClient, 'grpc'), (RegistrationServiceAsyncClient, 'grpc_asyncio'), (RegistrationServiceClient, 'rest')])
def test_registration_service_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('servicedirectory.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://servicedirectory.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.RegistrationServiceGrpcTransport, 'grpc'), (transports.RegistrationServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.RegistrationServiceRestTransport, 'rest')])
def test_registration_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(RegistrationServiceClient, 'grpc'), (RegistrationServiceAsyncClient, 'grpc_asyncio'), (RegistrationServiceClient, 'rest')])
def test_registration_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('servicedirectory.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://servicedirectory.googleapis.com')

def test_registration_service_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = RegistrationServiceClient.get_transport_class()
    available_transports = [transports.RegistrationServiceGrpcTransport, transports.RegistrationServiceRestTransport]
    assert transport in available_transports
    transport = RegistrationServiceClient.get_transport_class('grpc')
    assert transport == transports.RegistrationServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(RegistrationServiceClient, transports.RegistrationServiceGrpcTransport, 'grpc'), (RegistrationServiceAsyncClient, transports.RegistrationServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (RegistrationServiceClient, transports.RegistrationServiceRestTransport, 'rest')])
@mock.patch.object(RegistrationServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RegistrationServiceClient))
@mock.patch.object(RegistrationServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RegistrationServiceAsyncClient))
def test_registration_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(RegistrationServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(RegistrationServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(RegistrationServiceClient, transports.RegistrationServiceGrpcTransport, 'grpc', 'true'), (RegistrationServiceAsyncClient, transports.RegistrationServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (RegistrationServiceClient, transports.RegistrationServiceGrpcTransport, 'grpc', 'false'), (RegistrationServiceAsyncClient, transports.RegistrationServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (RegistrationServiceClient, transports.RegistrationServiceRestTransport, 'rest', 'true'), (RegistrationServiceClient, transports.RegistrationServiceRestTransport, 'rest', 'false')])
@mock.patch.object(RegistrationServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RegistrationServiceClient))
@mock.patch.object(RegistrationServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RegistrationServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_registration_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [RegistrationServiceClient, RegistrationServiceAsyncClient])
@mock.patch.object(RegistrationServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RegistrationServiceClient))
@mock.patch.object(RegistrationServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RegistrationServiceAsyncClient))
def test_registration_service_client_get_mtls_endpoint_and_cert_source(client_class):
    if False:
        i = 10
        return i + 15
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(RegistrationServiceClient, transports.RegistrationServiceGrpcTransport, 'grpc'), (RegistrationServiceAsyncClient, transports.RegistrationServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (RegistrationServiceClient, transports.RegistrationServiceRestTransport, 'rest')])
def test_registration_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(RegistrationServiceClient, transports.RegistrationServiceGrpcTransport, 'grpc', grpc_helpers), (RegistrationServiceAsyncClient, transports.RegistrationServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (RegistrationServiceClient, transports.RegistrationServiceRestTransport, 'rest', None)])
def test_registration_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_registration_service_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.servicedirectory_v1beta1.services.registration_service.transports.RegistrationServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = RegistrationServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(RegistrationServiceClient, transports.RegistrationServiceGrpcTransport, 'grpc', grpc_helpers), (RegistrationServiceAsyncClient, transports.RegistrationServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_registration_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('servicedirectory.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='servicedirectory.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [registration_service.CreateNamespaceRequest, dict])
def test_create_namespace(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_namespace), '__call__') as call:
        call.return_value = gcs_namespace.Namespace(name='name_value', uid='uid_value')
        response = client.create_namespace(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.CreateNamespaceRequest()
    assert isinstance(response, gcs_namespace.Namespace)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'

def test_create_namespace_empty_call():
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_namespace), '__call__') as call:
        client.create_namespace()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.CreateNamespaceRequest()

@pytest.mark.asyncio
async def test_create_namespace_async(transport: str='grpc_asyncio', request_type=registration_service.CreateNamespaceRequest):
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_namespace), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_namespace.Namespace(name='name_value', uid='uid_value'))
        response = await client.create_namespace(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.CreateNamespaceRequest()
    assert isinstance(response, gcs_namespace.Namespace)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'

@pytest.mark.asyncio
async def test_create_namespace_async_from_dict():
    await test_create_namespace_async(request_type=dict)

def test_create_namespace_field_headers():
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.CreateNamespaceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_namespace), '__call__') as call:
        call.return_value = gcs_namespace.Namespace()
        client.create_namespace(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_namespace_field_headers_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.CreateNamespaceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_namespace), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_namespace.Namespace())
        await client.create_namespace(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_namespace_flattened():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_namespace), '__call__') as call:
        call.return_value = gcs_namespace.Namespace()
        client.create_namespace(parent='parent_value', namespace=gcs_namespace.Namespace(name='name_value'), namespace_id='namespace_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].namespace
        mock_val = gcs_namespace.Namespace(name='name_value')
        assert arg == mock_val
        arg = args[0].namespace_id
        mock_val = 'namespace_id_value'
        assert arg == mock_val

def test_create_namespace_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_namespace(registration_service.CreateNamespaceRequest(), parent='parent_value', namespace=gcs_namespace.Namespace(name='name_value'), namespace_id='namespace_id_value')

@pytest.mark.asyncio
async def test_create_namespace_flattened_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_namespace), '__call__') as call:
        call.return_value = gcs_namespace.Namespace()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_namespace.Namespace())
        response = await client.create_namespace(parent='parent_value', namespace=gcs_namespace.Namespace(name='name_value'), namespace_id='namespace_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].namespace
        mock_val = gcs_namespace.Namespace(name='name_value')
        assert arg == mock_val
        arg = args[0].namespace_id
        mock_val = 'namespace_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_namespace_flattened_error_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_namespace(registration_service.CreateNamespaceRequest(), parent='parent_value', namespace=gcs_namespace.Namespace(name='name_value'), namespace_id='namespace_id_value')

@pytest.mark.parametrize('request_type', [registration_service.ListNamespacesRequest, dict])
def test_list_namespaces(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_namespaces), '__call__') as call:
        call.return_value = registration_service.ListNamespacesResponse(next_page_token='next_page_token_value')
        response = client.list_namespaces(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.ListNamespacesRequest()
    assert isinstance(response, pagers.ListNamespacesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_namespaces_empty_call():
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_namespaces), '__call__') as call:
        client.list_namespaces()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.ListNamespacesRequest()

@pytest.mark.asyncio
async def test_list_namespaces_async(transport: str='grpc_asyncio', request_type=registration_service.ListNamespacesRequest):
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_namespaces), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(registration_service.ListNamespacesResponse(next_page_token='next_page_token_value'))
        response = await client.list_namespaces(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.ListNamespacesRequest()
    assert isinstance(response, pagers.ListNamespacesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_namespaces_async_from_dict():
    await test_list_namespaces_async(request_type=dict)

def test_list_namespaces_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.ListNamespacesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_namespaces), '__call__') as call:
        call.return_value = registration_service.ListNamespacesResponse()
        client.list_namespaces(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_namespaces_field_headers_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.ListNamespacesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_namespaces), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(registration_service.ListNamespacesResponse())
        await client.list_namespaces(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_namespaces_flattened():
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_namespaces), '__call__') as call:
        call.return_value = registration_service.ListNamespacesResponse()
        client.list_namespaces(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_namespaces_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_namespaces(registration_service.ListNamespacesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_namespaces_flattened_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_namespaces), '__call__') as call:
        call.return_value = registration_service.ListNamespacesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(registration_service.ListNamespacesResponse())
        response = await client.list_namespaces(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_namespaces_flattened_error_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_namespaces(registration_service.ListNamespacesRequest(), parent='parent_value')

def test_list_namespaces_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_namespaces), '__call__') as call:
        call.side_effect = (registration_service.ListNamespacesResponse(namespaces=[namespace.Namespace(), namespace.Namespace(), namespace.Namespace()], next_page_token='abc'), registration_service.ListNamespacesResponse(namespaces=[], next_page_token='def'), registration_service.ListNamespacesResponse(namespaces=[namespace.Namespace()], next_page_token='ghi'), registration_service.ListNamespacesResponse(namespaces=[namespace.Namespace(), namespace.Namespace()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_namespaces(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, namespace.Namespace) for i in results))

def test_list_namespaces_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_namespaces), '__call__') as call:
        call.side_effect = (registration_service.ListNamespacesResponse(namespaces=[namespace.Namespace(), namespace.Namespace(), namespace.Namespace()], next_page_token='abc'), registration_service.ListNamespacesResponse(namespaces=[], next_page_token='def'), registration_service.ListNamespacesResponse(namespaces=[namespace.Namespace()], next_page_token='ghi'), registration_service.ListNamespacesResponse(namespaces=[namespace.Namespace(), namespace.Namespace()]), RuntimeError)
        pages = list(client.list_namespaces(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_namespaces_async_pager():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_namespaces), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (registration_service.ListNamespacesResponse(namespaces=[namespace.Namespace(), namespace.Namespace(), namespace.Namespace()], next_page_token='abc'), registration_service.ListNamespacesResponse(namespaces=[], next_page_token='def'), registration_service.ListNamespacesResponse(namespaces=[namespace.Namespace()], next_page_token='ghi'), registration_service.ListNamespacesResponse(namespaces=[namespace.Namespace(), namespace.Namespace()]), RuntimeError)
        async_pager = await client.list_namespaces(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, namespace.Namespace) for i in responses))

@pytest.mark.asyncio
async def test_list_namespaces_async_pages():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_namespaces), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (registration_service.ListNamespacesResponse(namespaces=[namespace.Namespace(), namespace.Namespace(), namespace.Namespace()], next_page_token='abc'), registration_service.ListNamespacesResponse(namespaces=[], next_page_token='def'), registration_service.ListNamespacesResponse(namespaces=[namespace.Namespace()], next_page_token='ghi'), registration_service.ListNamespacesResponse(namespaces=[namespace.Namespace(), namespace.Namespace()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_namespaces(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [registration_service.GetNamespaceRequest, dict])
def test_get_namespace(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_namespace), '__call__') as call:
        call.return_value = namespace.Namespace(name='name_value', uid='uid_value')
        response = client.get_namespace(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.GetNamespaceRequest()
    assert isinstance(response, namespace.Namespace)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'

def test_get_namespace_empty_call():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_namespace), '__call__') as call:
        client.get_namespace()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.GetNamespaceRequest()

@pytest.mark.asyncio
async def test_get_namespace_async(transport: str='grpc_asyncio', request_type=registration_service.GetNamespaceRequest):
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_namespace), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(namespace.Namespace(name='name_value', uid='uid_value'))
        response = await client.get_namespace(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.GetNamespaceRequest()
    assert isinstance(response, namespace.Namespace)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'

@pytest.mark.asyncio
async def test_get_namespace_async_from_dict():
    await test_get_namespace_async(request_type=dict)

def test_get_namespace_field_headers():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.GetNamespaceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_namespace), '__call__') as call:
        call.return_value = namespace.Namespace()
        client.get_namespace(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_namespace_field_headers_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.GetNamespaceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_namespace), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(namespace.Namespace())
        await client.get_namespace(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_namespace_flattened():
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_namespace), '__call__') as call:
        call.return_value = namespace.Namespace()
        client.get_namespace(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_namespace_flattened_error():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_namespace(registration_service.GetNamespaceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_namespace_flattened_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_namespace), '__call__') as call:
        call.return_value = namespace.Namespace()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(namespace.Namespace())
        response = await client.get_namespace(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_namespace_flattened_error_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_namespace(registration_service.GetNamespaceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [registration_service.UpdateNamespaceRequest, dict])
def test_update_namespace(request_type, transport: str='grpc'):
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_namespace), '__call__') as call:
        call.return_value = gcs_namespace.Namespace(name='name_value', uid='uid_value')
        response = client.update_namespace(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.UpdateNamespaceRequest()
    assert isinstance(response, gcs_namespace.Namespace)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'

def test_update_namespace_empty_call():
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_namespace), '__call__') as call:
        client.update_namespace()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.UpdateNamespaceRequest()

@pytest.mark.asyncio
async def test_update_namespace_async(transport: str='grpc_asyncio', request_type=registration_service.UpdateNamespaceRequest):
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_namespace), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_namespace.Namespace(name='name_value', uid='uid_value'))
        response = await client.update_namespace(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.UpdateNamespaceRequest()
    assert isinstance(response, gcs_namespace.Namespace)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'

@pytest.mark.asyncio
async def test_update_namespace_async_from_dict():
    await test_update_namespace_async(request_type=dict)

def test_update_namespace_field_headers():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.UpdateNamespaceRequest()
    request.namespace.name = 'name_value'
    with mock.patch.object(type(client.transport.update_namespace), '__call__') as call:
        call.return_value = gcs_namespace.Namespace()
        client.update_namespace(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'namespace.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_namespace_field_headers_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.UpdateNamespaceRequest()
    request.namespace.name = 'name_value'
    with mock.patch.object(type(client.transport.update_namespace), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_namespace.Namespace())
        await client.update_namespace(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'namespace.name=name_value') in kw['metadata']

def test_update_namespace_flattened():
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_namespace), '__call__') as call:
        call.return_value = gcs_namespace.Namespace()
        client.update_namespace(namespace=gcs_namespace.Namespace(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].namespace
        mock_val = gcs_namespace.Namespace(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_namespace_flattened_error():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_namespace(registration_service.UpdateNamespaceRequest(), namespace=gcs_namespace.Namespace(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_namespace_flattened_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_namespace), '__call__') as call:
        call.return_value = gcs_namespace.Namespace()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_namespace.Namespace())
        response = await client.update_namespace(namespace=gcs_namespace.Namespace(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].namespace
        mock_val = gcs_namespace.Namespace(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_namespace_flattened_error_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_namespace(registration_service.UpdateNamespaceRequest(), namespace=gcs_namespace.Namespace(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [registration_service.DeleteNamespaceRequest, dict])
def test_delete_namespace(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_namespace), '__call__') as call:
        call.return_value = None
        response = client.delete_namespace(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.DeleteNamespaceRequest()
    assert response is None

def test_delete_namespace_empty_call():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_namespace), '__call__') as call:
        client.delete_namespace()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.DeleteNamespaceRequest()

@pytest.mark.asyncio
async def test_delete_namespace_async(transport: str='grpc_asyncio', request_type=registration_service.DeleteNamespaceRequest):
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_namespace), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_namespace(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.DeleteNamespaceRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_namespace_async_from_dict():
    await test_delete_namespace_async(request_type=dict)

def test_delete_namespace_field_headers():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.DeleteNamespaceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_namespace), '__call__') as call:
        call.return_value = None
        client.delete_namespace(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_namespace_field_headers_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.DeleteNamespaceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_namespace), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_namespace(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_namespace_flattened():
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_namespace), '__call__') as call:
        call.return_value = None
        client.delete_namespace(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_namespace_flattened_error():
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_namespace(registration_service.DeleteNamespaceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_namespace_flattened_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_namespace), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_namespace(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_namespace_flattened_error_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_namespace(registration_service.DeleteNamespaceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [registration_service.CreateServiceRequest, dict])
def test_create_service(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        call.return_value = gcs_service.Service(name='name_value', uid='uid_value')
        response = client.create_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.CreateServiceRequest()
    assert isinstance(response, gcs_service.Service)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'

def test_create_service_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        client.create_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.CreateServiceRequest()

@pytest.mark.asyncio
async def test_create_service_async(transport: str='grpc_asyncio', request_type=registration_service.CreateServiceRequest):
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_service.Service(name='name_value', uid='uid_value'))
        response = await client.create_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.CreateServiceRequest()
    assert isinstance(response, gcs_service.Service)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'

@pytest.mark.asyncio
async def test_create_service_async_from_dict():
    await test_create_service_async(request_type=dict)

def test_create_service_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.CreateServiceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        call.return_value = gcs_service.Service()
        client.create_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_service_field_headers_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.CreateServiceRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_service.Service())
        await client.create_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_service_flattened():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        call.return_value = gcs_service.Service()
        client.create_service(parent='parent_value', service=gcs_service.Service(name='name_value'), service_id='service_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].service
        mock_val = gcs_service.Service(name='name_value')
        assert arg == mock_val
        arg = args[0].service_id
        mock_val = 'service_id_value'
        assert arg == mock_val

def test_create_service_flattened_error():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_service(registration_service.CreateServiceRequest(), parent='parent_value', service=gcs_service.Service(name='name_value'), service_id='service_id_value')

@pytest.mark.asyncio
async def test_create_service_flattened_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        call.return_value = gcs_service.Service()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_service.Service())
        response = await client.create_service(parent='parent_value', service=gcs_service.Service(name='name_value'), service_id='service_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].service
        mock_val = gcs_service.Service(name='name_value')
        assert arg == mock_val
        arg = args[0].service_id
        mock_val = 'service_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_service_flattened_error_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_service(registration_service.CreateServiceRequest(), parent='parent_value', service=gcs_service.Service(name='name_value'), service_id='service_id_value')

@pytest.mark.parametrize('request_type', [registration_service.ListServicesRequest, dict])
def test_list_services(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = registration_service.ListServicesResponse(next_page_token='next_page_token_value')
        response = client.list_services(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.ListServicesRequest()
    assert isinstance(response, pagers.ListServicesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_services_empty_call():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        client.list_services()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.ListServicesRequest()

@pytest.mark.asyncio
async def test_list_services_async(transport: str='grpc_asyncio', request_type=registration_service.ListServicesRequest):
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(registration_service.ListServicesResponse(next_page_token='next_page_token_value'))
        response = await client.list_services(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.ListServicesRequest()
    assert isinstance(response, pagers.ListServicesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_services_async_from_dict():
    await test_list_services_async(request_type=dict)

def test_list_services_field_headers():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.ListServicesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = registration_service.ListServicesResponse()
        client.list_services(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_services_field_headers_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.ListServicesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(registration_service.ListServicesResponse())
        await client.list_services(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_services_flattened():
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = registration_service.ListServicesResponse()
        client.list_services(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_services_flattened_error():
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_services(registration_service.ListServicesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_services_flattened_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = registration_service.ListServicesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(registration_service.ListServicesResponse())
        response = await client.list_services(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_services_flattened_error_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_services(registration_service.ListServicesRequest(), parent='parent_value')

def test_list_services_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.side_effect = (registration_service.ListServicesResponse(services=[service.Service(), service.Service(), service.Service()], next_page_token='abc'), registration_service.ListServicesResponse(services=[], next_page_token='def'), registration_service.ListServicesResponse(services=[service.Service()], next_page_token='ghi'), registration_service.ListServicesResponse(services=[service.Service(), service.Service()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_services(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service.Service) for i in results))

def test_list_services_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.side_effect = (registration_service.ListServicesResponse(services=[service.Service(), service.Service(), service.Service()], next_page_token='abc'), registration_service.ListServicesResponse(services=[], next_page_token='def'), registration_service.ListServicesResponse(services=[service.Service()], next_page_token='ghi'), registration_service.ListServicesResponse(services=[service.Service(), service.Service()]), RuntimeError)
        pages = list(client.list_services(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_services_async_pager():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_services), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (registration_service.ListServicesResponse(services=[service.Service(), service.Service(), service.Service()], next_page_token='abc'), registration_service.ListServicesResponse(services=[], next_page_token='def'), registration_service.ListServicesResponse(services=[service.Service()], next_page_token='ghi'), registration_service.ListServicesResponse(services=[service.Service(), service.Service()]), RuntimeError)
        async_pager = await client.list_services(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, service.Service) for i in responses))

@pytest.mark.asyncio
async def test_list_services_async_pages():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_services), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (registration_service.ListServicesResponse(services=[service.Service(), service.Service(), service.Service()], next_page_token='abc'), registration_service.ListServicesResponse(services=[], next_page_token='def'), registration_service.ListServicesResponse(services=[service.Service()], next_page_token='ghi'), registration_service.ListServicesResponse(services=[service.Service(), service.Service()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_services(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [registration_service.GetServiceRequest, dict])
def test_get_service(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = service.Service(name='name_value', uid='uid_value')
        response = client.get_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.GetServiceRequest()
    assert isinstance(response, service.Service)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'

def test_get_service_empty_call():
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        client.get_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.GetServiceRequest()

@pytest.mark.asyncio
async def test_get_service_async(transport: str='grpc_asyncio', request_type=registration_service.GetServiceRequest):
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.Service(name='name_value', uid='uid_value'))
        response = await client.get_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.GetServiceRequest()
    assert isinstance(response, service.Service)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'

@pytest.mark.asyncio
async def test_get_service_async_from_dict():
    await test_get_service_async(request_type=dict)

def test_get_service_field_headers():
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.GetServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = service.Service()
        client.get_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_service_field_headers_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.GetServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.Service())
        await client.get_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_service_flattened():
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = service.Service()
        client.get_service(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_service_flattened_error():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_service(registration_service.GetServiceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_service_flattened_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = service.Service()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service.Service())
        response = await client.get_service(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_service_flattened_error_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_service(registration_service.GetServiceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [registration_service.UpdateServiceRequest, dict])
def test_update_service(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_service), '__call__') as call:
        call.return_value = gcs_service.Service(name='name_value', uid='uid_value')
        response = client.update_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.UpdateServiceRequest()
    assert isinstance(response, gcs_service.Service)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'

def test_update_service_empty_call():
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_service), '__call__') as call:
        client.update_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.UpdateServiceRequest()

@pytest.mark.asyncio
async def test_update_service_async(transport: str='grpc_asyncio', request_type=registration_service.UpdateServiceRequest):
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_service.Service(name='name_value', uid='uid_value'))
        response = await client.update_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.UpdateServiceRequest()
    assert isinstance(response, gcs_service.Service)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'

@pytest.mark.asyncio
async def test_update_service_async_from_dict():
    await test_update_service_async(request_type=dict)

def test_update_service_field_headers():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.UpdateServiceRequest()
    request.service.name = 'name_value'
    with mock.patch.object(type(client.transport.update_service), '__call__') as call:
        call.return_value = gcs_service.Service()
        client.update_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_service_field_headers_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.UpdateServiceRequest()
    request.service.name = 'name_value'
    with mock.patch.object(type(client.transport.update_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_service.Service())
        await client.update_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service.name=name_value') in kw['metadata']

def test_update_service_flattened():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_service), '__call__') as call:
        call.return_value = gcs_service.Service()
        client.update_service(service=gcs_service.Service(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service
        mock_val = gcs_service.Service(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_service_flattened_error():
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_service(registration_service.UpdateServiceRequest(), service=gcs_service.Service(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_service_flattened_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_service), '__call__') as call:
        call.return_value = gcs_service.Service()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_service.Service())
        response = await client.update_service(service=gcs_service.Service(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service
        mock_val = gcs_service.Service(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_service_flattened_error_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_service(registration_service.UpdateServiceRequest(), service=gcs_service.Service(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [registration_service.DeleteServiceRequest, dict])
def test_delete_service(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        call.return_value = None
        response = client.delete_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.DeleteServiceRequest()
    assert response is None

def test_delete_service_empty_call():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        client.delete_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.DeleteServiceRequest()

@pytest.mark.asyncio
async def test_delete_service_async(transport: str='grpc_asyncio', request_type=registration_service.DeleteServiceRequest):
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.DeleteServiceRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_service_async_from_dict():
    await test_delete_service_async(request_type=dict)

def test_delete_service_field_headers():
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.DeleteServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        call.return_value = None
        client.delete_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_service_field_headers_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.DeleteServiceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_service_flattened():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        call.return_value = None
        client.delete_service(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_service_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_service(registration_service.DeleteServiceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_service_flattened_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_service(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_service_flattened_error_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_service(registration_service.DeleteServiceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [registration_service.CreateEndpointRequest, dict])
def test_create_endpoint(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_endpoint), '__call__') as call:
        call.return_value = gcs_endpoint.Endpoint(name='name_value', address='address_value', port=453, network='network_value', uid='uid_value')
        response = client.create_endpoint(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.CreateEndpointRequest()
    assert isinstance(response, gcs_endpoint.Endpoint)
    assert response.name == 'name_value'
    assert response.address == 'address_value'
    assert response.port == 453
    assert response.network == 'network_value'
    assert response.uid == 'uid_value'

def test_create_endpoint_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_endpoint), '__call__') as call:
        client.create_endpoint()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.CreateEndpointRequest()

@pytest.mark.asyncio
async def test_create_endpoint_async(transport: str='grpc_asyncio', request_type=registration_service.CreateEndpointRequest):
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_endpoint), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_endpoint.Endpoint(name='name_value', address='address_value', port=453, network='network_value', uid='uid_value'))
        response = await client.create_endpoint(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.CreateEndpointRequest()
    assert isinstance(response, gcs_endpoint.Endpoint)
    assert response.name == 'name_value'
    assert response.address == 'address_value'
    assert response.port == 453
    assert response.network == 'network_value'
    assert response.uid == 'uid_value'

@pytest.mark.asyncio
async def test_create_endpoint_async_from_dict():
    await test_create_endpoint_async(request_type=dict)

def test_create_endpoint_field_headers():
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.CreateEndpointRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_endpoint), '__call__') as call:
        call.return_value = gcs_endpoint.Endpoint()
        client.create_endpoint(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_endpoint_field_headers_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.CreateEndpointRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_endpoint), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_endpoint.Endpoint())
        await client.create_endpoint(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_endpoint_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_endpoint), '__call__') as call:
        call.return_value = gcs_endpoint.Endpoint()
        client.create_endpoint(parent='parent_value', endpoint=gcs_endpoint.Endpoint(name='name_value'), endpoint_id='endpoint_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].endpoint
        mock_val = gcs_endpoint.Endpoint(name='name_value')
        assert arg == mock_val
        arg = args[0].endpoint_id
        mock_val = 'endpoint_id_value'
        assert arg == mock_val

def test_create_endpoint_flattened_error():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_endpoint(registration_service.CreateEndpointRequest(), parent='parent_value', endpoint=gcs_endpoint.Endpoint(name='name_value'), endpoint_id='endpoint_id_value')

@pytest.mark.asyncio
async def test_create_endpoint_flattened_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_endpoint), '__call__') as call:
        call.return_value = gcs_endpoint.Endpoint()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_endpoint.Endpoint())
        response = await client.create_endpoint(parent='parent_value', endpoint=gcs_endpoint.Endpoint(name='name_value'), endpoint_id='endpoint_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].endpoint
        mock_val = gcs_endpoint.Endpoint(name='name_value')
        assert arg == mock_val
        arg = args[0].endpoint_id
        mock_val = 'endpoint_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_endpoint_flattened_error_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_endpoint(registration_service.CreateEndpointRequest(), parent='parent_value', endpoint=gcs_endpoint.Endpoint(name='name_value'), endpoint_id='endpoint_id_value')

@pytest.mark.parametrize('request_type', [registration_service.ListEndpointsRequest, dict])
def test_list_endpoints(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_endpoints), '__call__') as call:
        call.return_value = registration_service.ListEndpointsResponse(next_page_token='next_page_token_value')
        response = client.list_endpoints(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.ListEndpointsRequest()
    assert isinstance(response, pagers.ListEndpointsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_endpoints_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_endpoints), '__call__') as call:
        client.list_endpoints()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.ListEndpointsRequest()

@pytest.mark.asyncio
async def test_list_endpoints_async(transport: str='grpc_asyncio', request_type=registration_service.ListEndpointsRequest):
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_endpoints), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(registration_service.ListEndpointsResponse(next_page_token='next_page_token_value'))
        response = await client.list_endpoints(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.ListEndpointsRequest()
    assert isinstance(response, pagers.ListEndpointsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_endpoints_async_from_dict():
    await test_list_endpoints_async(request_type=dict)

def test_list_endpoints_field_headers():
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.ListEndpointsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_endpoints), '__call__') as call:
        call.return_value = registration_service.ListEndpointsResponse()
        client.list_endpoints(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_endpoints_field_headers_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.ListEndpointsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_endpoints), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(registration_service.ListEndpointsResponse())
        await client.list_endpoints(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_endpoints_flattened():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_endpoints), '__call__') as call:
        call.return_value = registration_service.ListEndpointsResponse()
        client.list_endpoints(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_endpoints_flattened_error():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_endpoints(registration_service.ListEndpointsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_endpoints_flattened_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_endpoints), '__call__') as call:
        call.return_value = registration_service.ListEndpointsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(registration_service.ListEndpointsResponse())
        response = await client.list_endpoints(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_endpoints_flattened_error_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_endpoints(registration_service.ListEndpointsRequest(), parent='parent_value')

def test_list_endpoints_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_endpoints), '__call__') as call:
        call.side_effect = (registration_service.ListEndpointsResponse(endpoints=[endpoint.Endpoint(), endpoint.Endpoint(), endpoint.Endpoint()], next_page_token='abc'), registration_service.ListEndpointsResponse(endpoints=[], next_page_token='def'), registration_service.ListEndpointsResponse(endpoints=[endpoint.Endpoint()], next_page_token='ghi'), registration_service.ListEndpointsResponse(endpoints=[endpoint.Endpoint(), endpoint.Endpoint()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_endpoints(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, endpoint.Endpoint) for i in results))

def test_list_endpoints_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_endpoints), '__call__') as call:
        call.side_effect = (registration_service.ListEndpointsResponse(endpoints=[endpoint.Endpoint(), endpoint.Endpoint(), endpoint.Endpoint()], next_page_token='abc'), registration_service.ListEndpointsResponse(endpoints=[], next_page_token='def'), registration_service.ListEndpointsResponse(endpoints=[endpoint.Endpoint()], next_page_token='ghi'), registration_service.ListEndpointsResponse(endpoints=[endpoint.Endpoint(), endpoint.Endpoint()]), RuntimeError)
        pages = list(client.list_endpoints(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_endpoints_async_pager():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_endpoints), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (registration_service.ListEndpointsResponse(endpoints=[endpoint.Endpoint(), endpoint.Endpoint(), endpoint.Endpoint()], next_page_token='abc'), registration_service.ListEndpointsResponse(endpoints=[], next_page_token='def'), registration_service.ListEndpointsResponse(endpoints=[endpoint.Endpoint()], next_page_token='ghi'), registration_service.ListEndpointsResponse(endpoints=[endpoint.Endpoint(), endpoint.Endpoint()]), RuntimeError)
        async_pager = await client.list_endpoints(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, endpoint.Endpoint) for i in responses))

@pytest.mark.asyncio
async def test_list_endpoints_async_pages():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_endpoints), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (registration_service.ListEndpointsResponse(endpoints=[endpoint.Endpoint(), endpoint.Endpoint(), endpoint.Endpoint()], next_page_token='abc'), registration_service.ListEndpointsResponse(endpoints=[], next_page_token='def'), registration_service.ListEndpointsResponse(endpoints=[endpoint.Endpoint()], next_page_token='ghi'), registration_service.ListEndpointsResponse(endpoints=[endpoint.Endpoint(), endpoint.Endpoint()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_endpoints(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [registration_service.GetEndpointRequest, dict])
def test_get_endpoint(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_endpoint), '__call__') as call:
        call.return_value = endpoint.Endpoint(name='name_value', address='address_value', port=453, network='network_value', uid='uid_value')
        response = client.get_endpoint(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.GetEndpointRequest()
    assert isinstance(response, endpoint.Endpoint)
    assert response.name == 'name_value'
    assert response.address == 'address_value'
    assert response.port == 453
    assert response.network == 'network_value'
    assert response.uid == 'uid_value'

def test_get_endpoint_empty_call():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_endpoint), '__call__') as call:
        client.get_endpoint()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.GetEndpointRequest()

@pytest.mark.asyncio
async def test_get_endpoint_async(transport: str='grpc_asyncio', request_type=registration_service.GetEndpointRequest):
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_endpoint), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(endpoint.Endpoint(name='name_value', address='address_value', port=453, network='network_value', uid='uid_value'))
        response = await client.get_endpoint(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.GetEndpointRequest()
    assert isinstance(response, endpoint.Endpoint)
    assert response.name == 'name_value'
    assert response.address == 'address_value'
    assert response.port == 453
    assert response.network == 'network_value'
    assert response.uid == 'uid_value'

@pytest.mark.asyncio
async def test_get_endpoint_async_from_dict():
    await test_get_endpoint_async(request_type=dict)

def test_get_endpoint_field_headers():
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.GetEndpointRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_endpoint), '__call__') as call:
        call.return_value = endpoint.Endpoint()
        client.get_endpoint(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_endpoint_field_headers_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.GetEndpointRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_endpoint), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(endpoint.Endpoint())
        await client.get_endpoint(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_endpoint_flattened():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_endpoint), '__call__') as call:
        call.return_value = endpoint.Endpoint()
        client.get_endpoint(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_endpoint_flattened_error():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_endpoint(registration_service.GetEndpointRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_endpoint_flattened_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_endpoint), '__call__') as call:
        call.return_value = endpoint.Endpoint()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(endpoint.Endpoint())
        response = await client.get_endpoint(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_endpoint_flattened_error_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_endpoint(registration_service.GetEndpointRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [registration_service.UpdateEndpointRequest, dict])
def test_update_endpoint(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_endpoint), '__call__') as call:
        call.return_value = gcs_endpoint.Endpoint(name='name_value', address='address_value', port=453, network='network_value', uid='uid_value')
        response = client.update_endpoint(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.UpdateEndpointRequest()
    assert isinstance(response, gcs_endpoint.Endpoint)
    assert response.name == 'name_value'
    assert response.address == 'address_value'
    assert response.port == 453
    assert response.network == 'network_value'
    assert response.uid == 'uid_value'

def test_update_endpoint_empty_call():
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_endpoint), '__call__') as call:
        client.update_endpoint()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.UpdateEndpointRequest()

@pytest.mark.asyncio
async def test_update_endpoint_async(transport: str='grpc_asyncio', request_type=registration_service.UpdateEndpointRequest):
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_endpoint), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_endpoint.Endpoint(name='name_value', address='address_value', port=453, network='network_value', uid='uid_value'))
        response = await client.update_endpoint(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.UpdateEndpointRequest()
    assert isinstance(response, gcs_endpoint.Endpoint)
    assert response.name == 'name_value'
    assert response.address == 'address_value'
    assert response.port == 453
    assert response.network == 'network_value'
    assert response.uid == 'uid_value'

@pytest.mark.asyncio
async def test_update_endpoint_async_from_dict():
    await test_update_endpoint_async(request_type=dict)

def test_update_endpoint_field_headers():
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.UpdateEndpointRequest()
    request.endpoint.name = 'name_value'
    with mock.patch.object(type(client.transport.update_endpoint), '__call__') as call:
        call.return_value = gcs_endpoint.Endpoint()
        client.update_endpoint(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'endpoint.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_endpoint_field_headers_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.UpdateEndpointRequest()
    request.endpoint.name = 'name_value'
    with mock.patch.object(type(client.transport.update_endpoint), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_endpoint.Endpoint())
        await client.update_endpoint(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'endpoint.name=name_value') in kw['metadata']

def test_update_endpoint_flattened():
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_endpoint), '__call__') as call:
        call.return_value = gcs_endpoint.Endpoint()
        client.update_endpoint(endpoint=gcs_endpoint.Endpoint(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].endpoint
        mock_val = gcs_endpoint.Endpoint(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_endpoint_flattened_error():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_endpoint(registration_service.UpdateEndpointRequest(), endpoint=gcs_endpoint.Endpoint(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_endpoint_flattened_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_endpoint), '__call__') as call:
        call.return_value = gcs_endpoint.Endpoint()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_endpoint.Endpoint())
        response = await client.update_endpoint(endpoint=gcs_endpoint.Endpoint(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].endpoint
        mock_val = gcs_endpoint.Endpoint(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_endpoint_flattened_error_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_endpoint(registration_service.UpdateEndpointRequest(), endpoint=gcs_endpoint.Endpoint(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [registration_service.DeleteEndpointRequest, dict])
def test_delete_endpoint(request_type, transport: str='grpc'):
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_endpoint), '__call__') as call:
        call.return_value = None
        response = client.delete_endpoint(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.DeleteEndpointRequest()
    assert response is None

def test_delete_endpoint_empty_call():
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_endpoint), '__call__') as call:
        client.delete_endpoint()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.DeleteEndpointRequest()

@pytest.mark.asyncio
async def test_delete_endpoint_async(transport: str='grpc_asyncio', request_type=registration_service.DeleteEndpointRequest):
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_endpoint), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_endpoint(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == registration_service.DeleteEndpointRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_endpoint_async_from_dict():
    await test_delete_endpoint_async(request_type=dict)

def test_delete_endpoint_field_headers():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.DeleteEndpointRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_endpoint), '__call__') as call:
        call.return_value = None
        client.delete_endpoint(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_endpoint_field_headers_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = registration_service.DeleteEndpointRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_endpoint), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_endpoint(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_endpoint_flattened():
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_endpoint), '__call__') as call:
        call.return_value = None
        client.delete_endpoint(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_endpoint_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_endpoint(registration_service.DeleteEndpointRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_endpoint_flattened_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_endpoint), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_endpoint(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_endpoint_flattened_error_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_endpoint(registration_service.DeleteEndpointRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy(request_type, transport: str='grpc'):
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        client.get_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()

@pytest.mark.asyncio
async def test_get_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.GetIamPolicyRequest):
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        client.set_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()

@pytest.mark.asyncio
async def test_set_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.SetIamPolicyRequest):
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774), 'update_mask': field_mask_pb2.FieldMask(paths=['paths_value'])})
        call.assert_called()

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        client.test_iam_permissions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()

@pytest.mark.asyncio
async def test_test_iam_permissions_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.parametrize('request_type', [registration_service.CreateNamespaceRequest, dict])
def test_create_namespace_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['namespace'] = {'name': 'name_value', 'labels': {}, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'uid': 'uid_value'}
    test_field = registration_service.CreateNamespaceRequest.meta.fields['namespace']

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
    for (field, value) in request_init['namespace'].items():
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
                for i in range(0, len(request_init['namespace'][field])):
                    del request_init['namespace'][field][i][subfield]
            else:
                del request_init['namespace'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_namespace.Namespace(name='name_value', uid='uid_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_namespace.Namespace.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_namespace(request)
    assert isinstance(response, gcs_namespace.Namespace)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'

def test_create_namespace_rest_required_fields(request_type=registration_service.CreateNamespaceRequest):
    if False:
        return 10
    transport_class = transports.RegistrationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['namespace_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'namespaceId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_namespace._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'namespaceId' in jsonified_request
    assert jsonified_request['namespaceId'] == request_init['namespace_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['namespaceId'] = 'namespace_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_namespace._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('namespace_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'namespaceId' in jsonified_request
    assert jsonified_request['namespaceId'] == 'namespace_id_value'
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcs_namespace.Namespace()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcs_namespace.Namespace.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_namespace(request)
            expected_params = [('namespaceId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_namespace_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_namespace._get_unset_required_fields({})
    assert set(unset_fields) == set(('namespaceId',)) & set(('parent', 'namespaceId', 'namespace'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_namespace_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegistrationServiceRestInterceptor())
    client = RegistrationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'post_create_namespace') as post, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'pre_create_namespace') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = registration_service.CreateNamespaceRequest.pb(registration_service.CreateNamespaceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcs_namespace.Namespace.to_json(gcs_namespace.Namespace())
        request = registration_service.CreateNamespaceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcs_namespace.Namespace()
        client.create_namespace(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_namespace_rest_bad_request(transport: str='rest', request_type=registration_service.CreateNamespaceRequest):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_namespace(request)

def test_create_namespace_rest_flattened():
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_namespace.Namespace()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', namespace=gcs_namespace.Namespace(name='name_value'), namespace_id='namespace_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_namespace.Namespace.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_namespace(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*/locations/*}/namespaces' % client.transport._host, args[1])

def test_create_namespace_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_namespace(registration_service.CreateNamespaceRequest(), parent='parent_value', namespace=gcs_namespace.Namespace(name='name_value'), namespace_id='namespace_id_value')

def test_create_namespace_rest_error():
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [registration_service.ListNamespacesRequest, dict])
def test_list_namespaces_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = registration_service.ListNamespacesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = registration_service.ListNamespacesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_namespaces(request)
    assert isinstance(response, pagers.ListNamespacesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_namespaces_rest_required_fields(request_type=registration_service.ListNamespacesRequest):
    if False:
        return 10
    transport_class = transports.RegistrationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_namespaces._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_namespaces._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = registration_service.ListNamespacesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = registration_service.ListNamespacesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_namespaces(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_namespaces_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_namespaces._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_namespaces_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegistrationServiceRestInterceptor())
    client = RegistrationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'post_list_namespaces') as post, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'pre_list_namespaces') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = registration_service.ListNamespacesRequest.pb(registration_service.ListNamespacesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = registration_service.ListNamespacesResponse.to_json(registration_service.ListNamespacesResponse())
        request = registration_service.ListNamespacesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = registration_service.ListNamespacesResponse()
        client.list_namespaces(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_namespaces_rest_bad_request(transport: str='rest', request_type=registration_service.ListNamespacesRequest):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_namespaces(request)

def test_list_namespaces_rest_flattened():
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = registration_service.ListNamespacesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = registration_service.ListNamespacesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_namespaces(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*/locations/*}/namespaces' % client.transport._host, args[1])

def test_list_namespaces_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_namespaces(registration_service.ListNamespacesRequest(), parent='parent_value')

def test_list_namespaces_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (registration_service.ListNamespacesResponse(namespaces=[namespace.Namespace(), namespace.Namespace(), namespace.Namespace()], next_page_token='abc'), registration_service.ListNamespacesResponse(namespaces=[], next_page_token='def'), registration_service.ListNamespacesResponse(namespaces=[namespace.Namespace()], next_page_token='ghi'), registration_service.ListNamespacesResponse(namespaces=[namespace.Namespace(), namespace.Namespace()]))
        response = response + response
        response = tuple((registration_service.ListNamespacesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_namespaces(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, namespace.Namespace) for i in results))
        pages = list(client.list_namespaces(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [registration_service.GetNamespaceRequest, dict])
def test_get_namespace_rest(request_type):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/namespaces/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = namespace.Namespace(name='name_value', uid='uid_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = namespace.Namespace.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_namespace(request)
    assert isinstance(response, namespace.Namespace)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'

def test_get_namespace_rest_required_fields(request_type=registration_service.GetNamespaceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegistrationServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_namespace._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_namespace._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = namespace.Namespace()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = namespace.Namespace.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_namespace(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_namespace_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_namespace._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_namespace_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegistrationServiceRestInterceptor())
    client = RegistrationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'post_get_namespace') as post, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'pre_get_namespace') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = registration_service.GetNamespaceRequest.pb(registration_service.GetNamespaceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = namespace.Namespace.to_json(namespace.Namespace())
        request = registration_service.GetNamespaceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = namespace.Namespace()
        client.get_namespace(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_namespace_rest_bad_request(transport: str='rest', request_type=registration_service.GetNamespaceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/namespaces/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_namespace(request)

def test_get_namespace_rest_flattened():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = namespace.Namespace()
        sample_request = {'name': 'projects/sample1/locations/sample2/namespaces/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = namespace.Namespace.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_namespace(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/namespaces/*}' % client.transport._host, args[1])

def test_get_namespace_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_namespace(registration_service.GetNamespaceRequest(), name='name_value')

def test_get_namespace_rest_error():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [registration_service.UpdateNamespaceRequest, dict])
def test_update_namespace_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'namespace': {'name': 'projects/sample1/locations/sample2/namespaces/sample3'}}
    request_init['namespace'] = {'name': 'projects/sample1/locations/sample2/namespaces/sample3', 'labels': {}, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'uid': 'uid_value'}
    test_field = registration_service.UpdateNamespaceRequest.meta.fields['namespace']

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
    for (field, value) in request_init['namespace'].items():
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
                for i in range(0, len(request_init['namespace'][field])):
                    del request_init['namespace'][field][i][subfield]
            else:
                del request_init['namespace'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_namespace.Namespace(name='name_value', uid='uid_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_namespace.Namespace.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_namespace(request)
    assert isinstance(response, gcs_namespace.Namespace)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'

def test_update_namespace_rest_required_fields(request_type=registration_service.UpdateNamespaceRequest):
    if False:
        print('Hello World!')
    transport_class = transports.RegistrationServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_namespace._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_namespace._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcs_namespace.Namespace()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcs_namespace.Namespace.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_namespace(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_namespace_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_namespace._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('namespace', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_namespace_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegistrationServiceRestInterceptor())
    client = RegistrationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'post_update_namespace') as post, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'pre_update_namespace') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = registration_service.UpdateNamespaceRequest.pb(registration_service.UpdateNamespaceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcs_namespace.Namespace.to_json(gcs_namespace.Namespace())
        request = registration_service.UpdateNamespaceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcs_namespace.Namespace()
        client.update_namespace(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_namespace_rest_bad_request(transport: str='rest', request_type=registration_service.UpdateNamespaceRequest):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'namespace': {'name': 'projects/sample1/locations/sample2/namespaces/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_namespace(request)

def test_update_namespace_rest_flattened():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_namespace.Namespace()
        sample_request = {'namespace': {'name': 'projects/sample1/locations/sample2/namespaces/sample3'}}
        mock_args = dict(namespace=gcs_namespace.Namespace(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_namespace.Namespace.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_namespace(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{namespace.name=projects/*/locations/*/namespaces/*}' % client.transport._host, args[1])

def test_update_namespace_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_namespace(registration_service.UpdateNamespaceRequest(), namespace=gcs_namespace.Namespace(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_namespace_rest_error():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [registration_service.DeleteNamespaceRequest, dict])
def test_delete_namespace_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/namespaces/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_namespace(request)
    assert response is None

def test_delete_namespace_rest_required_fields(request_type=registration_service.DeleteNamespaceRequest):
    if False:
        return 10
    transport_class = transports.RegistrationServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_namespace._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_namespace._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_namespace(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_namespace_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_namespace._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_namespace_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegistrationServiceRestInterceptor())
    client = RegistrationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'pre_delete_namespace') as pre:
        pre.assert_not_called()
        pb_message = registration_service.DeleteNamespaceRequest.pb(registration_service.DeleteNamespaceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = registration_service.DeleteNamespaceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_namespace(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_namespace_rest_bad_request(transport: str='rest', request_type=registration_service.DeleteNamespaceRequest):
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/namespaces/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_namespace(request)

def test_delete_namespace_rest_flattened():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/namespaces/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_namespace(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/namespaces/*}' % client.transport._host, args[1])

def test_delete_namespace_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_namespace(registration_service.DeleteNamespaceRequest(), name='name_value')

def test_delete_namespace_rest_error():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [registration_service.CreateServiceRequest, dict])
def test_create_service_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/namespaces/sample3'}
    request_init['service'] = {'name': 'name_value', 'metadata': {}, 'endpoints': [{'name': 'name_value', 'address': 'address_value', 'port': 453, 'metadata': {}, 'network': 'network_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'uid': 'uid_value'}], 'create_time': {}, 'update_time': {}, 'uid': 'uid_value'}
    test_field = registration_service.CreateServiceRequest.meta.fields['service']

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
    for (field, value) in request_init['service'].items():
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
                for i in range(0, len(request_init['service'][field])):
                    del request_init['service'][field][i][subfield]
            else:
                del request_init['service'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_service.Service(name='name_value', uid='uid_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_service.Service.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_service(request)
    assert isinstance(response, gcs_service.Service)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'

def test_create_service_rest_required_fields(request_type=registration_service.CreateServiceRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegistrationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['service_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'serviceId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'serviceId' in jsonified_request
    assert jsonified_request['serviceId'] == request_init['service_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['serviceId'] = 'service_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_service._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('service_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'serviceId' in jsonified_request
    assert jsonified_request['serviceId'] == 'service_id_value'
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcs_service.Service()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcs_service.Service.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_service(request)
            expected_params = [('serviceId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_service_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_service._get_unset_required_fields({})
    assert set(unset_fields) == set(('serviceId',)) & set(('parent', 'serviceId', 'service'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_service_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegistrationServiceRestInterceptor())
    client = RegistrationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'post_create_service') as post, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'pre_create_service') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = registration_service.CreateServiceRequest.pb(registration_service.CreateServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcs_service.Service.to_json(gcs_service.Service())
        request = registration_service.CreateServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcs_service.Service()
        client.create_service(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_service_rest_bad_request(transport: str='rest', request_type=registration_service.CreateServiceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/namespaces/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_service(request)

def test_create_service_rest_flattened():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_service.Service()
        sample_request = {'parent': 'projects/sample1/locations/sample2/namespaces/sample3'}
        mock_args = dict(parent='parent_value', service=gcs_service.Service(name='name_value'), service_id='service_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_service.Service.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_service(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*/locations/*/namespaces/*}/services' % client.transport._host, args[1])

def test_create_service_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_service(registration_service.CreateServiceRequest(), parent='parent_value', service=gcs_service.Service(name='name_value'), service_id='service_id_value')

def test_create_service_rest_error():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [registration_service.ListServicesRequest, dict])
def test_list_services_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/namespaces/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = registration_service.ListServicesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = registration_service.ListServicesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_services(request)
    assert isinstance(response, pagers.ListServicesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_services_rest_required_fields(request_type=registration_service.ListServicesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.RegistrationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_services._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_services._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = registration_service.ListServicesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = registration_service.ListServicesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_services(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_services_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_services._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_services_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegistrationServiceRestInterceptor())
    client = RegistrationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'post_list_services') as post, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'pre_list_services') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = registration_service.ListServicesRequest.pb(registration_service.ListServicesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = registration_service.ListServicesResponse.to_json(registration_service.ListServicesResponse())
        request = registration_service.ListServicesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = registration_service.ListServicesResponse()
        client.list_services(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_services_rest_bad_request(transport: str='rest', request_type=registration_service.ListServicesRequest):
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/namespaces/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_services(request)

def test_list_services_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = registration_service.ListServicesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/namespaces/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = registration_service.ListServicesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_services(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*/locations/*/namespaces/*}/services' % client.transport._host, args[1])

def test_list_services_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_services(registration_service.ListServicesRequest(), parent='parent_value')

def test_list_services_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (registration_service.ListServicesResponse(services=[service.Service(), service.Service(), service.Service()], next_page_token='abc'), registration_service.ListServicesResponse(services=[], next_page_token='def'), registration_service.ListServicesResponse(services=[service.Service()], next_page_token='ghi'), registration_service.ListServicesResponse(services=[service.Service(), service.Service()]))
        response = response + response
        response = tuple((registration_service.ListServicesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/namespaces/sample3'}
        pager = client.list_services(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service.Service) for i in results))
        pages = list(client.list_services(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [registration_service.GetServiceRequest, dict])
def test_get_service_rest(request_type):
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.Service(name='name_value', uid='uid_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = service.Service.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_service(request)
    assert isinstance(response, service.Service)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'

def test_get_service_rest_required_fields(request_type=registration_service.GetServiceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegistrationServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service.Service()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = service.Service.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_service(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_service_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_service._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_service_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegistrationServiceRestInterceptor())
    client = RegistrationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'post_get_service') as post, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'pre_get_service') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = registration_service.GetServiceRequest.pb(registration_service.GetServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = service.Service.to_json(service.Service())
        request = registration_service.GetServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service.Service()
        client.get_service(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_service_rest_bad_request(transport: str='rest', request_type=registration_service.GetServiceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_service(request)

def test_get_service_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service.Service()
        sample_request = {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = service.Service.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_service(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/namespaces/*/services/*}' % client.transport._host, args[1])

def test_get_service_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_service(registration_service.GetServiceRequest(), name='name_value')

def test_get_service_rest_error():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [registration_service.UpdateServiceRequest, dict])
def test_update_service_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'service': {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4'}}
    request_init['service'] = {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4', 'metadata': {}, 'endpoints': [{'name': 'name_value', 'address': 'address_value', 'port': 453, 'metadata': {}, 'network': 'network_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'uid': 'uid_value'}], 'create_time': {}, 'update_time': {}, 'uid': 'uid_value'}
    test_field = registration_service.UpdateServiceRequest.meta.fields['service']

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
    for (field, value) in request_init['service'].items():
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
                for i in range(0, len(request_init['service'][field])):
                    del request_init['service'][field][i][subfield]
            else:
                del request_init['service'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_service.Service(name='name_value', uid='uid_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_service.Service.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_service(request)
    assert isinstance(response, gcs_service.Service)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'

def test_update_service_rest_required_fields(request_type=registration_service.UpdateServiceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegistrationServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_service._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcs_service.Service()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcs_service.Service.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_service(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_service_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_service._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('service', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_service_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegistrationServiceRestInterceptor())
    client = RegistrationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'post_update_service') as post, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'pre_update_service') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = registration_service.UpdateServiceRequest.pb(registration_service.UpdateServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcs_service.Service.to_json(gcs_service.Service())
        request = registration_service.UpdateServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcs_service.Service()
        client.update_service(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_service_rest_bad_request(transport: str='rest', request_type=registration_service.UpdateServiceRequest):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'service': {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_service(request)

def test_update_service_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_service.Service()
        sample_request = {'service': {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4'}}
        mock_args = dict(service=gcs_service.Service(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_service.Service.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_service(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{service.name=projects/*/locations/*/namespaces/*/services/*}' % client.transport._host, args[1])

def test_update_service_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_service(registration_service.UpdateServiceRequest(), service=gcs_service.Service(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_service_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [registration_service.DeleteServiceRequest, dict])
def test_delete_service_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_service(request)
    assert response is None

def test_delete_service_rest_required_fields(request_type=registration_service.DeleteServiceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegistrationServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_service(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_service_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_service._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_service_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegistrationServiceRestInterceptor())
    client = RegistrationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'pre_delete_service') as pre:
        pre.assert_not_called()
        pb_message = registration_service.DeleteServiceRequest.pb(registration_service.DeleteServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = registration_service.DeleteServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_service(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_service_rest_bad_request(transport: str='rest', request_type=registration_service.DeleteServiceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_service(request)

def test_delete_service_rest_flattened():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_service(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/namespaces/*/services/*}' % client.transport._host, args[1])

def test_delete_service_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_service(registration_service.DeleteServiceRequest(), name='name_value')

def test_delete_service_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [registration_service.CreateEndpointRequest, dict])
def test_create_endpoint_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4'}
    request_init['endpoint'] = {'name': 'name_value', 'address': 'address_value', 'port': 453, 'metadata': {}, 'network': 'network_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'uid': 'uid_value'}
    test_field = registration_service.CreateEndpointRequest.meta.fields['endpoint']

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
    for (field, value) in request_init['endpoint'].items():
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
                for i in range(0, len(request_init['endpoint'][field])):
                    del request_init['endpoint'][field][i][subfield]
            else:
                del request_init['endpoint'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_endpoint.Endpoint(name='name_value', address='address_value', port=453, network='network_value', uid='uid_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_endpoint.Endpoint.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_endpoint(request)
    assert isinstance(response, gcs_endpoint.Endpoint)
    assert response.name == 'name_value'
    assert response.address == 'address_value'
    assert response.port == 453
    assert response.network == 'network_value'
    assert response.uid == 'uid_value'

def test_create_endpoint_rest_required_fields(request_type=registration_service.CreateEndpointRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegistrationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['endpoint_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'endpointId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_endpoint._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'endpointId' in jsonified_request
    assert jsonified_request['endpointId'] == request_init['endpoint_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['endpointId'] = 'endpoint_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_endpoint._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('endpoint_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'endpointId' in jsonified_request
    assert jsonified_request['endpointId'] == 'endpoint_id_value'
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcs_endpoint.Endpoint()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcs_endpoint.Endpoint.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_endpoint(request)
            expected_params = [('endpointId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_endpoint_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_endpoint._get_unset_required_fields({})
    assert set(unset_fields) == set(('endpointId',)) & set(('parent', 'endpointId', 'endpoint'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_endpoint_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegistrationServiceRestInterceptor())
    client = RegistrationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'post_create_endpoint') as post, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'pre_create_endpoint') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = registration_service.CreateEndpointRequest.pb(registration_service.CreateEndpointRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcs_endpoint.Endpoint.to_json(gcs_endpoint.Endpoint())
        request = registration_service.CreateEndpointRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcs_endpoint.Endpoint()
        client.create_endpoint(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_endpoint_rest_bad_request(transport: str='rest', request_type=registration_service.CreateEndpointRequest):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_endpoint(request)

def test_create_endpoint_rest_flattened():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_endpoint.Endpoint()
        sample_request = {'parent': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4'}
        mock_args = dict(parent='parent_value', endpoint=gcs_endpoint.Endpoint(name='name_value'), endpoint_id='endpoint_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_endpoint.Endpoint.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_endpoint(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*/locations/*/namespaces/*/services/*}/endpoints' % client.transport._host, args[1])

def test_create_endpoint_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_endpoint(registration_service.CreateEndpointRequest(), parent='parent_value', endpoint=gcs_endpoint.Endpoint(name='name_value'), endpoint_id='endpoint_id_value')

def test_create_endpoint_rest_error():
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [registration_service.ListEndpointsRequest, dict])
def test_list_endpoints_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = registration_service.ListEndpointsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = registration_service.ListEndpointsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_endpoints(request)
    assert isinstance(response, pagers.ListEndpointsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_endpoints_rest_required_fields(request_type=registration_service.ListEndpointsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegistrationServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_endpoints._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_endpoints._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = registration_service.ListEndpointsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = registration_service.ListEndpointsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_endpoints(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_endpoints_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_endpoints._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_endpoints_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegistrationServiceRestInterceptor())
    client = RegistrationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'post_list_endpoints') as post, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'pre_list_endpoints') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = registration_service.ListEndpointsRequest.pb(registration_service.ListEndpointsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = registration_service.ListEndpointsResponse.to_json(registration_service.ListEndpointsResponse())
        request = registration_service.ListEndpointsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = registration_service.ListEndpointsResponse()
        client.list_endpoints(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_endpoints_rest_bad_request(transport: str='rest', request_type=registration_service.ListEndpointsRequest):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_endpoints(request)

def test_list_endpoints_rest_flattened():
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = registration_service.ListEndpointsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = registration_service.ListEndpointsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_endpoints(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{parent=projects/*/locations/*/namespaces/*/services/*}/endpoints' % client.transport._host, args[1])

def test_list_endpoints_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_endpoints(registration_service.ListEndpointsRequest(), parent='parent_value')

def test_list_endpoints_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (registration_service.ListEndpointsResponse(endpoints=[endpoint.Endpoint(), endpoint.Endpoint(), endpoint.Endpoint()], next_page_token='abc'), registration_service.ListEndpointsResponse(endpoints=[], next_page_token='def'), registration_service.ListEndpointsResponse(endpoints=[endpoint.Endpoint()], next_page_token='ghi'), registration_service.ListEndpointsResponse(endpoints=[endpoint.Endpoint(), endpoint.Endpoint()]))
        response = response + response
        response = tuple((registration_service.ListEndpointsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4'}
        pager = client.list_endpoints(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, endpoint.Endpoint) for i in results))
        pages = list(client.list_endpoints(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [registration_service.GetEndpointRequest, dict])
def test_get_endpoint_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4/endpoints/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = endpoint.Endpoint(name='name_value', address='address_value', port=453, network='network_value', uid='uid_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = endpoint.Endpoint.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_endpoint(request)
    assert isinstance(response, endpoint.Endpoint)
    assert response.name == 'name_value'
    assert response.address == 'address_value'
    assert response.port == 453
    assert response.network == 'network_value'
    assert response.uid == 'uid_value'

def test_get_endpoint_rest_required_fields(request_type=registration_service.GetEndpointRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegistrationServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_endpoint._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_endpoint._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = endpoint.Endpoint()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = endpoint.Endpoint.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_endpoint(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_endpoint_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_endpoint._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_endpoint_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegistrationServiceRestInterceptor())
    client = RegistrationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'post_get_endpoint') as post, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'pre_get_endpoint') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = registration_service.GetEndpointRequest.pb(registration_service.GetEndpointRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = endpoint.Endpoint.to_json(endpoint.Endpoint())
        request = registration_service.GetEndpointRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = endpoint.Endpoint()
        client.get_endpoint(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_endpoint_rest_bad_request(transport: str='rest', request_type=registration_service.GetEndpointRequest):
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4/endpoints/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_endpoint(request)

def test_get_endpoint_rest_flattened():
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = endpoint.Endpoint()
        sample_request = {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4/endpoints/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = endpoint.Endpoint.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_endpoint(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/namespaces/*/services/*/endpoints/*}' % client.transport._host, args[1])

def test_get_endpoint_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_endpoint(registration_service.GetEndpointRequest(), name='name_value')

def test_get_endpoint_rest_error():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [registration_service.UpdateEndpointRequest, dict])
def test_update_endpoint_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'endpoint': {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4/endpoints/sample5'}}
    request_init['endpoint'] = {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4/endpoints/sample5', 'address': 'address_value', 'port': 453, 'metadata': {}, 'network': 'network_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'uid': 'uid_value'}
    test_field = registration_service.UpdateEndpointRequest.meta.fields['endpoint']

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
    for (field, value) in request_init['endpoint'].items():
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
                for i in range(0, len(request_init['endpoint'][field])):
                    del request_init['endpoint'][field][i][subfield]
            else:
                del request_init['endpoint'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_endpoint.Endpoint(name='name_value', address='address_value', port=453, network='network_value', uid='uid_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_endpoint.Endpoint.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_endpoint(request)
    assert isinstance(response, gcs_endpoint.Endpoint)
    assert response.name == 'name_value'
    assert response.address == 'address_value'
    assert response.port == 453
    assert response.network == 'network_value'
    assert response.uid == 'uid_value'

def test_update_endpoint_rest_required_fields(request_type=registration_service.UpdateEndpointRequest):
    if False:
        print('Hello World!')
    transport_class = transports.RegistrationServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_endpoint._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_endpoint._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcs_endpoint.Endpoint()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcs_endpoint.Endpoint.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_endpoint(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_endpoint_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_endpoint._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('endpoint', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_endpoint_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegistrationServiceRestInterceptor())
    client = RegistrationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'post_update_endpoint') as post, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'pre_update_endpoint') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = registration_service.UpdateEndpointRequest.pb(registration_service.UpdateEndpointRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcs_endpoint.Endpoint.to_json(gcs_endpoint.Endpoint())
        request = registration_service.UpdateEndpointRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcs_endpoint.Endpoint()
        client.update_endpoint(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_endpoint_rest_bad_request(transport: str='rest', request_type=registration_service.UpdateEndpointRequest):
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'endpoint': {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4/endpoints/sample5'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_endpoint(request)

def test_update_endpoint_rest_flattened():
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_endpoint.Endpoint()
        sample_request = {'endpoint': {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4/endpoints/sample5'}}
        mock_args = dict(endpoint=gcs_endpoint.Endpoint(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_endpoint.Endpoint.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_endpoint(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{endpoint.name=projects/*/locations/*/namespaces/*/services/*/endpoints/*}' % client.transport._host, args[1])

def test_update_endpoint_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_endpoint(registration_service.UpdateEndpointRequest(), endpoint=gcs_endpoint.Endpoint(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_endpoint_rest_error():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [registration_service.DeleteEndpointRequest, dict])
def test_delete_endpoint_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4/endpoints/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_endpoint(request)
    assert response is None

def test_delete_endpoint_rest_required_fields(request_type=registration_service.DeleteEndpointRequest):
    if False:
        return 10
    transport_class = transports.RegistrationServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_endpoint._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_endpoint._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_endpoint(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_endpoint_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_endpoint._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_endpoint_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegistrationServiceRestInterceptor())
    client = RegistrationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'pre_delete_endpoint') as pre:
        pre.assert_not_called()
        pb_message = registration_service.DeleteEndpointRequest.pb(registration_service.DeleteEndpointRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = registration_service.DeleteEndpointRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_endpoint(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_endpoint_rest_bad_request(transport: str='rest', request_type=registration_service.DeleteEndpointRequest):
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4/endpoints/sample5'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_endpoint(request)

def test_delete_endpoint_rest_flattened():
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/namespaces/sample3/services/sample4/endpoints/sample5'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_endpoint(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1beta1/{name=projects/*/locations/*/namespaces/*/services/*/endpoints/*}' % client.transport._host, args[1])

def test_delete_endpoint_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_endpoint(registration_service.DeleteEndpointRequest(), name='name_value')

def test_delete_endpoint_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/namespaces/sample3'}
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
    transport_class = transports.RegistrationServiceRestTransport
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
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_iam_policy_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegistrationServiceRestInterceptor())
    client = RegistrationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'post_get_iam_policy') as post, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'pre_get_iam_policy') as pre:
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
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1/locations/sample2/namespaces/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

def test_get_iam_policy_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/namespaces/sample3'}
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
        return 10
    transport_class = transports.RegistrationServiceRestTransport
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
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'policy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_iam_policy_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegistrationServiceRestInterceptor())
    client = RegistrationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'post_set_iam_policy') as post, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'pre_set_iam_policy') as pre:
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
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1/locations/sample2/namespaces/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

def test_set_iam_policy_rest_error():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/namespaces/sample3'}
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
        i = 10
        return i + 15
    transport_class = transports.RegistrationServiceRestTransport
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
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.test_iam_permissions._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'permissions'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_test_iam_permissions_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RegistrationServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegistrationServiceRestInterceptor())
    client = RegistrationServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'post_test_iam_permissions') as post, mock.patch.object(transports.RegistrationServiceRestInterceptor, 'pre_test_iam_permissions') as pre:
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
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'projects/sample1/locations/sample2/namespaces/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

def test_test_iam_permissions_rest_error():
    if False:
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.RegistrationServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.RegistrationServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = RegistrationServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.RegistrationServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = RegistrationServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = RegistrationServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.RegistrationServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = RegistrationServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.RegistrationServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = RegistrationServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegistrationServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.RegistrationServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.RegistrationServiceGrpcTransport, transports.RegistrationServiceGrpcAsyncIOTransport, transports.RegistrationServiceRestTransport])
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
        return 10
    transport = RegistrationServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.RegistrationServiceGrpcTransport)

def test_registration_service_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.RegistrationServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_registration_service_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.servicedirectory_v1beta1.services.registration_service.transports.RegistrationServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.RegistrationServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_namespace', 'list_namespaces', 'get_namespace', 'update_namespace', 'delete_namespace', 'create_service', 'list_services', 'get_service', 'update_service', 'delete_service', 'create_endpoint', 'list_endpoints', 'get_endpoint', 'update_endpoint', 'delete_endpoint', 'get_iam_policy', 'set_iam_policy', 'test_iam_permissions', 'get_location', 'list_locations')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_registration_service_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.servicedirectory_v1beta1.services.registration_service.transports.RegistrationServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.RegistrationServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_registration_service_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.servicedirectory_v1beta1.services.registration_service.transports.RegistrationServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.RegistrationServiceTransport()
        adc.assert_called_once()

def test_registration_service_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        RegistrationServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.RegistrationServiceGrpcTransport, transports.RegistrationServiceGrpcAsyncIOTransport])
def test_registration_service_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.RegistrationServiceGrpcTransport, transports.RegistrationServiceGrpcAsyncIOTransport, transports.RegistrationServiceRestTransport])
def test_registration_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.RegistrationServiceGrpcTransport, grpc_helpers), (transports.RegistrationServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_registration_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('servicedirectory.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='servicedirectory.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.RegistrationServiceGrpcTransport, transports.RegistrationServiceGrpcAsyncIOTransport])
def test_registration_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_registration_service_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.RegistrationServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_registration_service_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='servicedirectory.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('servicedirectory.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://servicedirectory.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_registration_service_host_with_port(transport_name):
    if False:
        return 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='servicedirectory.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('servicedirectory.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://servicedirectory.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_registration_service_client_transport_session_collision(transport_name):
    if False:
        while True:
            i = 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = RegistrationServiceClient(credentials=creds1, transport=transport_name)
    client2 = RegistrationServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_namespace._session
    session2 = client2.transport.create_namespace._session
    assert session1 != session2
    session1 = client1.transport.list_namespaces._session
    session2 = client2.transport.list_namespaces._session
    assert session1 != session2
    session1 = client1.transport.get_namespace._session
    session2 = client2.transport.get_namespace._session
    assert session1 != session2
    session1 = client1.transport.update_namespace._session
    session2 = client2.transport.update_namespace._session
    assert session1 != session2
    session1 = client1.transport.delete_namespace._session
    session2 = client2.transport.delete_namespace._session
    assert session1 != session2
    session1 = client1.transport.create_service._session
    session2 = client2.transport.create_service._session
    assert session1 != session2
    session1 = client1.transport.list_services._session
    session2 = client2.transport.list_services._session
    assert session1 != session2
    session1 = client1.transport.get_service._session
    session2 = client2.transport.get_service._session
    assert session1 != session2
    session1 = client1.transport.update_service._session
    session2 = client2.transport.update_service._session
    assert session1 != session2
    session1 = client1.transport.delete_service._session
    session2 = client2.transport.delete_service._session
    assert session1 != session2
    session1 = client1.transport.create_endpoint._session
    session2 = client2.transport.create_endpoint._session
    assert session1 != session2
    session1 = client1.transport.list_endpoints._session
    session2 = client2.transport.list_endpoints._session
    assert session1 != session2
    session1 = client1.transport.get_endpoint._session
    session2 = client2.transport.get_endpoint._session
    assert session1 != session2
    session1 = client1.transport.update_endpoint._session
    session2 = client2.transport.update_endpoint._session
    assert session1 != session2
    session1 = client1.transport.delete_endpoint._session
    session2 = client2.transport.delete_endpoint._session
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

def test_registration_service_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.RegistrationServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_registration_service_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.RegistrationServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.RegistrationServiceGrpcTransport, transports.RegistrationServiceGrpcAsyncIOTransport])
def test_registration_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.RegistrationServiceGrpcTransport, transports.RegistrationServiceGrpcAsyncIOTransport])
def test_registration_service_transport_channel_mtls_with_adc(transport_class):
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

def test_endpoint_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    namespace = 'whelk'
    service = 'octopus'
    endpoint = 'oyster'
    expected = 'projects/{project}/locations/{location}/namespaces/{namespace}/services/{service}/endpoints/{endpoint}'.format(project=project, location=location, namespace=namespace, service=service, endpoint=endpoint)
    actual = RegistrationServiceClient.endpoint_path(project, location, namespace, service, endpoint)
    assert expected == actual

def test_parse_endpoint_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'namespace': 'mussel', 'service': 'winkle', 'endpoint': 'nautilus'}
    path = RegistrationServiceClient.endpoint_path(**expected)
    actual = RegistrationServiceClient.parse_endpoint_path(path)
    assert expected == actual

def test_namespace_path():
    if False:
        while True:
            i = 10
    project = 'scallop'
    location = 'abalone'
    namespace = 'squid'
    expected = 'projects/{project}/locations/{location}/namespaces/{namespace}'.format(project=project, location=location, namespace=namespace)
    actual = RegistrationServiceClient.namespace_path(project, location, namespace)
    assert expected == actual

def test_parse_namespace_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'clam', 'location': 'whelk', 'namespace': 'octopus'}
    path = RegistrationServiceClient.namespace_path(**expected)
    actual = RegistrationServiceClient.parse_namespace_path(path)
    assert expected == actual

def test_network_path():
    if False:
        i = 10
        return i + 15
    project = 'oyster'
    network = 'nudibranch'
    expected = 'projects/{project}/locations/global/networks/{network}'.format(project=project, network=network)
    actual = RegistrationServiceClient.network_path(project, network)
    assert expected == actual

def test_parse_network_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'cuttlefish', 'network': 'mussel'}
    path = RegistrationServiceClient.network_path(**expected)
    actual = RegistrationServiceClient.parse_network_path(path)
    assert expected == actual

def test_service_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    location = 'nautilus'
    namespace = 'scallop'
    service = 'abalone'
    expected = 'projects/{project}/locations/{location}/namespaces/{namespace}/services/{service}'.format(project=project, location=location, namespace=namespace, service=service)
    actual = RegistrationServiceClient.service_path(project, location, namespace, service)
    assert expected == actual

def test_parse_service_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'squid', 'location': 'clam', 'namespace': 'whelk', 'service': 'octopus'}
    path = RegistrationServiceClient.service_path(**expected)
    actual = RegistrationServiceClient.parse_service_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = RegistrationServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'nudibranch'}
    path = RegistrationServiceClient.common_billing_account_path(**expected)
    actual = RegistrationServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = RegistrationServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'mussel'}
    path = RegistrationServiceClient.common_folder_path(**expected)
    actual = RegistrationServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = RegistrationServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'nautilus'}
    path = RegistrationServiceClient.common_organization_path(**expected)
    actual = RegistrationServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = RegistrationServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'abalone'}
    path = RegistrationServiceClient.common_project_path(**expected)
    actual = RegistrationServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = RegistrationServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = RegistrationServiceClient.common_location_path(**expected)
    actual = RegistrationServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.RegistrationServiceTransport, '_prep_wrapped_messages') as prep:
        client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.RegistrationServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = RegistrationServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_list_locations(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = RegistrationServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        for i in range(10):
            print('nop')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = RegistrationServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(RegistrationServiceClient, transports.RegistrationServiceGrpcTransport), (RegistrationServiceAsyncClient, transports.RegistrationServiceGrpcAsyncIOTransport)])
def test_api_key_credentials(client_class, transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth._default, 'get_api_key_credentials', create=True) as get_api_key_credentials:
        mock_cred = mock.Mock()
        get_api_key_credentials.return_value = mock_cred
        options = client_options.ClientOptions()
        options.api_key = 'api_key'
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options)
            patched.assert_called_once_with(credentials=mock_cred, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)