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
from google.protobuf import empty_pb2
from google.protobuf import json_format
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.vpcaccess_v1.services.vpc_access_service import VpcAccessServiceAsyncClient, VpcAccessServiceClient, pagers, transports
from google.cloud.vpcaccess_v1.types import vpc_access

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
        print('Hello World!')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert VpcAccessServiceClient._get_default_mtls_endpoint(None) is None
    assert VpcAccessServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert VpcAccessServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert VpcAccessServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert VpcAccessServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert VpcAccessServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(VpcAccessServiceClient, 'grpc'), (VpcAccessServiceAsyncClient, 'grpc_asyncio'), (VpcAccessServiceClient, 'rest')])
def test_vpc_access_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('vpcaccess.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://vpcaccess.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.VpcAccessServiceGrpcTransport, 'grpc'), (transports.VpcAccessServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.VpcAccessServiceRestTransport, 'rest')])
def test_vpc_access_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(VpcAccessServiceClient, 'grpc'), (VpcAccessServiceAsyncClient, 'grpc_asyncio'), (VpcAccessServiceClient, 'rest')])
def test_vpc_access_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('vpcaccess.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://vpcaccess.googleapis.com')

def test_vpc_access_service_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = VpcAccessServiceClient.get_transport_class()
    available_transports = [transports.VpcAccessServiceGrpcTransport, transports.VpcAccessServiceRestTransport]
    assert transport in available_transports
    transport = VpcAccessServiceClient.get_transport_class('grpc')
    assert transport == transports.VpcAccessServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(VpcAccessServiceClient, transports.VpcAccessServiceGrpcTransport, 'grpc'), (VpcAccessServiceAsyncClient, transports.VpcAccessServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (VpcAccessServiceClient, transports.VpcAccessServiceRestTransport, 'rest')])
@mock.patch.object(VpcAccessServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VpcAccessServiceClient))
@mock.patch.object(VpcAccessServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VpcAccessServiceAsyncClient))
def test_vpc_access_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(VpcAccessServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(VpcAccessServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(VpcAccessServiceClient, transports.VpcAccessServiceGrpcTransport, 'grpc', 'true'), (VpcAccessServiceAsyncClient, transports.VpcAccessServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (VpcAccessServiceClient, transports.VpcAccessServiceGrpcTransport, 'grpc', 'false'), (VpcAccessServiceAsyncClient, transports.VpcAccessServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (VpcAccessServiceClient, transports.VpcAccessServiceRestTransport, 'rest', 'true'), (VpcAccessServiceClient, transports.VpcAccessServiceRestTransport, 'rest', 'false')])
@mock.patch.object(VpcAccessServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VpcAccessServiceClient))
@mock.patch.object(VpcAccessServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VpcAccessServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_vpc_access_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [VpcAccessServiceClient, VpcAccessServiceAsyncClient])
@mock.patch.object(VpcAccessServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VpcAccessServiceClient))
@mock.patch.object(VpcAccessServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(VpcAccessServiceAsyncClient))
def test_vpc_access_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(VpcAccessServiceClient, transports.VpcAccessServiceGrpcTransport, 'grpc'), (VpcAccessServiceAsyncClient, transports.VpcAccessServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (VpcAccessServiceClient, transports.VpcAccessServiceRestTransport, 'rest')])
def test_vpc_access_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(VpcAccessServiceClient, transports.VpcAccessServiceGrpcTransport, 'grpc', grpc_helpers), (VpcAccessServiceAsyncClient, transports.VpcAccessServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (VpcAccessServiceClient, transports.VpcAccessServiceRestTransport, 'rest', None)])
def test_vpc_access_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_vpc_access_service_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.vpcaccess_v1.services.vpc_access_service.transports.VpcAccessServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = VpcAccessServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(VpcAccessServiceClient, transports.VpcAccessServiceGrpcTransport, 'grpc', grpc_helpers), (VpcAccessServiceAsyncClient, transports.VpcAccessServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_vpc_access_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('vpcaccess.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='vpcaccess.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [vpc_access.CreateConnectorRequest, dict])
def test_create_connector(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_connector), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_connector(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vpc_access.CreateConnectorRequest()
    assert isinstance(response, future.Future)

def test_create_connector_empty_call():
    if False:
        while True:
            i = 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_connector), '__call__') as call:
        client.create_connector()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vpc_access.CreateConnectorRequest()

@pytest.mark.asyncio
async def test_create_connector_async(transport: str='grpc_asyncio', request_type=vpc_access.CreateConnectorRequest):
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_connector), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_connector(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vpc_access.CreateConnectorRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_connector_async_from_dict():
    await test_create_connector_async(request_type=dict)

def test_create_connector_field_headers():
    if False:
        print('Hello World!')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = vpc_access.CreateConnectorRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_connector), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_connector(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_connector_field_headers_async():
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vpc_access.CreateConnectorRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_connector), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_connector(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_connector_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_connector), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_connector(parent='parent_value', connector_id='connector_id_value', connector=vpc_access.Connector(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].connector_id
        mock_val = 'connector_id_value'
        assert arg == mock_val
        arg = args[0].connector
        mock_val = vpc_access.Connector(name='name_value')
        assert arg == mock_val

def test_create_connector_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_connector(vpc_access.CreateConnectorRequest(), parent='parent_value', connector_id='connector_id_value', connector=vpc_access.Connector(name='name_value'))

@pytest.mark.asyncio
async def test_create_connector_flattened_async():
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_connector), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_connector(parent='parent_value', connector_id='connector_id_value', connector=vpc_access.Connector(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].connector_id
        mock_val = 'connector_id_value'
        assert arg == mock_val
        arg = args[0].connector
        mock_val = vpc_access.Connector(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_connector_flattened_error_async():
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_connector(vpc_access.CreateConnectorRequest(), parent='parent_value', connector_id='connector_id_value', connector=vpc_access.Connector(name='name_value'))

@pytest.mark.parametrize('request_type', [vpc_access.GetConnectorRequest, dict])
def test_get_connector(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_connector), '__call__') as call:
        call.return_value = vpc_access.Connector(name='name_value', network='network_value', ip_cidr_range='ip_cidr_range_value', state=vpc_access.Connector.State.READY, min_throughput=1533, max_throughput=1535, connected_projects=['connected_projects_value'], machine_type='machine_type_value', min_instances=1387, max_instances=1389)
        response = client.get_connector(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vpc_access.GetConnectorRequest()
    assert isinstance(response, vpc_access.Connector)
    assert response.name == 'name_value'
    assert response.network == 'network_value'
    assert response.ip_cidr_range == 'ip_cidr_range_value'
    assert response.state == vpc_access.Connector.State.READY
    assert response.min_throughput == 1533
    assert response.max_throughput == 1535
    assert response.connected_projects == ['connected_projects_value']
    assert response.machine_type == 'machine_type_value'
    assert response.min_instances == 1387
    assert response.max_instances == 1389

def test_get_connector_empty_call():
    if False:
        while True:
            i = 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_connector), '__call__') as call:
        client.get_connector()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vpc_access.GetConnectorRequest()

@pytest.mark.asyncio
async def test_get_connector_async(transport: str='grpc_asyncio', request_type=vpc_access.GetConnectorRequest):
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_connector), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vpc_access.Connector(name='name_value', network='network_value', ip_cidr_range='ip_cidr_range_value', state=vpc_access.Connector.State.READY, min_throughput=1533, max_throughput=1535, connected_projects=['connected_projects_value'], machine_type='machine_type_value', min_instances=1387, max_instances=1389))
        response = await client.get_connector(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vpc_access.GetConnectorRequest()
    assert isinstance(response, vpc_access.Connector)
    assert response.name == 'name_value'
    assert response.network == 'network_value'
    assert response.ip_cidr_range == 'ip_cidr_range_value'
    assert response.state == vpc_access.Connector.State.READY
    assert response.min_throughput == 1533
    assert response.max_throughput == 1535
    assert response.connected_projects == ['connected_projects_value']
    assert response.machine_type == 'machine_type_value'
    assert response.min_instances == 1387
    assert response.max_instances == 1389

@pytest.mark.asyncio
async def test_get_connector_async_from_dict():
    await test_get_connector_async(request_type=dict)

def test_get_connector_field_headers():
    if False:
        return 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = vpc_access.GetConnectorRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_connector), '__call__') as call:
        call.return_value = vpc_access.Connector()
        client.get_connector(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_connector_field_headers_async():
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vpc_access.GetConnectorRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_connector), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vpc_access.Connector())
        await client.get_connector(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_connector_flattened():
    if False:
        print('Hello World!')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_connector), '__call__') as call:
        call.return_value = vpc_access.Connector()
        client.get_connector(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_connector_flattened_error():
    if False:
        i = 10
        return i + 15
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_connector(vpc_access.GetConnectorRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_connector_flattened_async():
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_connector), '__call__') as call:
        call.return_value = vpc_access.Connector()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vpc_access.Connector())
        response = await client.get_connector(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_connector_flattened_error_async():
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_connector(vpc_access.GetConnectorRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vpc_access.ListConnectorsRequest, dict])
def test_list_connectors(request_type, transport: str='grpc'):
    if False:
        return 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_connectors), '__call__') as call:
        call.return_value = vpc_access.ListConnectorsResponse(next_page_token='next_page_token_value')
        response = client.list_connectors(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vpc_access.ListConnectorsRequest()
    assert isinstance(response, pagers.ListConnectorsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_connectors_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_connectors), '__call__') as call:
        client.list_connectors()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vpc_access.ListConnectorsRequest()

@pytest.mark.asyncio
async def test_list_connectors_async(transport: str='grpc_asyncio', request_type=vpc_access.ListConnectorsRequest):
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_connectors), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vpc_access.ListConnectorsResponse(next_page_token='next_page_token_value'))
        response = await client.list_connectors(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vpc_access.ListConnectorsRequest()
    assert isinstance(response, pagers.ListConnectorsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_connectors_async_from_dict():
    await test_list_connectors_async(request_type=dict)

def test_list_connectors_field_headers():
    if False:
        print('Hello World!')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = vpc_access.ListConnectorsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_connectors), '__call__') as call:
        call.return_value = vpc_access.ListConnectorsResponse()
        client.list_connectors(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_connectors_field_headers_async():
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vpc_access.ListConnectorsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_connectors), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vpc_access.ListConnectorsResponse())
        await client.list_connectors(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_connectors_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_connectors), '__call__') as call:
        call.return_value = vpc_access.ListConnectorsResponse()
        client.list_connectors(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_connectors_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_connectors(vpc_access.ListConnectorsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_connectors_flattened_async():
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_connectors), '__call__') as call:
        call.return_value = vpc_access.ListConnectorsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vpc_access.ListConnectorsResponse())
        response = await client.list_connectors(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_connectors_flattened_error_async():
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_connectors(vpc_access.ListConnectorsRequest(), parent='parent_value')

def test_list_connectors_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_connectors), '__call__') as call:
        call.side_effect = (vpc_access.ListConnectorsResponse(connectors=[vpc_access.Connector(), vpc_access.Connector(), vpc_access.Connector()], next_page_token='abc'), vpc_access.ListConnectorsResponse(connectors=[], next_page_token='def'), vpc_access.ListConnectorsResponse(connectors=[vpc_access.Connector()], next_page_token='ghi'), vpc_access.ListConnectorsResponse(connectors=[vpc_access.Connector(), vpc_access.Connector()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_connectors(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vpc_access.Connector) for i in results))

def test_list_connectors_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_connectors), '__call__') as call:
        call.side_effect = (vpc_access.ListConnectorsResponse(connectors=[vpc_access.Connector(), vpc_access.Connector(), vpc_access.Connector()], next_page_token='abc'), vpc_access.ListConnectorsResponse(connectors=[], next_page_token='def'), vpc_access.ListConnectorsResponse(connectors=[vpc_access.Connector()], next_page_token='ghi'), vpc_access.ListConnectorsResponse(connectors=[vpc_access.Connector(), vpc_access.Connector()]), RuntimeError)
        pages = list(client.list_connectors(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_connectors_async_pager():
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_connectors), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vpc_access.ListConnectorsResponse(connectors=[vpc_access.Connector(), vpc_access.Connector(), vpc_access.Connector()], next_page_token='abc'), vpc_access.ListConnectorsResponse(connectors=[], next_page_token='def'), vpc_access.ListConnectorsResponse(connectors=[vpc_access.Connector()], next_page_token='ghi'), vpc_access.ListConnectorsResponse(connectors=[vpc_access.Connector(), vpc_access.Connector()]), RuntimeError)
        async_pager = await client.list_connectors(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vpc_access.Connector) for i in responses))

@pytest.mark.asyncio
async def test_list_connectors_async_pages():
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_connectors), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vpc_access.ListConnectorsResponse(connectors=[vpc_access.Connector(), vpc_access.Connector(), vpc_access.Connector()], next_page_token='abc'), vpc_access.ListConnectorsResponse(connectors=[], next_page_token='def'), vpc_access.ListConnectorsResponse(connectors=[vpc_access.Connector()], next_page_token='ghi'), vpc_access.ListConnectorsResponse(connectors=[vpc_access.Connector(), vpc_access.Connector()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_connectors(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vpc_access.DeleteConnectorRequest, dict])
def test_delete_connector(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_connector), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_connector(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vpc_access.DeleteConnectorRequest()
    assert isinstance(response, future.Future)

def test_delete_connector_empty_call():
    if False:
        print('Hello World!')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_connector), '__call__') as call:
        client.delete_connector()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vpc_access.DeleteConnectorRequest()

@pytest.mark.asyncio
async def test_delete_connector_async(transport: str='grpc_asyncio', request_type=vpc_access.DeleteConnectorRequest):
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_connector), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_connector(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vpc_access.DeleteConnectorRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_connector_async_from_dict():
    await test_delete_connector_async(request_type=dict)

def test_delete_connector_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = vpc_access.DeleteConnectorRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_connector), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_connector(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_connector_field_headers_async():
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vpc_access.DeleteConnectorRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_connector), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_connector(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_connector_flattened():
    if False:
        return 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_connector), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_connector(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_connector_flattened_error():
    if False:
        print('Hello World!')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_connector(vpc_access.DeleteConnectorRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_connector_flattened_async():
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_connector), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_connector(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_connector_flattened_error_async():
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_connector(vpc_access.DeleteConnectorRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vpc_access.CreateConnectorRequest, dict])
def test_create_connector_rest(request_type):
    if False:
        print('Hello World!')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['connector'] = {'name': 'name_value', 'network': 'network_value', 'ip_cidr_range': 'ip_cidr_range_value', 'state': 1, 'min_throughput': 1533, 'max_throughput': 1535, 'connected_projects': ['connected_projects_value1', 'connected_projects_value2'], 'subnet': {'name': 'name_value', 'project_id': 'project_id_value'}, 'machine_type': 'machine_type_value', 'min_instances': 1387, 'max_instances': 1389}
    test_field = vpc_access.CreateConnectorRequest.meta.fields['connector']

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
    for (field, value) in request_init['connector'].items():
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
                for i in range(0, len(request_init['connector'][field])):
                    del request_init['connector'][field][i][subfield]
            else:
                del request_init['connector'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_connector(request)
    assert response.operation.name == 'operations/spam'

def test_create_connector_rest_required_fields(request_type=vpc_access.CreateConnectorRequest):
    if False:
        print('Hello World!')
    transport_class = transports.VpcAccessServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['connector_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'connectorId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_connector._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'connectorId' in jsonified_request
    assert jsonified_request['connectorId'] == request_init['connector_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['connectorId'] = 'connector_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_connector._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('connector_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'connectorId' in jsonified_request
    assert jsonified_request['connectorId'] == 'connector_id_value'
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_connector(request)
            expected_params = [('connectorId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_connector_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VpcAccessServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_connector._get_unset_required_fields({})
    assert set(unset_fields) == set(('connectorId',)) & set(('parent', 'connectorId', 'connector'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_connector_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VpcAccessServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VpcAccessServiceRestInterceptor())
    client = VpcAccessServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VpcAccessServiceRestInterceptor, 'post_create_connector') as post, mock.patch.object(transports.VpcAccessServiceRestInterceptor, 'pre_create_connector') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vpc_access.CreateConnectorRequest.pb(vpc_access.CreateConnectorRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vpc_access.CreateConnectorRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_connector(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_connector_rest_bad_request(transport: str='rest', request_type=vpc_access.CreateConnectorRequest):
    if False:
        while True:
            i = 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_connector(request)

def test_create_connector_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', connector_id='connector_id_value', connector=vpc_access.Connector(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_connector(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/connectors' % client.transport._host, args[1])

def test_create_connector_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_connector(vpc_access.CreateConnectorRequest(), parent='parent_value', connector_id='connector_id_value', connector=vpc_access.Connector(name='name_value'))

def test_create_connector_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vpc_access.GetConnectorRequest, dict])
def test_get_connector_rest(request_type):
    if False:
        print('Hello World!')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/connectors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vpc_access.Connector(name='name_value', network='network_value', ip_cidr_range='ip_cidr_range_value', state=vpc_access.Connector.State.READY, min_throughput=1533, max_throughput=1535, connected_projects=['connected_projects_value'], machine_type='machine_type_value', min_instances=1387, max_instances=1389)
        response_value = Response()
        response_value.status_code = 200
        return_value = vpc_access.Connector.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_connector(request)
    assert isinstance(response, vpc_access.Connector)
    assert response.name == 'name_value'
    assert response.network == 'network_value'
    assert response.ip_cidr_range == 'ip_cidr_range_value'
    assert response.state == vpc_access.Connector.State.READY
    assert response.min_throughput == 1533
    assert response.max_throughput == 1535
    assert response.connected_projects == ['connected_projects_value']
    assert response.machine_type == 'machine_type_value'
    assert response.min_instances == 1387
    assert response.max_instances == 1389

def test_get_connector_rest_required_fields(request_type=vpc_access.GetConnectorRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.VpcAccessServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_connector._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_connector._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vpc_access.Connector()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vpc_access.Connector.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_connector(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_connector_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.VpcAccessServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_connector._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_connector_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.VpcAccessServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VpcAccessServiceRestInterceptor())
    client = VpcAccessServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VpcAccessServiceRestInterceptor, 'post_get_connector') as post, mock.patch.object(transports.VpcAccessServiceRestInterceptor, 'pre_get_connector') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vpc_access.GetConnectorRequest.pb(vpc_access.GetConnectorRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vpc_access.Connector.to_json(vpc_access.Connector())
        request = vpc_access.GetConnectorRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vpc_access.Connector()
        client.get_connector(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_connector_rest_bad_request(transport: str='rest', request_type=vpc_access.GetConnectorRequest):
    if False:
        while True:
            i = 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/connectors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_connector(request)

def test_get_connector_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vpc_access.Connector()
        sample_request = {'name': 'projects/sample1/locations/sample2/connectors/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vpc_access.Connector.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_connector(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/connectors/*}' % client.transport._host, args[1])

def test_get_connector_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_connector(vpc_access.GetConnectorRequest(), name='name_value')

def test_get_connector_rest_error():
    if False:
        while True:
            i = 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vpc_access.ListConnectorsRequest, dict])
def test_list_connectors_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vpc_access.ListConnectorsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = vpc_access.ListConnectorsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_connectors(request)
    assert isinstance(response, pagers.ListConnectorsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_connectors_rest_required_fields(request_type=vpc_access.ListConnectorsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.VpcAccessServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_connectors._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_connectors._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vpc_access.ListConnectorsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vpc_access.ListConnectorsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_connectors(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_connectors_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VpcAccessServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_connectors._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_connectors_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.VpcAccessServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VpcAccessServiceRestInterceptor())
    client = VpcAccessServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.VpcAccessServiceRestInterceptor, 'post_list_connectors') as post, mock.patch.object(transports.VpcAccessServiceRestInterceptor, 'pre_list_connectors') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vpc_access.ListConnectorsRequest.pb(vpc_access.ListConnectorsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vpc_access.ListConnectorsResponse.to_json(vpc_access.ListConnectorsResponse())
        request = vpc_access.ListConnectorsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vpc_access.ListConnectorsResponse()
        client.list_connectors(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_connectors_rest_bad_request(transport: str='rest', request_type=vpc_access.ListConnectorsRequest):
    if False:
        return 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_connectors(request)

def test_list_connectors_rest_flattened():
    if False:
        return 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vpc_access.ListConnectorsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vpc_access.ListConnectorsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_connectors(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/connectors' % client.transport._host, args[1])

def test_list_connectors_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_connectors(vpc_access.ListConnectorsRequest(), parent='parent_value')

def test_list_connectors_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vpc_access.ListConnectorsResponse(connectors=[vpc_access.Connector(), vpc_access.Connector(), vpc_access.Connector()], next_page_token='abc'), vpc_access.ListConnectorsResponse(connectors=[], next_page_token='def'), vpc_access.ListConnectorsResponse(connectors=[vpc_access.Connector()], next_page_token='ghi'), vpc_access.ListConnectorsResponse(connectors=[vpc_access.Connector(), vpc_access.Connector()]))
        response = response + response
        response = tuple((vpc_access.ListConnectorsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_connectors(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vpc_access.Connector) for i in results))
        pages = list(client.list_connectors(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vpc_access.DeleteConnectorRequest, dict])
def test_delete_connector_rest(request_type):
    if False:
        return 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/connectors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_connector(request)
    assert response.operation.name == 'operations/spam'

def test_delete_connector_rest_required_fields(request_type=vpc_access.DeleteConnectorRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.VpcAccessServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_connector._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_connector._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_connector(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_connector_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VpcAccessServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_connector._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_connector_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.VpcAccessServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.VpcAccessServiceRestInterceptor())
    client = VpcAccessServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.VpcAccessServiceRestInterceptor, 'post_delete_connector') as post, mock.patch.object(transports.VpcAccessServiceRestInterceptor, 'pre_delete_connector') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vpc_access.DeleteConnectorRequest.pb(vpc_access.DeleteConnectorRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = vpc_access.DeleteConnectorRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_connector(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_connector_rest_bad_request(transport: str='rest', request_type=vpc_access.DeleteConnectorRequest):
    if False:
        print('Hello World!')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/connectors/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_connector(request)

def test_delete_connector_rest_flattened():
    if False:
        while True:
            i = 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/connectors/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_connector(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/connectors/*}' % client.transport._host, args[1])

def test_delete_connector_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_connector(vpc_access.DeleteConnectorRequest(), name='name_value')

def test_delete_connector_rest_error():
    if False:
        return 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.VpcAccessServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.VpcAccessServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = VpcAccessServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.VpcAccessServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = VpcAccessServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = VpcAccessServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.VpcAccessServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = VpcAccessServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.VpcAccessServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = VpcAccessServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.VpcAccessServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.VpcAccessServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.VpcAccessServiceGrpcTransport, transports.VpcAccessServiceGrpcAsyncIOTransport, transports.VpcAccessServiceRestTransport])
def test_transport_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        print('Hello World!')
    transport = VpcAccessServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.VpcAccessServiceGrpcTransport)

def test_vpc_access_service_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.VpcAccessServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_vpc_access_service_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.vpcaccess_v1.services.vpc_access_service.transports.VpcAccessServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.VpcAccessServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_connector', 'get_connector', 'list_connectors', 'delete_connector', 'list_locations', 'get_operation', 'list_operations')
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

def test_vpc_access_service_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.vpcaccess_v1.services.vpc_access_service.transports.VpcAccessServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.VpcAccessServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_vpc_access_service_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.vpcaccess_v1.services.vpc_access_service.transports.VpcAccessServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.VpcAccessServiceTransport()
        adc.assert_called_once()

def test_vpc_access_service_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        VpcAccessServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.VpcAccessServiceGrpcTransport, transports.VpcAccessServiceGrpcAsyncIOTransport])
def test_vpc_access_service_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.VpcAccessServiceGrpcTransport, transports.VpcAccessServiceGrpcAsyncIOTransport, transports.VpcAccessServiceRestTransport])
def test_vpc_access_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.VpcAccessServiceGrpcTransport, grpc_helpers), (transports.VpcAccessServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_vpc_access_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('vpcaccess.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='vpcaccess.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.VpcAccessServiceGrpcTransport, transports.VpcAccessServiceGrpcAsyncIOTransport])
def test_vpc_access_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_vpc_access_service_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.VpcAccessServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_vpc_access_service_rest_lro_client():
    if False:
        return 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_vpc_access_service_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='vpcaccess.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('vpcaccess.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://vpcaccess.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_vpc_access_service_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='vpcaccess.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('vpcaccess.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://vpcaccess.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_vpc_access_service_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = VpcAccessServiceClient(credentials=creds1, transport=transport_name)
    client2 = VpcAccessServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_connector._session
    session2 = client2.transport.create_connector._session
    assert session1 != session2
    session1 = client1.transport.get_connector._session
    session2 = client2.transport.get_connector._session
    assert session1 != session2
    session1 = client1.transport.list_connectors._session
    session2 = client2.transport.list_connectors._session
    assert session1 != session2
    session1 = client1.transport.delete_connector._session
    session2 = client2.transport.delete_connector._session
    assert session1 != session2

def test_vpc_access_service_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.VpcAccessServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_vpc_access_service_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.VpcAccessServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.VpcAccessServiceGrpcTransport, transports.VpcAccessServiceGrpcAsyncIOTransport])
def test_vpc_access_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.VpcAccessServiceGrpcTransport, transports.VpcAccessServiceGrpcAsyncIOTransport])
def test_vpc_access_service_transport_channel_mtls_with_adc(transport_class):
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

def test_vpc_access_service_grpc_lro_client():
    if False:
        return 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_vpc_access_service_grpc_lro_async_client():
    if False:
        print('Hello World!')
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_connector_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    connector = 'whelk'
    expected = 'projects/{project}/locations/{location}/connectors/{connector}'.format(project=project, location=location, connector=connector)
    actual = VpcAccessServiceClient.connector_path(project, location, connector)
    assert expected == actual

def test_parse_connector_path():
    if False:
        print('Hello World!')
    expected = {'project': 'octopus', 'location': 'oyster', 'connector': 'nudibranch'}
    path = VpcAccessServiceClient.connector_path(**expected)
    actual = VpcAccessServiceClient.parse_connector_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = VpcAccessServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'mussel'}
    path = VpcAccessServiceClient.common_billing_account_path(**expected)
    actual = VpcAccessServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = VpcAccessServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'nautilus'}
    path = VpcAccessServiceClient.common_folder_path(**expected)
    actual = VpcAccessServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = VpcAccessServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'abalone'}
    path = VpcAccessServiceClient.common_organization_path(**expected)
    actual = VpcAccessServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = VpcAccessServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'clam'}
    path = VpcAccessServiceClient.common_project_path(**expected)
    actual = VpcAccessServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = VpcAccessServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = VpcAccessServiceClient.common_location_path(**expected)
    actual = VpcAccessServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.VpcAccessServiceTransport, '_prep_wrapped_messages') as prep:
        client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.VpcAccessServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = VpcAccessServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_list_locations_rest_bad_request(transport: str='rest', request_type=locations_pb2.ListLocationsRequest):
    if False:
        return 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        return 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_get_operation(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = VpcAccessServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = VpcAccessServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(VpcAccessServiceClient, transports.VpcAccessServiceGrpcTransport), (VpcAccessServiceAsyncClient, transports.VpcAccessServiceGrpcAsyncIOTransport)])
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