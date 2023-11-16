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
from google.protobuf import empty_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from google.cloud.managedidentities_v1.services.managed_identities_service import ManagedIdentitiesServiceAsyncClient, ManagedIdentitiesServiceClient, pagers, transports
from google.cloud.managedidentities_v1.types import managed_identities_service, resource

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
        while True:
            i = 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert ManagedIdentitiesServiceClient._get_default_mtls_endpoint(None) is None
    assert ManagedIdentitiesServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert ManagedIdentitiesServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert ManagedIdentitiesServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert ManagedIdentitiesServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert ManagedIdentitiesServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(ManagedIdentitiesServiceClient, 'grpc'), (ManagedIdentitiesServiceAsyncClient, 'grpc_asyncio')])
def test_managed_identities_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == 'managedidentities.googleapis.com:443'

@pytest.mark.parametrize('transport_class,transport_name', [(transports.ManagedIdentitiesServiceGrpcTransport, 'grpc'), (transports.ManagedIdentitiesServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_managed_identities_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(ManagedIdentitiesServiceClient, 'grpc'), (ManagedIdentitiesServiceAsyncClient, 'grpc_asyncio')])
def test_managed_identities_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == 'managedidentities.googleapis.com:443'

def test_managed_identities_service_client_get_transport_class():
    if False:
        for i in range(10):
            print('nop')
    transport = ManagedIdentitiesServiceClient.get_transport_class()
    available_transports = [transports.ManagedIdentitiesServiceGrpcTransport]
    assert transport in available_transports
    transport = ManagedIdentitiesServiceClient.get_transport_class('grpc')
    assert transport == transports.ManagedIdentitiesServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ManagedIdentitiesServiceClient, transports.ManagedIdentitiesServiceGrpcTransport, 'grpc'), (ManagedIdentitiesServiceAsyncClient, transports.ManagedIdentitiesServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
@mock.patch.object(ManagedIdentitiesServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ManagedIdentitiesServiceClient))
@mock.patch.object(ManagedIdentitiesServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ManagedIdentitiesServiceAsyncClient))
def test_managed_identities_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(ManagedIdentitiesServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(ManagedIdentitiesServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(ManagedIdentitiesServiceClient, transports.ManagedIdentitiesServiceGrpcTransport, 'grpc', 'true'), (ManagedIdentitiesServiceAsyncClient, transports.ManagedIdentitiesServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (ManagedIdentitiesServiceClient, transports.ManagedIdentitiesServiceGrpcTransport, 'grpc', 'false'), (ManagedIdentitiesServiceAsyncClient, transports.ManagedIdentitiesServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false')])
@mock.patch.object(ManagedIdentitiesServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ManagedIdentitiesServiceClient))
@mock.patch.object(ManagedIdentitiesServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ManagedIdentitiesServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_managed_identities_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [ManagedIdentitiesServiceClient, ManagedIdentitiesServiceAsyncClient])
@mock.patch.object(ManagedIdentitiesServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ManagedIdentitiesServiceClient))
@mock.patch.object(ManagedIdentitiesServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ManagedIdentitiesServiceAsyncClient))
def test_managed_identities_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ManagedIdentitiesServiceClient, transports.ManagedIdentitiesServiceGrpcTransport, 'grpc'), (ManagedIdentitiesServiceAsyncClient, transports.ManagedIdentitiesServiceGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_managed_identities_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ManagedIdentitiesServiceClient, transports.ManagedIdentitiesServiceGrpcTransport, 'grpc', grpc_helpers), (ManagedIdentitiesServiceAsyncClient, transports.ManagedIdentitiesServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_managed_identities_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_managed_identities_service_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.managedidentities_v1.services.managed_identities_service.transports.ManagedIdentitiesServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = ManagedIdentitiesServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ManagedIdentitiesServiceClient, transports.ManagedIdentitiesServiceGrpcTransport, 'grpc', grpc_helpers), (ManagedIdentitiesServiceAsyncClient, transports.ManagedIdentitiesServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_managed_identities_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('managedidentities.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='managedidentities.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [managed_identities_service.CreateMicrosoftAdDomainRequest, dict])
def test_create_microsoft_ad_domain(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_microsoft_ad_domain), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_microsoft_ad_domain(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.CreateMicrosoftAdDomainRequest()
    assert isinstance(response, future.Future)

def test_create_microsoft_ad_domain_empty_call():
    if False:
        print('Hello World!')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_microsoft_ad_domain), '__call__') as call:
        client.create_microsoft_ad_domain()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.CreateMicrosoftAdDomainRequest()

@pytest.mark.asyncio
async def test_create_microsoft_ad_domain_async(transport: str='grpc_asyncio', request_type=managed_identities_service.CreateMicrosoftAdDomainRequest):
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_microsoft_ad_domain), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_microsoft_ad_domain(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.CreateMicrosoftAdDomainRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_microsoft_ad_domain_async_from_dict():
    await test_create_microsoft_ad_domain_async(request_type=dict)

def test_create_microsoft_ad_domain_field_headers():
    if False:
        i = 10
        return i + 15
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.CreateMicrosoftAdDomainRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_microsoft_ad_domain), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_microsoft_ad_domain(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_microsoft_ad_domain_field_headers_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.CreateMicrosoftAdDomainRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_microsoft_ad_domain), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_microsoft_ad_domain(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_microsoft_ad_domain_flattened():
    if False:
        while True:
            i = 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_microsoft_ad_domain), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_microsoft_ad_domain(parent='parent_value', domain_name='domain_name_value', domain=resource.Domain(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].domain_name
        mock_val = 'domain_name_value'
        assert arg == mock_val
        arg = args[0].domain
        mock_val = resource.Domain(name='name_value')
        assert arg == mock_val

def test_create_microsoft_ad_domain_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_microsoft_ad_domain(managed_identities_service.CreateMicrosoftAdDomainRequest(), parent='parent_value', domain_name='domain_name_value', domain=resource.Domain(name='name_value'))

@pytest.mark.asyncio
async def test_create_microsoft_ad_domain_flattened_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_microsoft_ad_domain), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_microsoft_ad_domain(parent='parent_value', domain_name='domain_name_value', domain=resource.Domain(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].domain_name
        mock_val = 'domain_name_value'
        assert arg == mock_val
        arg = args[0].domain
        mock_val = resource.Domain(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_microsoft_ad_domain_flattened_error_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_microsoft_ad_domain(managed_identities_service.CreateMicrosoftAdDomainRequest(), parent='parent_value', domain_name='domain_name_value', domain=resource.Domain(name='name_value'))

@pytest.mark.parametrize('request_type', [managed_identities_service.ResetAdminPasswordRequest, dict])
def test_reset_admin_password(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reset_admin_password), '__call__') as call:
        call.return_value = managed_identities_service.ResetAdminPasswordResponse(password='password_value')
        response = client.reset_admin_password(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.ResetAdminPasswordRequest()
    assert isinstance(response, managed_identities_service.ResetAdminPasswordResponse)
    assert response.password == 'password_value'

def test_reset_admin_password_empty_call():
    if False:
        print('Hello World!')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.reset_admin_password), '__call__') as call:
        client.reset_admin_password()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.ResetAdminPasswordRequest()

@pytest.mark.asyncio
async def test_reset_admin_password_async(transport: str='grpc_asyncio', request_type=managed_identities_service.ResetAdminPasswordRequest):
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reset_admin_password), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(managed_identities_service.ResetAdminPasswordResponse(password='password_value'))
        response = await client.reset_admin_password(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.ResetAdminPasswordRequest()
    assert isinstance(response, managed_identities_service.ResetAdminPasswordResponse)
    assert response.password == 'password_value'

@pytest.mark.asyncio
async def test_reset_admin_password_async_from_dict():
    await test_reset_admin_password_async(request_type=dict)

def test_reset_admin_password_field_headers():
    if False:
        i = 10
        return i + 15
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.ResetAdminPasswordRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.reset_admin_password), '__call__') as call:
        call.return_value = managed_identities_service.ResetAdminPasswordResponse()
        client.reset_admin_password(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_reset_admin_password_field_headers_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.ResetAdminPasswordRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.reset_admin_password), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(managed_identities_service.ResetAdminPasswordResponse())
        await client.reset_admin_password(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_reset_admin_password_flattened():
    if False:
        while True:
            i = 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.reset_admin_password), '__call__') as call:
        call.return_value = managed_identities_service.ResetAdminPasswordResponse()
        client.reset_admin_password(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_reset_admin_password_flattened_error():
    if False:
        while True:
            i = 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.reset_admin_password(managed_identities_service.ResetAdminPasswordRequest(), name='name_value')

@pytest.mark.asyncio
async def test_reset_admin_password_flattened_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.reset_admin_password), '__call__') as call:
        call.return_value = managed_identities_service.ResetAdminPasswordResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(managed_identities_service.ResetAdminPasswordResponse())
        response = await client.reset_admin_password(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_reset_admin_password_flattened_error_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.reset_admin_password(managed_identities_service.ResetAdminPasswordRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [managed_identities_service.ListDomainsRequest, dict])
def test_list_domains(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_domains), '__call__') as call:
        call.return_value = managed_identities_service.ListDomainsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_domains(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.ListDomainsRequest()
    assert isinstance(response, pagers.ListDomainsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_domains_empty_call():
    if False:
        print('Hello World!')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_domains), '__call__') as call:
        client.list_domains()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.ListDomainsRequest()

@pytest.mark.asyncio
async def test_list_domains_async(transport: str='grpc_asyncio', request_type=managed_identities_service.ListDomainsRequest):
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_domains), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(managed_identities_service.ListDomainsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_domains(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.ListDomainsRequest()
    assert isinstance(response, pagers.ListDomainsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_domains_async_from_dict():
    await test_list_domains_async(request_type=dict)

def test_list_domains_field_headers():
    if False:
        while True:
            i = 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.ListDomainsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_domains), '__call__') as call:
        call.return_value = managed_identities_service.ListDomainsResponse()
        client.list_domains(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_domains_field_headers_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.ListDomainsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_domains), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(managed_identities_service.ListDomainsResponse())
        await client.list_domains(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_domains_flattened():
    if False:
        return 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_domains), '__call__') as call:
        call.return_value = managed_identities_service.ListDomainsResponse()
        client.list_domains(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_domains_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_domains(managed_identities_service.ListDomainsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_domains_flattened_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_domains), '__call__') as call:
        call.return_value = managed_identities_service.ListDomainsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(managed_identities_service.ListDomainsResponse())
        response = await client.list_domains(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_domains_flattened_error_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_domains(managed_identities_service.ListDomainsRequest(), parent='parent_value')

def test_list_domains_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_domains), '__call__') as call:
        call.side_effect = (managed_identities_service.ListDomainsResponse(domains=[resource.Domain(), resource.Domain(), resource.Domain()], next_page_token='abc'), managed_identities_service.ListDomainsResponse(domains=[], next_page_token='def'), managed_identities_service.ListDomainsResponse(domains=[resource.Domain()], next_page_token='ghi'), managed_identities_service.ListDomainsResponse(domains=[resource.Domain(), resource.Domain()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_domains(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resource.Domain) for i in results))

def test_list_domains_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_domains), '__call__') as call:
        call.side_effect = (managed_identities_service.ListDomainsResponse(domains=[resource.Domain(), resource.Domain(), resource.Domain()], next_page_token='abc'), managed_identities_service.ListDomainsResponse(domains=[], next_page_token='def'), managed_identities_service.ListDomainsResponse(domains=[resource.Domain()], next_page_token='ghi'), managed_identities_service.ListDomainsResponse(domains=[resource.Domain(), resource.Domain()]), RuntimeError)
        pages = list(client.list_domains(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_domains_async_pager():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_domains), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (managed_identities_service.ListDomainsResponse(domains=[resource.Domain(), resource.Domain(), resource.Domain()], next_page_token='abc'), managed_identities_service.ListDomainsResponse(domains=[], next_page_token='def'), managed_identities_service.ListDomainsResponse(domains=[resource.Domain()], next_page_token='ghi'), managed_identities_service.ListDomainsResponse(domains=[resource.Domain(), resource.Domain()]), RuntimeError)
        async_pager = await client.list_domains(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resource.Domain) for i in responses))

@pytest.mark.asyncio
async def test_list_domains_async_pages():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_domains), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (managed_identities_service.ListDomainsResponse(domains=[resource.Domain(), resource.Domain(), resource.Domain()], next_page_token='abc'), managed_identities_service.ListDomainsResponse(domains=[], next_page_token='def'), managed_identities_service.ListDomainsResponse(domains=[resource.Domain()], next_page_token='ghi'), managed_identities_service.ListDomainsResponse(domains=[resource.Domain(), resource.Domain()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_domains(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [managed_identities_service.GetDomainRequest, dict])
def test_get_domain(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_domain), '__call__') as call:
        call.return_value = resource.Domain(name='name_value', authorized_networks=['authorized_networks_value'], reserved_ip_range='reserved_ip_range_value', locations=['locations_value'], admin='admin_value', fqdn='fqdn_value', state=resource.Domain.State.CREATING, status_message='status_message_value')
        response = client.get_domain(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.GetDomainRequest()
    assert isinstance(response, resource.Domain)
    assert response.name == 'name_value'
    assert response.authorized_networks == ['authorized_networks_value']
    assert response.reserved_ip_range == 'reserved_ip_range_value'
    assert response.locations == ['locations_value']
    assert response.admin == 'admin_value'
    assert response.fqdn == 'fqdn_value'
    assert response.state == resource.Domain.State.CREATING
    assert response.status_message == 'status_message_value'

def test_get_domain_empty_call():
    if False:
        print('Hello World!')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_domain), '__call__') as call:
        client.get_domain()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.GetDomainRequest()

@pytest.mark.asyncio
async def test_get_domain_async(transport: str='grpc_asyncio', request_type=managed_identities_service.GetDomainRequest):
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_domain), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resource.Domain(name='name_value', authorized_networks=['authorized_networks_value'], reserved_ip_range='reserved_ip_range_value', locations=['locations_value'], admin='admin_value', fqdn='fqdn_value', state=resource.Domain.State.CREATING, status_message='status_message_value'))
        response = await client.get_domain(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.GetDomainRequest()
    assert isinstance(response, resource.Domain)
    assert response.name == 'name_value'
    assert response.authorized_networks == ['authorized_networks_value']
    assert response.reserved_ip_range == 'reserved_ip_range_value'
    assert response.locations == ['locations_value']
    assert response.admin == 'admin_value'
    assert response.fqdn == 'fqdn_value'
    assert response.state == resource.Domain.State.CREATING
    assert response.status_message == 'status_message_value'

@pytest.mark.asyncio
async def test_get_domain_async_from_dict():
    await test_get_domain_async(request_type=dict)

def test_get_domain_field_headers():
    if False:
        i = 10
        return i + 15
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.GetDomainRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_domain), '__call__') as call:
        call.return_value = resource.Domain()
        client.get_domain(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_domain_field_headers_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.GetDomainRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_domain), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resource.Domain())
        await client.get_domain(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_domain_flattened():
    if False:
        return 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_domain), '__call__') as call:
        call.return_value = resource.Domain()
        client.get_domain(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_domain_flattened_error():
    if False:
        print('Hello World!')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_domain(managed_identities_service.GetDomainRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_domain_flattened_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_domain), '__call__') as call:
        call.return_value = resource.Domain()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resource.Domain())
        response = await client.get_domain(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_domain_flattened_error_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_domain(managed_identities_service.GetDomainRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [managed_identities_service.UpdateDomainRequest, dict])
def test_update_domain(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_domain), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_domain(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.UpdateDomainRequest()
    assert isinstance(response, future.Future)

def test_update_domain_empty_call():
    if False:
        while True:
            i = 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_domain), '__call__') as call:
        client.update_domain()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.UpdateDomainRequest()

@pytest.mark.asyncio
async def test_update_domain_async(transport: str='grpc_asyncio', request_type=managed_identities_service.UpdateDomainRequest):
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_domain), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_domain(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.UpdateDomainRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_domain_async_from_dict():
    await test_update_domain_async(request_type=dict)

def test_update_domain_field_headers():
    if False:
        while True:
            i = 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.UpdateDomainRequest()
    request.domain.name = 'name_value'
    with mock.patch.object(type(client.transport.update_domain), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_domain(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'domain.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_domain_field_headers_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.UpdateDomainRequest()
    request.domain.name = 'name_value'
    with mock.patch.object(type(client.transport.update_domain), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_domain(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'domain.name=name_value') in kw['metadata']

def test_update_domain_flattened():
    if False:
        return 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_domain), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_domain(domain=resource.Domain(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].domain
        mock_val = resource.Domain(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_domain_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_domain(managed_identities_service.UpdateDomainRequest(), domain=resource.Domain(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_domain_flattened_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_domain), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_domain(domain=resource.Domain(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].domain
        mock_val = resource.Domain(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_domain_flattened_error_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_domain(managed_identities_service.UpdateDomainRequest(), domain=resource.Domain(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [managed_identities_service.DeleteDomainRequest, dict])
def test_delete_domain(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_domain), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_domain(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.DeleteDomainRequest()
    assert isinstance(response, future.Future)

def test_delete_domain_empty_call():
    if False:
        while True:
            i = 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_domain), '__call__') as call:
        client.delete_domain()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.DeleteDomainRequest()

@pytest.mark.asyncio
async def test_delete_domain_async(transport: str='grpc_asyncio', request_type=managed_identities_service.DeleteDomainRequest):
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_domain), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_domain(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.DeleteDomainRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_domain_async_from_dict():
    await test_delete_domain_async(request_type=dict)

def test_delete_domain_field_headers():
    if False:
        i = 10
        return i + 15
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.DeleteDomainRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_domain), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_domain(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_domain_field_headers_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.DeleteDomainRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_domain), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_domain(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_domain_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_domain), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_domain(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_domain_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_domain(managed_identities_service.DeleteDomainRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_domain_flattened_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_domain), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_domain(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_domain_flattened_error_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_domain(managed_identities_service.DeleteDomainRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [managed_identities_service.AttachTrustRequest, dict])
def test_attach_trust(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.attach_trust), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.attach_trust(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.AttachTrustRequest()
    assert isinstance(response, future.Future)

def test_attach_trust_empty_call():
    if False:
        return 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.attach_trust), '__call__') as call:
        client.attach_trust()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.AttachTrustRequest()

@pytest.mark.asyncio
async def test_attach_trust_async(transport: str='grpc_asyncio', request_type=managed_identities_service.AttachTrustRequest):
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.attach_trust), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.attach_trust(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.AttachTrustRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_attach_trust_async_from_dict():
    await test_attach_trust_async(request_type=dict)

def test_attach_trust_field_headers():
    if False:
        return 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.AttachTrustRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.attach_trust), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.attach_trust(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_attach_trust_field_headers_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.AttachTrustRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.attach_trust), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.attach_trust(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_attach_trust_flattened():
    if False:
        return 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.attach_trust), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.attach_trust(name='name_value', trust=resource.Trust(target_domain_name='target_domain_name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].trust
        mock_val = resource.Trust(target_domain_name='target_domain_name_value')
        assert arg == mock_val

def test_attach_trust_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.attach_trust(managed_identities_service.AttachTrustRequest(), name='name_value', trust=resource.Trust(target_domain_name='target_domain_name_value'))

@pytest.mark.asyncio
async def test_attach_trust_flattened_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.attach_trust), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.attach_trust(name='name_value', trust=resource.Trust(target_domain_name='target_domain_name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].trust
        mock_val = resource.Trust(target_domain_name='target_domain_name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_attach_trust_flattened_error_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.attach_trust(managed_identities_service.AttachTrustRequest(), name='name_value', trust=resource.Trust(target_domain_name='target_domain_name_value'))

@pytest.mark.parametrize('request_type', [managed_identities_service.ReconfigureTrustRequest, dict])
def test_reconfigure_trust(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reconfigure_trust), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.reconfigure_trust(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.ReconfigureTrustRequest()
    assert isinstance(response, future.Future)

def test_reconfigure_trust_empty_call():
    if False:
        return 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.reconfigure_trust), '__call__') as call:
        client.reconfigure_trust()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.ReconfigureTrustRequest()

@pytest.mark.asyncio
async def test_reconfigure_trust_async(transport: str='grpc_asyncio', request_type=managed_identities_service.ReconfigureTrustRequest):
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.reconfigure_trust), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.reconfigure_trust(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.ReconfigureTrustRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_reconfigure_trust_async_from_dict():
    await test_reconfigure_trust_async(request_type=dict)

def test_reconfigure_trust_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.ReconfigureTrustRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.reconfigure_trust), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.reconfigure_trust(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_reconfigure_trust_field_headers_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.ReconfigureTrustRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.reconfigure_trust), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.reconfigure_trust(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_reconfigure_trust_flattened():
    if False:
        print('Hello World!')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.reconfigure_trust), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.reconfigure_trust(name='name_value', target_domain_name='target_domain_name_value', target_dns_ip_addresses=['target_dns_ip_addresses_value'])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].target_domain_name
        mock_val = 'target_domain_name_value'
        assert arg == mock_val
        arg = args[0].target_dns_ip_addresses
        mock_val = ['target_dns_ip_addresses_value']
        assert arg == mock_val

def test_reconfigure_trust_flattened_error():
    if False:
        print('Hello World!')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.reconfigure_trust(managed_identities_service.ReconfigureTrustRequest(), name='name_value', target_domain_name='target_domain_name_value', target_dns_ip_addresses=['target_dns_ip_addresses_value'])

@pytest.mark.asyncio
async def test_reconfigure_trust_flattened_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.reconfigure_trust), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.reconfigure_trust(name='name_value', target_domain_name='target_domain_name_value', target_dns_ip_addresses=['target_dns_ip_addresses_value'])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].target_domain_name
        mock_val = 'target_domain_name_value'
        assert arg == mock_val
        arg = args[0].target_dns_ip_addresses
        mock_val = ['target_dns_ip_addresses_value']
        assert arg == mock_val

@pytest.mark.asyncio
async def test_reconfigure_trust_flattened_error_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.reconfigure_trust(managed_identities_service.ReconfigureTrustRequest(), name='name_value', target_domain_name='target_domain_name_value', target_dns_ip_addresses=['target_dns_ip_addresses_value'])

@pytest.mark.parametrize('request_type', [managed_identities_service.DetachTrustRequest, dict])
def test_detach_trust(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.detach_trust), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.detach_trust(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.DetachTrustRequest()
    assert isinstance(response, future.Future)

def test_detach_trust_empty_call():
    if False:
        print('Hello World!')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.detach_trust), '__call__') as call:
        client.detach_trust()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.DetachTrustRequest()

@pytest.mark.asyncio
async def test_detach_trust_async(transport: str='grpc_asyncio', request_type=managed_identities_service.DetachTrustRequest):
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.detach_trust), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.detach_trust(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.DetachTrustRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_detach_trust_async_from_dict():
    await test_detach_trust_async(request_type=dict)

def test_detach_trust_field_headers():
    if False:
        while True:
            i = 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.DetachTrustRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.detach_trust), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.detach_trust(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_detach_trust_field_headers_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.DetachTrustRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.detach_trust), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.detach_trust(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_detach_trust_flattened():
    if False:
        while True:
            i = 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.detach_trust), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.detach_trust(name='name_value', trust=resource.Trust(target_domain_name='target_domain_name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].trust
        mock_val = resource.Trust(target_domain_name='target_domain_name_value')
        assert arg == mock_val

def test_detach_trust_flattened_error():
    if False:
        print('Hello World!')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.detach_trust(managed_identities_service.DetachTrustRequest(), name='name_value', trust=resource.Trust(target_domain_name='target_domain_name_value'))

@pytest.mark.asyncio
async def test_detach_trust_flattened_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.detach_trust), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.detach_trust(name='name_value', trust=resource.Trust(target_domain_name='target_domain_name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].trust
        mock_val = resource.Trust(target_domain_name='target_domain_name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_detach_trust_flattened_error_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.detach_trust(managed_identities_service.DetachTrustRequest(), name='name_value', trust=resource.Trust(target_domain_name='target_domain_name_value'))

@pytest.mark.parametrize('request_type', [managed_identities_service.ValidateTrustRequest, dict])
def test_validate_trust(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.validate_trust), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.validate_trust(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.ValidateTrustRequest()
    assert isinstance(response, future.Future)

def test_validate_trust_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.validate_trust), '__call__') as call:
        client.validate_trust()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.ValidateTrustRequest()

@pytest.mark.asyncio
async def test_validate_trust_async(transport: str='grpc_asyncio', request_type=managed_identities_service.ValidateTrustRequest):
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.validate_trust), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.validate_trust(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == managed_identities_service.ValidateTrustRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_validate_trust_async_from_dict():
    await test_validate_trust_async(request_type=dict)

def test_validate_trust_field_headers():
    if False:
        print('Hello World!')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.ValidateTrustRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.validate_trust), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.validate_trust(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_validate_trust_field_headers_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = managed_identities_service.ValidateTrustRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.validate_trust), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.validate_trust(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_validate_trust_flattened():
    if False:
        while True:
            i = 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.validate_trust), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.validate_trust(name='name_value', trust=resource.Trust(target_domain_name='target_domain_name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].trust
        mock_val = resource.Trust(target_domain_name='target_domain_name_value')
        assert arg == mock_val

def test_validate_trust_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.validate_trust(managed_identities_service.ValidateTrustRequest(), name='name_value', trust=resource.Trust(target_domain_name='target_domain_name_value'))

@pytest.mark.asyncio
async def test_validate_trust_flattened_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.validate_trust), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.validate_trust(name='name_value', trust=resource.Trust(target_domain_name='target_domain_name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].trust
        mock_val = resource.Trust(target_domain_name='target_domain_name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_validate_trust_flattened_error_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.validate_trust(managed_identities_service.ValidateTrustRequest(), name='name_value', trust=resource.Trust(target_domain_name='target_domain_name_value'))

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.ManagedIdentitiesServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.ManagedIdentitiesServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ManagedIdentitiesServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.ManagedIdentitiesServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ManagedIdentitiesServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ManagedIdentitiesServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.ManagedIdentitiesServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ManagedIdentitiesServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.ManagedIdentitiesServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = ManagedIdentitiesServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.ManagedIdentitiesServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.ManagedIdentitiesServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.ManagedIdentitiesServiceGrpcTransport, transports.ManagedIdentitiesServiceGrpcAsyncIOTransport])
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
        return 10
    transport = ManagedIdentitiesServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.ManagedIdentitiesServiceGrpcTransport)

def test_managed_identities_service_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.ManagedIdentitiesServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_managed_identities_service_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.managedidentities_v1.services.managed_identities_service.transports.ManagedIdentitiesServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.ManagedIdentitiesServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_microsoft_ad_domain', 'reset_admin_password', 'list_domains', 'get_domain', 'update_domain', 'delete_domain', 'attach_trust', 'reconfigure_trust', 'detach_trust', 'validate_trust')
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

def test_managed_identities_service_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.managedidentities_v1.services.managed_identities_service.transports.ManagedIdentitiesServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ManagedIdentitiesServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_managed_identities_service_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.managedidentities_v1.services.managed_identities_service.transports.ManagedIdentitiesServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ManagedIdentitiesServiceTransport()
        adc.assert_called_once()

def test_managed_identities_service_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        ManagedIdentitiesServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.ManagedIdentitiesServiceGrpcTransport, transports.ManagedIdentitiesServiceGrpcAsyncIOTransport])
def test_managed_identities_service_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.ManagedIdentitiesServiceGrpcTransport, transports.ManagedIdentitiesServiceGrpcAsyncIOTransport])
def test_managed_identities_service_transport_auth_gdch_credentials(transport_class):
    if False:
        for i in range(10):
            print('nop')
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.ManagedIdentitiesServiceGrpcTransport, grpc_helpers), (transports.ManagedIdentitiesServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_managed_identities_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('managedidentities.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='managedidentities.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.ManagedIdentitiesServiceGrpcTransport, transports.ManagedIdentitiesServiceGrpcAsyncIOTransport])
def test_managed_identities_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_managed_identities_service_host_no_port(transport_name):
    if False:
        return 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='managedidentities.googleapis.com'), transport=transport_name)
    assert client.transport._host == 'managedidentities.googleapis.com:443'

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_managed_identities_service_host_with_port(transport_name):
    if False:
        return 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='managedidentities.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == 'managedidentities.googleapis.com:8000'

def test_managed_identities_service_grpc_transport_channel():
    if False:
        print('Hello World!')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ManagedIdentitiesServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_managed_identities_service_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ManagedIdentitiesServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.ManagedIdentitiesServiceGrpcTransport, transports.ManagedIdentitiesServiceGrpcAsyncIOTransport])
def test_managed_identities_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.ManagedIdentitiesServiceGrpcTransport, transports.ManagedIdentitiesServiceGrpcAsyncIOTransport])
def test_managed_identities_service_transport_channel_mtls_with_adc(transport_class):
    if False:
        while True:
            i = 10
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

def test_managed_identities_service_grpc_lro_client():
    if False:
        while True:
            i = 10
    client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_managed_identities_service_grpc_lro_async_client():
    if False:
        for i in range(10):
            print('nop')
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_domain_path():
    if False:
        print('Hello World!')
    project = 'squid'
    location = 'clam'
    domain = 'whelk'
    expected = 'projects/{project}/locations/{location}/domains/{domain}'.format(project=project, location=location, domain=domain)
    actual = ManagedIdentitiesServiceClient.domain_path(project, location, domain)
    assert expected == actual

def test_parse_domain_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'octopus', 'location': 'oyster', 'domain': 'nudibranch'}
    path = ManagedIdentitiesServiceClient.domain_path(**expected)
    actual = ManagedIdentitiesServiceClient.parse_domain_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = ManagedIdentitiesServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'mussel'}
    path = ManagedIdentitiesServiceClient.common_billing_account_path(**expected)
    actual = ManagedIdentitiesServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = ManagedIdentitiesServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'nautilus'}
    path = ManagedIdentitiesServiceClient.common_folder_path(**expected)
    actual = ManagedIdentitiesServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = ManagedIdentitiesServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'abalone'}
    path = ManagedIdentitiesServiceClient.common_organization_path(**expected)
    actual = ManagedIdentitiesServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = ManagedIdentitiesServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'clam'}
    path = ManagedIdentitiesServiceClient.common_project_path(**expected)
    actual = ManagedIdentitiesServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = ManagedIdentitiesServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = ManagedIdentitiesServiceClient.common_location_path(**expected)
    actual = ManagedIdentitiesServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.ManagedIdentitiesServiceTransport, '_prep_wrapped_messages') as prep:
        client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.ManagedIdentitiesServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = ManagedIdentitiesServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = ManagedIdentitiesServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        for i in range(10):
            print('nop')
    transports = {'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['grpc']
    for transport in transports:
        client = ManagedIdentitiesServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(ManagedIdentitiesServiceClient, transports.ManagedIdentitiesServiceGrpcTransport), (ManagedIdentitiesServiceAsyncClient, transports.ManagedIdentitiesServiceGrpcAsyncIOTransport)])
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