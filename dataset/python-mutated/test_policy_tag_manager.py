import os
try:
    from unittest import mock
    from unittest.mock import AsyncMock
except ImportError:
    import mock
import math
from google.api_core import gapic_v1, grpc_helpers, grpc_helpers_async, path_template
from google.api_core import client_options
from google.api_core import exceptions as core_exceptions
import google.auth
from google.auth import credentials as ga_credentials
from google.auth.exceptions import MutualTLSChannelError
from google.iam.v1 import iam_policy_pb2
from google.iam.v1 import options_pb2
from google.iam.v1 import policy_pb2
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2
from google.type import expr_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from google.cloud.datacatalog_v1beta1.services.policy_tag_manager import PolicyTagManagerAsyncClient, PolicyTagManagerClient, pagers, transports
from google.cloud.datacatalog_v1beta1.types import common, policytagmanager, timestamps

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
        for i in range(10):
            print('nop')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert PolicyTagManagerClient._get_default_mtls_endpoint(None) is None
    assert PolicyTagManagerClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert PolicyTagManagerClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert PolicyTagManagerClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert PolicyTagManagerClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert PolicyTagManagerClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(PolicyTagManagerClient, 'grpc'), (PolicyTagManagerAsyncClient, 'grpc_asyncio')])
def test_policy_tag_manager_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == 'datacatalog.googleapis.com:443'

@pytest.mark.parametrize('transport_class,transport_name', [(transports.PolicyTagManagerGrpcTransport, 'grpc'), (transports.PolicyTagManagerGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_policy_tag_manager_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(PolicyTagManagerClient, 'grpc'), (PolicyTagManagerAsyncClient, 'grpc_asyncio')])
def test_policy_tag_manager_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == 'datacatalog.googleapis.com:443'

def test_policy_tag_manager_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = PolicyTagManagerClient.get_transport_class()
    available_transports = [transports.PolicyTagManagerGrpcTransport]
    assert transport in available_transports
    transport = PolicyTagManagerClient.get_transport_class('grpc')
    assert transport == transports.PolicyTagManagerGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(PolicyTagManagerClient, transports.PolicyTagManagerGrpcTransport, 'grpc'), (PolicyTagManagerAsyncClient, transports.PolicyTagManagerGrpcAsyncIOTransport, 'grpc_asyncio')])
@mock.patch.object(PolicyTagManagerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(PolicyTagManagerClient))
@mock.patch.object(PolicyTagManagerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(PolicyTagManagerAsyncClient))
def test_policy_tag_manager_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(PolicyTagManagerClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(PolicyTagManagerClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(PolicyTagManagerClient, transports.PolicyTagManagerGrpcTransport, 'grpc', 'true'), (PolicyTagManagerAsyncClient, transports.PolicyTagManagerGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (PolicyTagManagerClient, transports.PolicyTagManagerGrpcTransport, 'grpc', 'false'), (PolicyTagManagerAsyncClient, transports.PolicyTagManagerGrpcAsyncIOTransport, 'grpc_asyncio', 'false')])
@mock.patch.object(PolicyTagManagerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(PolicyTagManagerClient))
@mock.patch.object(PolicyTagManagerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(PolicyTagManagerAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_policy_tag_manager_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [PolicyTagManagerClient, PolicyTagManagerAsyncClient])
@mock.patch.object(PolicyTagManagerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(PolicyTagManagerClient))
@mock.patch.object(PolicyTagManagerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(PolicyTagManagerAsyncClient))
def test_policy_tag_manager_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(PolicyTagManagerClient, transports.PolicyTagManagerGrpcTransport, 'grpc'), (PolicyTagManagerAsyncClient, transports.PolicyTagManagerGrpcAsyncIOTransport, 'grpc_asyncio')])
def test_policy_tag_manager_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(PolicyTagManagerClient, transports.PolicyTagManagerGrpcTransport, 'grpc', grpc_helpers), (PolicyTagManagerAsyncClient, transports.PolicyTagManagerGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_policy_tag_manager_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_policy_tag_manager_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.datacatalog_v1beta1.services.policy_tag_manager.transports.PolicyTagManagerGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = PolicyTagManagerClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(PolicyTagManagerClient, transports.PolicyTagManagerGrpcTransport, 'grpc', grpc_helpers), (PolicyTagManagerAsyncClient, transports.PolicyTagManagerGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_policy_tag_manager_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('datacatalog.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='datacatalog.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [policytagmanager.CreateTaxonomyRequest, dict])
def test_create_taxonomy(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_taxonomy), '__call__') as call:
        call.return_value = policytagmanager.Taxonomy(name='name_value', display_name='display_name_value', description='description_value', policy_tag_count=1715, activated_policy_types=[policytagmanager.Taxonomy.PolicyType.FINE_GRAINED_ACCESS_CONTROL])
        response = client.create_taxonomy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.CreateTaxonomyRequest()
    assert isinstance(response, policytagmanager.Taxonomy)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.policy_tag_count == 1715
    assert response.activated_policy_types == [policytagmanager.Taxonomy.PolicyType.FINE_GRAINED_ACCESS_CONTROL]

def test_create_taxonomy_empty_call():
    if False:
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_taxonomy), '__call__') as call:
        client.create_taxonomy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.CreateTaxonomyRequest()

@pytest.mark.asyncio
async def test_create_taxonomy_async(transport: str='grpc_asyncio', request_type=policytagmanager.CreateTaxonomyRequest):
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_taxonomy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.Taxonomy(name='name_value', display_name='display_name_value', description='description_value', policy_tag_count=1715, activated_policy_types=[policytagmanager.Taxonomy.PolicyType.FINE_GRAINED_ACCESS_CONTROL]))
        response = await client.create_taxonomy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.CreateTaxonomyRequest()
    assert isinstance(response, policytagmanager.Taxonomy)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.policy_tag_count == 1715
    assert response.activated_policy_types == [policytagmanager.Taxonomy.PolicyType.FINE_GRAINED_ACCESS_CONTROL]

@pytest.mark.asyncio
async def test_create_taxonomy_async_from_dict():
    await test_create_taxonomy_async(request_type=dict)

def test_create_taxonomy_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.CreateTaxonomyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_taxonomy), '__call__') as call:
        call.return_value = policytagmanager.Taxonomy()
        client.create_taxonomy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_taxonomy_field_headers_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.CreateTaxonomyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_taxonomy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.Taxonomy())
        await client.create_taxonomy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_taxonomy_flattened():
    if False:
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_taxonomy), '__call__') as call:
        call.return_value = policytagmanager.Taxonomy()
        client.create_taxonomy(parent='parent_value', taxonomy=policytagmanager.Taxonomy(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].taxonomy
        mock_val = policytagmanager.Taxonomy(name='name_value')
        assert arg == mock_val

def test_create_taxonomy_flattened_error():
    if False:
        while True:
            i = 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_taxonomy(policytagmanager.CreateTaxonomyRequest(), parent='parent_value', taxonomy=policytagmanager.Taxonomy(name='name_value'))

@pytest.mark.asyncio
async def test_create_taxonomy_flattened_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_taxonomy), '__call__') as call:
        call.return_value = policytagmanager.Taxonomy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.Taxonomy())
        response = await client.create_taxonomy(parent='parent_value', taxonomy=policytagmanager.Taxonomy(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].taxonomy
        mock_val = policytagmanager.Taxonomy(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_taxonomy_flattened_error_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_taxonomy(policytagmanager.CreateTaxonomyRequest(), parent='parent_value', taxonomy=policytagmanager.Taxonomy(name='name_value'))

@pytest.mark.parametrize('request_type', [policytagmanager.DeleteTaxonomyRequest, dict])
def test_delete_taxonomy(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_taxonomy), '__call__') as call:
        call.return_value = None
        response = client.delete_taxonomy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.DeleteTaxonomyRequest()
    assert response is None

def test_delete_taxonomy_empty_call():
    if False:
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_taxonomy), '__call__') as call:
        client.delete_taxonomy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.DeleteTaxonomyRequest()

@pytest.mark.asyncio
async def test_delete_taxonomy_async(transport: str='grpc_asyncio', request_type=policytagmanager.DeleteTaxonomyRequest):
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_taxonomy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_taxonomy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.DeleteTaxonomyRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_taxonomy_async_from_dict():
    await test_delete_taxonomy_async(request_type=dict)

def test_delete_taxonomy_field_headers():
    if False:
        while True:
            i = 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.DeleteTaxonomyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_taxonomy), '__call__') as call:
        call.return_value = None
        client.delete_taxonomy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_taxonomy_field_headers_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.DeleteTaxonomyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_taxonomy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_taxonomy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_taxonomy_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_taxonomy), '__call__') as call:
        call.return_value = None
        client.delete_taxonomy(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_taxonomy_flattened_error():
    if False:
        print('Hello World!')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_taxonomy(policytagmanager.DeleteTaxonomyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_taxonomy_flattened_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_taxonomy), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_taxonomy(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_taxonomy_flattened_error_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_taxonomy(policytagmanager.DeleteTaxonomyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [policytagmanager.UpdateTaxonomyRequest, dict])
def test_update_taxonomy(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_taxonomy), '__call__') as call:
        call.return_value = policytagmanager.Taxonomy(name='name_value', display_name='display_name_value', description='description_value', policy_tag_count=1715, activated_policy_types=[policytagmanager.Taxonomy.PolicyType.FINE_GRAINED_ACCESS_CONTROL])
        response = client.update_taxonomy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.UpdateTaxonomyRequest()
    assert isinstance(response, policytagmanager.Taxonomy)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.policy_tag_count == 1715
    assert response.activated_policy_types == [policytagmanager.Taxonomy.PolicyType.FINE_GRAINED_ACCESS_CONTROL]

def test_update_taxonomy_empty_call():
    if False:
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_taxonomy), '__call__') as call:
        client.update_taxonomy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.UpdateTaxonomyRequest()

@pytest.mark.asyncio
async def test_update_taxonomy_async(transport: str='grpc_asyncio', request_type=policytagmanager.UpdateTaxonomyRequest):
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_taxonomy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.Taxonomy(name='name_value', display_name='display_name_value', description='description_value', policy_tag_count=1715, activated_policy_types=[policytagmanager.Taxonomy.PolicyType.FINE_GRAINED_ACCESS_CONTROL]))
        response = await client.update_taxonomy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.UpdateTaxonomyRequest()
    assert isinstance(response, policytagmanager.Taxonomy)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.policy_tag_count == 1715
    assert response.activated_policy_types == [policytagmanager.Taxonomy.PolicyType.FINE_GRAINED_ACCESS_CONTROL]

@pytest.mark.asyncio
async def test_update_taxonomy_async_from_dict():
    await test_update_taxonomy_async(request_type=dict)

def test_update_taxonomy_field_headers():
    if False:
        print('Hello World!')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.UpdateTaxonomyRequest()
    request.taxonomy.name = 'name_value'
    with mock.patch.object(type(client.transport.update_taxonomy), '__call__') as call:
        call.return_value = policytagmanager.Taxonomy()
        client.update_taxonomy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'taxonomy.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_taxonomy_field_headers_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.UpdateTaxonomyRequest()
    request.taxonomy.name = 'name_value'
    with mock.patch.object(type(client.transport.update_taxonomy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.Taxonomy())
        await client.update_taxonomy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'taxonomy.name=name_value') in kw['metadata']

def test_update_taxonomy_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_taxonomy), '__call__') as call:
        call.return_value = policytagmanager.Taxonomy()
        client.update_taxonomy(taxonomy=policytagmanager.Taxonomy(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].taxonomy
        mock_val = policytagmanager.Taxonomy(name='name_value')
        assert arg == mock_val

def test_update_taxonomy_flattened_error():
    if False:
        return 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_taxonomy(policytagmanager.UpdateTaxonomyRequest(), taxonomy=policytagmanager.Taxonomy(name='name_value'))

@pytest.mark.asyncio
async def test_update_taxonomy_flattened_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_taxonomy), '__call__') as call:
        call.return_value = policytagmanager.Taxonomy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.Taxonomy())
        response = await client.update_taxonomy(taxonomy=policytagmanager.Taxonomy(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].taxonomy
        mock_val = policytagmanager.Taxonomy(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_taxonomy_flattened_error_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_taxonomy(policytagmanager.UpdateTaxonomyRequest(), taxonomy=policytagmanager.Taxonomy(name='name_value'))

@pytest.mark.parametrize('request_type', [policytagmanager.ListTaxonomiesRequest, dict])
def test_list_taxonomies(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_taxonomies), '__call__') as call:
        call.return_value = policytagmanager.ListTaxonomiesResponse(next_page_token='next_page_token_value')
        response = client.list_taxonomies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.ListTaxonomiesRequest()
    assert isinstance(response, pagers.ListTaxonomiesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_taxonomies_empty_call():
    if False:
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_taxonomies), '__call__') as call:
        client.list_taxonomies()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.ListTaxonomiesRequest()

@pytest.mark.asyncio
async def test_list_taxonomies_async(transport: str='grpc_asyncio', request_type=policytagmanager.ListTaxonomiesRequest):
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_taxonomies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.ListTaxonomiesResponse(next_page_token='next_page_token_value'))
        response = await client.list_taxonomies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.ListTaxonomiesRequest()
    assert isinstance(response, pagers.ListTaxonomiesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_taxonomies_async_from_dict():
    await test_list_taxonomies_async(request_type=dict)

def test_list_taxonomies_field_headers():
    if False:
        while True:
            i = 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.ListTaxonomiesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_taxonomies), '__call__') as call:
        call.return_value = policytagmanager.ListTaxonomiesResponse()
        client.list_taxonomies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_taxonomies_field_headers_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.ListTaxonomiesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_taxonomies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.ListTaxonomiesResponse())
        await client.list_taxonomies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_taxonomies_flattened():
    if False:
        while True:
            i = 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_taxonomies), '__call__') as call:
        call.return_value = policytagmanager.ListTaxonomiesResponse()
        client.list_taxonomies(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_taxonomies_flattened_error():
    if False:
        print('Hello World!')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_taxonomies(policytagmanager.ListTaxonomiesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_taxonomies_flattened_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_taxonomies), '__call__') as call:
        call.return_value = policytagmanager.ListTaxonomiesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.ListTaxonomiesResponse())
        response = await client.list_taxonomies(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_taxonomies_flattened_error_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_taxonomies(policytagmanager.ListTaxonomiesRequest(), parent='parent_value')

def test_list_taxonomies_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_taxonomies), '__call__') as call:
        call.side_effect = (policytagmanager.ListTaxonomiesResponse(taxonomies=[policytagmanager.Taxonomy(), policytagmanager.Taxonomy(), policytagmanager.Taxonomy()], next_page_token='abc'), policytagmanager.ListTaxonomiesResponse(taxonomies=[], next_page_token='def'), policytagmanager.ListTaxonomiesResponse(taxonomies=[policytagmanager.Taxonomy()], next_page_token='ghi'), policytagmanager.ListTaxonomiesResponse(taxonomies=[policytagmanager.Taxonomy(), policytagmanager.Taxonomy()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_taxonomies(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, policytagmanager.Taxonomy) for i in results))

def test_list_taxonomies_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_taxonomies), '__call__') as call:
        call.side_effect = (policytagmanager.ListTaxonomiesResponse(taxonomies=[policytagmanager.Taxonomy(), policytagmanager.Taxonomy(), policytagmanager.Taxonomy()], next_page_token='abc'), policytagmanager.ListTaxonomiesResponse(taxonomies=[], next_page_token='def'), policytagmanager.ListTaxonomiesResponse(taxonomies=[policytagmanager.Taxonomy()], next_page_token='ghi'), policytagmanager.ListTaxonomiesResponse(taxonomies=[policytagmanager.Taxonomy(), policytagmanager.Taxonomy()]), RuntimeError)
        pages = list(client.list_taxonomies(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_taxonomies_async_pager():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_taxonomies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (policytagmanager.ListTaxonomiesResponse(taxonomies=[policytagmanager.Taxonomy(), policytagmanager.Taxonomy(), policytagmanager.Taxonomy()], next_page_token='abc'), policytagmanager.ListTaxonomiesResponse(taxonomies=[], next_page_token='def'), policytagmanager.ListTaxonomiesResponse(taxonomies=[policytagmanager.Taxonomy()], next_page_token='ghi'), policytagmanager.ListTaxonomiesResponse(taxonomies=[policytagmanager.Taxonomy(), policytagmanager.Taxonomy()]), RuntimeError)
        async_pager = await client.list_taxonomies(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, policytagmanager.Taxonomy) for i in responses))

@pytest.mark.asyncio
async def test_list_taxonomies_async_pages():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_taxonomies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (policytagmanager.ListTaxonomiesResponse(taxonomies=[policytagmanager.Taxonomy(), policytagmanager.Taxonomy(), policytagmanager.Taxonomy()], next_page_token='abc'), policytagmanager.ListTaxonomiesResponse(taxonomies=[], next_page_token='def'), policytagmanager.ListTaxonomiesResponse(taxonomies=[policytagmanager.Taxonomy()], next_page_token='ghi'), policytagmanager.ListTaxonomiesResponse(taxonomies=[policytagmanager.Taxonomy(), policytagmanager.Taxonomy()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_taxonomies(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [policytagmanager.GetTaxonomyRequest, dict])
def test_get_taxonomy(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_taxonomy), '__call__') as call:
        call.return_value = policytagmanager.Taxonomy(name='name_value', display_name='display_name_value', description='description_value', policy_tag_count=1715, activated_policy_types=[policytagmanager.Taxonomy.PolicyType.FINE_GRAINED_ACCESS_CONTROL])
        response = client.get_taxonomy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.GetTaxonomyRequest()
    assert isinstance(response, policytagmanager.Taxonomy)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.policy_tag_count == 1715
    assert response.activated_policy_types == [policytagmanager.Taxonomy.PolicyType.FINE_GRAINED_ACCESS_CONTROL]

def test_get_taxonomy_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_taxonomy), '__call__') as call:
        client.get_taxonomy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.GetTaxonomyRequest()

@pytest.mark.asyncio
async def test_get_taxonomy_async(transport: str='grpc_asyncio', request_type=policytagmanager.GetTaxonomyRequest):
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_taxonomy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.Taxonomy(name='name_value', display_name='display_name_value', description='description_value', policy_tag_count=1715, activated_policy_types=[policytagmanager.Taxonomy.PolicyType.FINE_GRAINED_ACCESS_CONTROL]))
        response = await client.get_taxonomy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.GetTaxonomyRequest()
    assert isinstance(response, policytagmanager.Taxonomy)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.policy_tag_count == 1715
    assert response.activated_policy_types == [policytagmanager.Taxonomy.PolicyType.FINE_GRAINED_ACCESS_CONTROL]

@pytest.mark.asyncio
async def test_get_taxonomy_async_from_dict():
    await test_get_taxonomy_async(request_type=dict)

def test_get_taxonomy_field_headers():
    if False:
        print('Hello World!')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.GetTaxonomyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_taxonomy), '__call__') as call:
        call.return_value = policytagmanager.Taxonomy()
        client.get_taxonomy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_taxonomy_field_headers_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.GetTaxonomyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_taxonomy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.Taxonomy())
        await client.get_taxonomy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_taxonomy_flattened():
    if False:
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_taxonomy), '__call__') as call:
        call.return_value = policytagmanager.Taxonomy()
        client.get_taxonomy(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_taxonomy_flattened_error():
    if False:
        while True:
            i = 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_taxonomy(policytagmanager.GetTaxonomyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_taxonomy_flattened_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_taxonomy), '__call__') as call:
        call.return_value = policytagmanager.Taxonomy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.Taxonomy())
        response = await client.get_taxonomy(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_taxonomy_flattened_error_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_taxonomy(policytagmanager.GetTaxonomyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [policytagmanager.CreatePolicyTagRequest, dict])
def test_create_policy_tag(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_policy_tag), '__call__') as call:
        call.return_value = policytagmanager.PolicyTag(name='name_value', display_name='display_name_value', description='description_value', parent_policy_tag='parent_policy_tag_value', child_policy_tags=['child_policy_tags_value'])
        response = client.create_policy_tag(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.CreatePolicyTagRequest()
    assert isinstance(response, policytagmanager.PolicyTag)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.parent_policy_tag == 'parent_policy_tag_value'
    assert response.child_policy_tags == ['child_policy_tags_value']

def test_create_policy_tag_empty_call():
    if False:
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_policy_tag), '__call__') as call:
        client.create_policy_tag()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.CreatePolicyTagRequest()

@pytest.mark.asyncio
async def test_create_policy_tag_async(transport: str='grpc_asyncio', request_type=policytagmanager.CreatePolicyTagRequest):
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_policy_tag), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.PolicyTag(name='name_value', display_name='display_name_value', description='description_value', parent_policy_tag='parent_policy_tag_value', child_policy_tags=['child_policy_tags_value']))
        response = await client.create_policy_tag(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.CreatePolicyTagRequest()
    assert isinstance(response, policytagmanager.PolicyTag)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.parent_policy_tag == 'parent_policy_tag_value'
    assert response.child_policy_tags == ['child_policy_tags_value']

@pytest.mark.asyncio
async def test_create_policy_tag_async_from_dict():
    await test_create_policy_tag_async(request_type=dict)

def test_create_policy_tag_field_headers():
    if False:
        print('Hello World!')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.CreatePolicyTagRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_policy_tag), '__call__') as call:
        call.return_value = policytagmanager.PolicyTag()
        client.create_policy_tag(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_policy_tag_field_headers_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.CreatePolicyTagRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_policy_tag), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.PolicyTag())
        await client.create_policy_tag(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_policy_tag_flattened():
    if False:
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_policy_tag), '__call__') as call:
        call.return_value = policytagmanager.PolicyTag()
        client.create_policy_tag(parent='parent_value', policy_tag=policytagmanager.PolicyTag(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].policy_tag
        mock_val = policytagmanager.PolicyTag(name='name_value')
        assert arg == mock_val

def test_create_policy_tag_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_policy_tag(policytagmanager.CreatePolicyTagRequest(), parent='parent_value', policy_tag=policytagmanager.PolicyTag(name='name_value'))

@pytest.mark.asyncio
async def test_create_policy_tag_flattened_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_policy_tag), '__call__') as call:
        call.return_value = policytagmanager.PolicyTag()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.PolicyTag())
        response = await client.create_policy_tag(parent='parent_value', policy_tag=policytagmanager.PolicyTag(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].policy_tag
        mock_val = policytagmanager.PolicyTag(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_policy_tag_flattened_error_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_policy_tag(policytagmanager.CreatePolicyTagRequest(), parent='parent_value', policy_tag=policytagmanager.PolicyTag(name='name_value'))

@pytest.mark.parametrize('request_type', [policytagmanager.DeletePolicyTagRequest, dict])
def test_delete_policy_tag(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_policy_tag), '__call__') as call:
        call.return_value = None
        response = client.delete_policy_tag(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.DeletePolicyTagRequest()
    assert response is None

def test_delete_policy_tag_empty_call():
    if False:
        print('Hello World!')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_policy_tag), '__call__') as call:
        client.delete_policy_tag()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.DeletePolicyTagRequest()

@pytest.mark.asyncio
async def test_delete_policy_tag_async(transport: str='grpc_asyncio', request_type=policytagmanager.DeletePolicyTagRequest):
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_policy_tag), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_policy_tag(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.DeletePolicyTagRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_policy_tag_async_from_dict():
    await test_delete_policy_tag_async(request_type=dict)

def test_delete_policy_tag_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.DeletePolicyTagRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_policy_tag), '__call__') as call:
        call.return_value = None
        client.delete_policy_tag(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_policy_tag_field_headers_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.DeletePolicyTagRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_policy_tag), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_policy_tag(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_policy_tag_flattened():
    if False:
        print('Hello World!')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_policy_tag), '__call__') as call:
        call.return_value = None
        client.delete_policy_tag(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_policy_tag_flattened_error():
    if False:
        print('Hello World!')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_policy_tag(policytagmanager.DeletePolicyTagRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_policy_tag_flattened_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_policy_tag), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_policy_tag(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_policy_tag_flattened_error_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_policy_tag(policytagmanager.DeletePolicyTagRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [policytagmanager.UpdatePolicyTagRequest, dict])
def test_update_policy_tag(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_policy_tag), '__call__') as call:
        call.return_value = policytagmanager.PolicyTag(name='name_value', display_name='display_name_value', description='description_value', parent_policy_tag='parent_policy_tag_value', child_policy_tags=['child_policy_tags_value'])
        response = client.update_policy_tag(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.UpdatePolicyTagRequest()
    assert isinstance(response, policytagmanager.PolicyTag)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.parent_policy_tag == 'parent_policy_tag_value'
    assert response.child_policy_tags == ['child_policy_tags_value']

def test_update_policy_tag_empty_call():
    if False:
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_policy_tag), '__call__') as call:
        client.update_policy_tag()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.UpdatePolicyTagRequest()

@pytest.mark.asyncio
async def test_update_policy_tag_async(transport: str='grpc_asyncio', request_type=policytagmanager.UpdatePolicyTagRequest):
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_policy_tag), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.PolicyTag(name='name_value', display_name='display_name_value', description='description_value', parent_policy_tag='parent_policy_tag_value', child_policy_tags=['child_policy_tags_value']))
        response = await client.update_policy_tag(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.UpdatePolicyTagRequest()
    assert isinstance(response, policytagmanager.PolicyTag)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.parent_policy_tag == 'parent_policy_tag_value'
    assert response.child_policy_tags == ['child_policy_tags_value']

@pytest.mark.asyncio
async def test_update_policy_tag_async_from_dict():
    await test_update_policy_tag_async(request_type=dict)

def test_update_policy_tag_field_headers():
    if False:
        return 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.UpdatePolicyTagRequest()
    request.policy_tag.name = 'name_value'
    with mock.patch.object(type(client.transport.update_policy_tag), '__call__') as call:
        call.return_value = policytagmanager.PolicyTag()
        client.update_policy_tag(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'policy_tag.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_policy_tag_field_headers_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.UpdatePolicyTagRequest()
    request.policy_tag.name = 'name_value'
    with mock.patch.object(type(client.transport.update_policy_tag), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.PolicyTag())
        await client.update_policy_tag(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'policy_tag.name=name_value') in kw['metadata']

def test_update_policy_tag_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_policy_tag), '__call__') as call:
        call.return_value = policytagmanager.PolicyTag()
        client.update_policy_tag(policy_tag=policytagmanager.PolicyTag(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].policy_tag
        mock_val = policytagmanager.PolicyTag(name='name_value')
        assert arg == mock_val

def test_update_policy_tag_flattened_error():
    if False:
        return 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_policy_tag(policytagmanager.UpdatePolicyTagRequest(), policy_tag=policytagmanager.PolicyTag(name='name_value'))

@pytest.mark.asyncio
async def test_update_policy_tag_flattened_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_policy_tag), '__call__') as call:
        call.return_value = policytagmanager.PolicyTag()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.PolicyTag())
        response = await client.update_policy_tag(policy_tag=policytagmanager.PolicyTag(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].policy_tag
        mock_val = policytagmanager.PolicyTag(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_policy_tag_flattened_error_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_policy_tag(policytagmanager.UpdatePolicyTagRequest(), policy_tag=policytagmanager.PolicyTag(name='name_value'))

@pytest.mark.parametrize('request_type', [policytagmanager.ListPolicyTagsRequest, dict])
def test_list_policy_tags(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_policy_tags), '__call__') as call:
        call.return_value = policytagmanager.ListPolicyTagsResponse(next_page_token='next_page_token_value')
        response = client.list_policy_tags(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.ListPolicyTagsRequest()
    assert isinstance(response, pagers.ListPolicyTagsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_policy_tags_empty_call():
    if False:
        return 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_policy_tags), '__call__') as call:
        client.list_policy_tags()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.ListPolicyTagsRequest()

@pytest.mark.asyncio
async def test_list_policy_tags_async(transport: str='grpc_asyncio', request_type=policytagmanager.ListPolicyTagsRequest):
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_policy_tags), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.ListPolicyTagsResponse(next_page_token='next_page_token_value'))
        response = await client.list_policy_tags(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.ListPolicyTagsRequest()
    assert isinstance(response, pagers.ListPolicyTagsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_policy_tags_async_from_dict():
    await test_list_policy_tags_async(request_type=dict)

def test_list_policy_tags_field_headers():
    if False:
        return 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.ListPolicyTagsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_policy_tags), '__call__') as call:
        call.return_value = policytagmanager.ListPolicyTagsResponse()
        client.list_policy_tags(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_policy_tags_field_headers_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.ListPolicyTagsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_policy_tags), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.ListPolicyTagsResponse())
        await client.list_policy_tags(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_policy_tags_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_policy_tags), '__call__') as call:
        call.return_value = policytagmanager.ListPolicyTagsResponse()
        client.list_policy_tags(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_policy_tags_flattened_error():
    if False:
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_policy_tags(policytagmanager.ListPolicyTagsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_policy_tags_flattened_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_policy_tags), '__call__') as call:
        call.return_value = policytagmanager.ListPolicyTagsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.ListPolicyTagsResponse())
        response = await client.list_policy_tags(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_policy_tags_flattened_error_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_policy_tags(policytagmanager.ListPolicyTagsRequest(), parent='parent_value')

def test_list_policy_tags_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_policy_tags), '__call__') as call:
        call.side_effect = (policytagmanager.ListPolicyTagsResponse(policy_tags=[policytagmanager.PolicyTag(), policytagmanager.PolicyTag(), policytagmanager.PolicyTag()], next_page_token='abc'), policytagmanager.ListPolicyTagsResponse(policy_tags=[], next_page_token='def'), policytagmanager.ListPolicyTagsResponse(policy_tags=[policytagmanager.PolicyTag()], next_page_token='ghi'), policytagmanager.ListPolicyTagsResponse(policy_tags=[policytagmanager.PolicyTag(), policytagmanager.PolicyTag()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_policy_tags(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, policytagmanager.PolicyTag) for i in results))

def test_list_policy_tags_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_policy_tags), '__call__') as call:
        call.side_effect = (policytagmanager.ListPolicyTagsResponse(policy_tags=[policytagmanager.PolicyTag(), policytagmanager.PolicyTag(), policytagmanager.PolicyTag()], next_page_token='abc'), policytagmanager.ListPolicyTagsResponse(policy_tags=[], next_page_token='def'), policytagmanager.ListPolicyTagsResponse(policy_tags=[policytagmanager.PolicyTag()], next_page_token='ghi'), policytagmanager.ListPolicyTagsResponse(policy_tags=[policytagmanager.PolicyTag(), policytagmanager.PolicyTag()]), RuntimeError)
        pages = list(client.list_policy_tags(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_policy_tags_async_pager():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_policy_tags), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (policytagmanager.ListPolicyTagsResponse(policy_tags=[policytagmanager.PolicyTag(), policytagmanager.PolicyTag(), policytagmanager.PolicyTag()], next_page_token='abc'), policytagmanager.ListPolicyTagsResponse(policy_tags=[], next_page_token='def'), policytagmanager.ListPolicyTagsResponse(policy_tags=[policytagmanager.PolicyTag()], next_page_token='ghi'), policytagmanager.ListPolicyTagsResponse(policy_tags=[policytagmanager.PolicyTag(), policytagmanager.PolicyTag()]), RuntimeError)
        async_pager = await client.list_policy_tags(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, policytagmanager.PolicyTag) for i in responses))

@pytest.mark.asyncio
async def test_list_policy_tags_async_pages():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_policy_tags), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (policytagmanager.ListPolicyTagsResponse(policy_tags=[policytagmanager.PolicyTag(), policytagmanager.PolicyTag(), policytagmanager.PolicyTag()], next_page_token='abc'), policytagmanager.ListPolicyTagsResponse(policy_tags=[], next_page_token='def'), policytagmanager.ListPolicyTagsResponse(policy_tags=[policytagmanager.PolicyTag()], next_page_token='ghi'), policytagmanager.ListPolicyTagsResponse(policy_tags=[policytagmanager.PolicyTag(), policytagmanager.PolicyTag()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_policy_tags(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [policytagmanager.GetPolicyTagRequest, dict])
def test_get_policy_tag(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_policy_tag), '__call__') as call:
        call.return_value = policytagmanager.PolicyTag(name='name_value', display_name='display_name_value', description='description_value', parent_policy_tag='parent_policy_tag_value', child_policy_tags=['child_policy_tags_value'])
        response = client.get_policy_tag(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.GetPolicyTagRequest()
    assert isinstance(response, policytagmanager.PolicyTag)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.parent_policy_tag == 'parent_policy_tag_value'
    assert response.child_policy_tags == ['child_policy_tags_value']

def test_get_policy_tag_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_policy_tag), '__call__') as call:
        client.get_policy_tag()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.GetPolicyTagRequest()

@pytest.mark.asyncio
async def test_get_policy_tag_async(transport: str='grpc_asyncio', request_type=policytagmanager.GetPolicyTagRequest):
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_policy_tag), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.PolicyTag(name='name_value', display_name='display_name_value', description='description_value', parent_policy_tag='parent_policy_tag_value', child_policy_tags=['child_policy_tags_value']))
        response = await client.get_policy_tag(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == policytagmanager.GetPolicyTagRequest()
    assert isinstance(response, policytagmanager.PolicyTag)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.parent_policy_tag == 'parent_policy_tag_value'
    assert response.child_policy_tags == ['child_policy_tags_value']

@pytest.mark.asyncio
async def test_get_policy_tag_async_from_dict():
    await test_get_policy_tag_async(request_type=dict)

def test_get_policy_tag_field_headers():
    if False:
        return 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.GetPolicyTagRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_policy_tag), '__call__') as call:
        call.return_value = policytagmanager.PolicyTag()
        client.get_policy_tag(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_policy_tag_field_headers_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = policytagmanager.GetPolicyTagRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_policy_tag), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.PolicyTag())
        await client.get_policy_tag(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_policy_tag_flattened():
    if False:
        while True:
            i = 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_policy_tag), '__call__') as call:
        call.return_value = policytagmanager.PolicyTag()
        client.get_policy_tag(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_policy_tag_flattened_error():
    if False:
        return 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_policy_tag(policytagmanager.GetPolicyTagRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_policy_tag_flattened_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_policy_tag), '__call__') as call:
        call.return_value = policytagmanager.PolicyTag()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policytagmanager.PolicyTag())
        response = await client.get_policy_tag(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_policy_tag_flattened_error_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_policy_tag(policytagmanager.GetPolicyTagRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy(request_type, transport: str='grpc'):
    if False:
        return 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        client.get_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()

@pytest.mark.asyncio
async def test_get_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.GetIamPolicyRequest):
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        client.set_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()

@pytest.mark.asyncio
async def test_set_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.SetIamPolicyRequest):
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774), 'update_mask': field_mask_pb2.FieldMask(paths=['paths_value'])})
        call.assert_called()

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        client.test_iam_permissions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()

@pytest.mark.asyncio
async def test_test_iam_permissions_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.PolicyTagManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.PolicyTagManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = PolicyTagManagerClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.PolicyTagManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = PolicyTagManagerClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = PolicyTagManagerClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.PolicyTagManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = PolicyTagManagerClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.PolicyTagManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = PolicyTagManagerClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.PolicyTagManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.PolicyTagManagerGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.PolicyTagManagerGrpcTransport, transports.PolicyTagManagerGrpcAsyncIOTransport])
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
        while True:
            i = 10
    transport = PolicyTagManagerClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.PolicyTagManagerGrpcTransport)

def test_policy_tag_manager_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.PolicyTagManagerTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_policy_tag_manager_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.datacatalog_v1beta1.services.policy_tag_manager.transports.PolicyTagManagerTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.PolicyTagManagerTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_taxonomy', 'delete_taxonomy', 'update_taxonomy', 'list_taxonomies', 'get_taxonomy', 'create_policy_tag', 'delete_policy_tag', 'update_policy_tag', 'list_policy_tags', 'get_policy_tag', 'get_iam_policy', 'set_iam_policy', 'test_iam_permissions')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_policy_tag_manager_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.datacatalog_v1beta1.services.policy_tag_manager.transports.PolicyTagManagerTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.PolicyTagManagerTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_policy_tag_manager_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.datacatalog_v1beta1.services.policy_tag_manager.transports.PolicyTagManagerTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.PolicyTagManagerTransport()
        adc.assert_called_once()

def test_policy_tag_manager_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        PolicyTagManagerClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.PolicyTagManagerGrpcTransport, transports.PolicyTagManagerGrpcAsyncIOTransport])
def test_policy_tag_manager_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.PolicyTagManagerGrpcTransport, transports.PolicyTagManagerGrpcAsyncIOTransport])
def test_policy_tag_manager_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.PolicyTagManagerGrpcTransport, grpc_helpers), (transports.PolicyTagManagerGrpcAsyncIOTransport, grpc_helpers_async)])
def test_policy_tag_manager_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('datacatalog.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='datacatalog.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.PolicyTagManagerGrpcTransport, transports.PolicyTagManagerGrpcAsyncIOTransport])
def test_policy_tag_manager_grpc_transport_client_cert_source_for_mtls(transport_class):
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
def test_policy_tag_manager_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='datacatalog.googleapis.com'), transport=transport_name)
    assert client.transport._host == 'datacatalog.googleapis.com:443'

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio'])
def test_policy_tag_manager_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='datacatalog.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == 'datacatalog.googleapis.com:8000'

def test_policy_tag_manager_grpc_transport_channel():
    if False:
        print('Hello World!')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.PolicyTagManagerGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_policy_tag_manager_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.PolicyTagManagerGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.PolicyTagManagerGrpcTransport, transports.PolicyTagManagerGrpcAsyncIOTransport])
def test_policy_tag_manager_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.PolicyTagManagerGrpcTransport, transports.PolicyTagManagerGrpcAsyncIOTransport])
def test_policy_tag_manager_transport_channel_mtls_with_adc(transport_class):
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

def test_policy_tag_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    taxonomy = 'whelk'
    policy_tag = 'octopus'
    expected = 'projects/{project}/locations/{location}/taxonomies/{taxonomy}/policyTags/{policy_tag}'.format(project=project, location=location, taxonomy=taxonomy, policy_tag=policy_tag)
    actual = PolicyTagManagerClient.policy_tag_path(project, location, taxonomy, policy_tag)
    assert expected == actual

def test_parse_policy_tag_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'oyster', 'location': 'nudibranch', 'taxonomy': 'cuttlefish', 'policy_tag': 'mussel'}
    path = PolicyTagManagerClient.policy_tag_path(**expected)
    actual = PolicyTagManagerClient.parse_policy_tag_path(path)
    assert expected == actual

def test_taxonomy_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    location = 'nautilus'
    taxonomy = 'scallop'
    expected = 'projects/{project}/locations/{location}/taxonomies/{taxonomy}'.format(project=project, location=location, taxonomy=taxonomy)
    actual = PolicyTagManagerClient.taxonomy_path(project, location, taxonomy)
    assert expected == actual

def test_parse_taxonomy_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'abalone', 'location': 'squid', 'taxonomy': 'clam'}
    path = PolicyTagManagerClient.taxonomy_path(**expected)
    actual = PolicyTagManagerClient.parse_taxonomy_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = PolicyTagManagerClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'octopus'}
    path = PolicyTagManagerClient.common_billing_account_path(**expected)
    actual = PolicyTagManagerClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = PolicyTagManagerClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'nudibranch'}
    path = PolicyTagManagerClient.common_folder_path(**expected)
    actual = PolicyTagManagerClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        while True:
            i = 10
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = PolicyTagManagerClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'mussel'}
    path = PolicyTagManagerClient.common_organization_path(**expected)
    actual = PolicyTagManagerClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = PolicyTagManagerClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nautilus'}
    path = PolicyTagManagerClient.common_project_path(**expected)
    actual = PolicyTagManagerClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = PolicyTagManagerClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'squid', 'location': 'clam'}
    path = PolicyTagManagerClient.common_location_path(**expected)
    actual = PolicyTagManagerClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.PolicyTagManagerTransport, '_prep_wrapped_messages') as prep:
        client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.PolicyTagManagerTransport, '_prep_wrapped_messages') as prep:
        transport_class = PolicyTagManagerClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = PolicyTagManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        return 10
    transports = {'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['grpc']
    for transport in transports:
        client = PolicyTagManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(PolicyTagManagerClient, transports.PolicyTagManagerGrpcTransport), (PolicyTagManagerAsyncClient, transports.PolicyTagManagerGrpcAsyncIOTransport)])
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