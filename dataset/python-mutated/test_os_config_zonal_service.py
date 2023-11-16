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
from google.protobuf import duration_pb2
from google.protobuf import empty_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.osconfig_v1alpha.services.os_config_zonal_service import OsConfigZonalServiceAsyncClient, OsConfigZonalServiceClient, pagers, transports
from google.cloud.osconfig_v1alpha.types import config_common, instance_os_policies_compliance, inventory, os_policy, os_policy_assignment_reports, os_policy_assignments, osconfig_common, vulnerability

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        while True:
            i = 10
    return 'foo.googleapis.com' if 'localhost' in client.DEFAULT_ENDPOINT else client.DEFAULT_ENDPOINT

def test__get_default_mtls_endpoint():
    if False:
        print('Hello World!')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert OsConfigZonalServiceClient._get_default_mtls_endpoint(None) is None
    assert OsConfigZonalServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert OsConfigZonalServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert OsConfigZonalServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert OsConfigZonalServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert OsConfigZonalServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(OsConfigZonalServiceClient, 'grpc'), (OsConfigZonalServiceAsyncClient, 'grpc_asyncio'), (OsConfigZonalServiceClient, 'rest')])
def test_os_config_zonal_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('osconfig.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://osconfig.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.OsConfigZonalServiceGrpcTransport, 'grpc'), (transports.OsConfigZonalServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.OsConfigZonalServiceRestTransport, 'rest')])
def test_os_config_zonal_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(OsConfigZonalServiceClient, 'grpc'), (OsConfigZonalServiceAsyncClient, 'grpc_asyncio'), (OsConfigZonalServiceClient, 'rest')])
def test_os_config_zonal_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('osconfig.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://osconfig.googleapis.com')

def test_os_config_zonal_service_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = OsConfigZonalServiceClient.get_transport_class()
    available_transports = [transports.OsConfigZonalServiceGrpcTransport, transports.OsConfigZonalServiceRestTransport]
    assert transport in available_transports
    transport = OsConfigZonalServiceClient.get_transport_class('grpc')
    assert transport == transports.OsConfigZonalServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(OsConfigZonalServiceClient, transports.OsConfigZonalServiceGrpcTransport, 'grpc'), (OsConfigZonalServiceAsyncClient, transports.OsConfigZonalServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (OsConfigZonalServiceClient, transports.OsConfigZonalServiceRestTransport, 'rest')])
@mock.patch.object(OsConfigZonalServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(OsConfigZonalServiceClient))
@mock.patch.object(OsConfigZonalServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(OsConfigZonalServiceAsyncClient))
def test_os_config_zonal_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(OsConfigZonalServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(OsConfigZonalServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(OsConfigZonalServiceClient, transports.OsConfigZonalServiceGrpcTransport, 'grpc', 'true'), (OsConfigZonalServiceAsyncClient, transports.OsConfigZonalServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (OsConfigZonalServiceClient, transports.OsConfigZonalServiceGrpcTransport, 'grpc', 'false'), (OsConfigZonalServiceAsyncClient, transports.OsConfigZonalServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (OsConfigZonalServiceClient, transports.OsConfigZonalServiceRestTransport, 'rest', 'true'), (OsConfigZonalServiceClient, transports.OsConfigZonalServiceRestTransport, 'rest', 'false')])
@mock.patch.object(OsConfigZonalServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(OsConfigZonalServiceClient))
@mock.patch.object(OsConfigZonalServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(OsConfigZonalServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_os_config_zonal_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [OsConfigZonalServiceClient, OsConfigZonalServiceAsyncClient])
@mock.patch.object(OsConfigZonalServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(OsConfigZonalServiceClient))
@mock.patch.object(OsConfigZonalServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(OsConfigZonalServiceAsyncClient))
def test_os_config_zonal_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(OsConfigZonalServiceClient, transports.OsConfigZonalServiceGrpcTransport, 'grpc'), (OsConfigZonalServiceAsyncClient, transports.OsConfigZonalServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (OsConfigZonalServiceClient, transports.OsConfigZonalServiceRestTransport, 'rest')])
def test_os_config_zonal_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(OsConfigZonalServiceClient, transports.OsConfigZonalServiceGrpcTransport, 'grpc', grpc_helpers), (OsConfigZonalServiceAsyncClient, transports.OsConfigZonalServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (OsConfigZonalServiceClient, transports.OsConfigZonalServiceRestTransport, 'rest', None)])
def test_os_config_zonal_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_os_config_zonal_service_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.osconfig_v1alpha.services.os_config_zonal_service.transports.OsConfigZonalServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = OsConfigZonalServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(OsConfigZonalServiceClient, transports.OsConfigZonalServiceGrpcTransport, 'grpc', grpc_helpers), (OsConfigZonalServiceAsyncClient, transports.OsConfigZonalServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_os_config_zonal_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('osconfig.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='osconfig.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [os_policy_assignments.CreateOSPolicyAssignmentRequest, dict])
def test_create_os_policy_assignment(request_type, transport: str='grpc'):
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_os_policy_assignment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_os_policy_assignment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignments.CreateOSPolicyAssignmentRequest()
    assert isinstance(response, future.Future)

def test_create_os_policy_assignment_empty_call():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_os_policy_assignment), '__call__') as call:
        client.create_os_policy_assignment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignments.CreateOSPolicyAssignmentRequest()

@pytest.mark.asyncio
async def test_create_os_policy_assignment_async(transport: str='grpc_asyncio', request_type=os_policy_assignments.CreateOSPolicyAssignmentRequest):
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_os_policy_assignment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_os_policy_assignment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignments.CreateOSPolicyAssignmentRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_os_policy_assignment_async_from_dict():
    await test_create_os_policy_assignment_async(request_type=dict)

def test_create_os_policy_assignment_field_headers():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = os_policy_assignments.CreateOSPolicyAssignmentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_os_policy_assignment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_os_policy_assignment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_os_policy_assignment_field_headers_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = os_policy_assignments.CreateOSPolicyAssignmentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_os_policy_assignment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_os_policy_assignment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_os_policy_assignment_flattened():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_os_policy_assignment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_os_policy_assignment(parent='parent_value', os_policy_assignment=os_policy_assignments.OSPolicyAssignment(name='name_value'), os_policy_assignment_id='os_policy_assignment_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].os_policy_assignment
        mock_val = os_policy_assignments.OSPolicyAssignment(name='name_value')
        assert arg == mock_val
        arg = args[0].os_policy_assignment_id
        mock_val = 'os_policy_assignment_id_value'
        assert arg == mock_val

def test_create_os_policy_assignment_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_os_policy_assignment(os_policy_assignments.CreateOSPolicyAssignmentRequest(), parent='parent_value', os_policy_assignment=os_policy_assignments.OSPolicyAssignment(name='name_value'), os_policy_assignment_id='os_policy_assignment_id_value')

@pytest.mark.asyncio
async def test_create_os_policy_assignment_flattened_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_os_policy_assignment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_os_policy_assignment(parent='parent_value', os_policy_assignment=os_policy_assignments.OSPolicyAssignment(name='name_value'), os_policy_assignment_id='os_policy_assignment_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].os_policy_assignment
        mock_val = os_policy_assignments.OSPolicyAssignment(name='name_value')
        assert arg == mock_val
        arg = args[0].os_policy_assignment_id
        mock_val = 'os_policy_assignment_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_os_policy_assignment_flattened_error_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_os_policy_assignment(os_policy_assignments.CreateOSPolicyAssignmentRequest(), parent='parent_value', os_policy_assignment=os_policy_assignments.OSPolicyAssignment(name='name_value'), os_policy_assignment_id='os_policy_assignment_id_value')

@pytest.mark.parametrize('request_type', [os_policy_assignments.UpdateOSPolicyAssignmentRequest, dict])
def test_update_os_policy_assignment(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_os_policy_assignment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_os_policy_assignment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignments.UpdateOSPolicyAssignmentRequest()
    assert isinstance(response, future.Future)

def test_update_os_policy_assignment_empty_call():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_os_policy_assignment), '__call__') as call:
        client.update_os_policy_assignment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignments.UpdateOSPolicyAssignmentRequest()

@pytest.mark.asyncio
async def test_update_os_policy_assignment_async(transport: str='grpc_asyncio', request_type=os_policy_assignments.UpdateOSPolicyAssignmentRequest):
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_os_policy_assignment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_os_policy_assignment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignments.UpdateOSPolicyAssignmentRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_os_policy_assignment_async_from_dict():
    await test_update_os_policy_assignment_async(request_type=dict)

def test_update_os_policy_assignment_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = os_policy_assignments.UpdateOSPolicyAssignmentRequest()
    request.os_policy_assignment.name = 'name_value'
    with mock.patch.object(type(client.transport.update_os_policy_assignment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_os_policy_assignment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'os_policy_assignment.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_os_policy_assignment_field_headers_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = os_policy_assignments.UpdateOSPolicyAssignmentRequest()
    request.os_policy_assignment.name = 'name_value'
    with mock.patch.object(type(client.transport.update_os_policy_assignment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_os_policy_assignment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'os_policy_assignment.name=name_value') in kw['metadata']

def test_update_os_policy_assignment_flattened():
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_os_policy_assignment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_os_policy_assignment(os_policy_assignment=os_policy_assignments.OSPolicyAssignment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].os_policy_assignment
        mock_val = os_policy_assignments.OSPolicyAssignment(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_os_policy_assignment_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_os_policy_assignment(os_policy_assignments.UpdateOSPolicyAssignmentRequest(), os_policy_assignment=os_policy_assignments.OSPolicyAssignment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_os_policy_assignment_flattened_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_os_policy_assignment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_os_policy_assignment(os_policy_assignment=os_policy_assignments.OSPolicyAssignment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].os_policy_assignment
        mock_val = os_policy_assignments.OSPolicyAssignment(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_os_policy_assignment_flattened_error_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_os_policy_assignment(os_policy_assignments.UpdateOSPolicyAssignmentRequest(), os_policy_assignment=os_policy_assignments.OSPolicyAssignment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [os_policy_assignments.GetOSPolicyAssignmentRequest, dict])
def test_get_os_policy_assignment(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_os_policy_assignment), '__call__') as call:
        call.return_value = os_policy_assignments.OSPolicyAssignment(name='name_value', description='description_value', revision_id='revision_id_value', etag='etag_value', rollout_state=os_policy_assignments.OSPolicyAssignment.RolloutState.IN_PROGRESS, baseline=True, deleted=True, reconciling=True, uid='uid_value')
        response = client.get_os_policy_assignment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignments.GetOSPolicyAssignmentRequest()
    assert isinstance(response, os_policy_assignments.OSPolicyAssignment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.revision_id == 'revision_id_value'
    assert response.etag == 'etag_value'
    assert response.rollout_state == os_policy_assignments.OSPolicyAssignment.RolloutState.IN_PROGRESS
    assert response.baseline is True
    assert response.deleted is True
    assert response.reconciling is True
    assert response.uid == 'uid_value'

def test_get_os_policy_assignment_empty_call():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_os_policy_assignment), '__call__') as call:
        client.get_os_policy_assignment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignments.GetOSPolicyAssignmentRequest()

@pytest.mark.asyncio
async def test_get_os_policy_assignment_async(transport: str='grpc_asyncio', request_type=os_policy_assignments.GetOSPolicyAssignmentRequest):
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_os_policy_assignment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(os_policy_assignments.OSPolicyAssignment(name='name_value', description='description_value', revision_id='revision_id_value', etag='etag_value', rollout_state=os_policy_assignments.OSPolicyAssignment.RolloutState.IN_PROGRESS, baseline=True, deleted=True, reconciling=True, uid='uid_value'))
        response = await client.get_os_policy_assignment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignments.GetOSPolicyAssignmentRequest()
    assert isinstance(response, os_policy_assignments.OSPolicyAssignment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.revision_id == 'revision_id_value'
    assert response.etag == 'etag_value'
    assert response.rollout_state == os_policy_assignments.OSPolicyAssignment.RolloutState.IN_PROGRESS
    assert response.baseline is True
    assert response.deleted is True
    assert response.reconciling is True
    assert response.uid == 'uid_value'

@pytest.mark.asyncio
async def test_get_os_policy_assignment_async_from_dict():
    await test_get_os_policy_assignment_async(request_type=dict)

def test_get_os_policy_assignment_field_headers():
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = os_policy_assignments.GetOSPolicyAssignmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_os_policy_assignment), '__call__') as call:
        call.return_value = os_policy_assignments.OSPolicyAssignment()
        client.get_os_policy_assignment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_os_policy_assignment_field_headers_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = os_policy_assignments.GetOSPolicyAssignmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_os_policy_assignment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(os_policy_assignments.OSPolicyAssignment())
        await client.get_os_policy_assignment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_os_policy_assignment_flattened():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_os_policy_assignment), '__call__') as call:
        call.return_value = os_policy_assignments.OSPolicyAssignment()
        client.get_os_policy_assignment(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_os_policy_assignment_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_os_policy_assignment(os_policy_assignments.GetOSPolicyAssignmentRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_os_policy_assignment_flattened_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_os_policy_assignment), '__call__') as call:
        call.return_value = os_policy_assignments.OSPolicyAssignment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(os_policy_assignments.OSPolicyAssignment())
        response = await client.get_os_policy_assignment(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_os_policy_assignment_flattened_error_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_os_policy_assignment(os_policy_assignments.GetOSPolicyAssignmentRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [os_policy_assignments.ListOSPolicyAssignmentsRequest, dict])
def test_list_os_policy_assignments(request_type, transport: str='grpc'):
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_os_policy_assignments), '__call__') as call:
        call.return_value = os_policy_assignments.ListOSPolicyAssignmentsResponse(next_page_token='next_page_token_value')
        response = client.list_os_policy_assignments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignments.ListOSPolicyAssignmentsRequest()
    assert isinstance(response, pagers.ListOSPolicyAssignmentsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_os_policy_assignments_empty_call():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_os_policy_assignments), '__call__') as call:
        client.list_os_policy_assignments()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignments.ListOSPolicyAssignmentsRequest()

@pytest.mark.asyncio
async def test_list_os_policy_assignments_async(transport: str='grpc_asyncio', request_type=os_policy_assignments.ListOSPolicyAssignmentsRequest):
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_os_policy_assignments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(os_policy_assignments.ListOSPolicyAssignmentsResponse(next_page_token='next_page_token_value'))
        response = await client.list_os_policy_assignments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignments.ListOSPolicyAssignmentsRequest()
    assert isinstance(response, pagers.ListOSPolicyAssignmentsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_os_policy_assignments_async_from_dict():
    await test_list_os_policy_assignments_async(request_type=dict)

def test_list_os_policy_assignments_field_headers():
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = os_policy_assignments.ListOSPolicyAssignmentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_os_policy_assignments), '__call__') as call:
        call.return_value = os_policy_assignments.ListOSPolicyAssignmentsResponse()
        client.list_os_policy_assignments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_os_policy_assignments_field_headers_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = os_policy_assignments.ListOSPolicyAssignmentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_os_policy_assignments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(os_policy_assignments.ListOSPolicyAssignmentsResponse())
        await client.list_os_policy_assignments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_os_policy_assignments_flattened():
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_os_policy_assignments), '__call__') as call:
        call.return_value = os_policy_assignments.ListOSPolicyAssignmentsResponse()
        client.list_os_policy_assignments(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_os_policy_assignments_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_os_policy_assignments(os_policy_assignments.ListOSPolicyAssignmentsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_os_policy_assignments_flattened_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_os_policy_assignments), '__call__') as call:
        call.return_value = os_policy_assignments.ListOSPolicyAssignmentsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(os_policy_assignments.ListOSPolicyAssignmentsResponse())
        response = await client.list_os_policy_assignments(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_os_policy_assignments_flattened_error_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_os_policy_assignments(os_policy_assignments.ListOSPolicyAssignmentsRequest(), parent='parent_value')

def test_list_os_policy_assignments_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_os_policy_assignments), '__call__') as call:
        call.side_effect = (os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()], next_page_token='abc'), os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[], next_page_token='def'), os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment()], next_page_token='ghi'), os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_os_policy_assignments(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, os_policy_assignments.OSPolicyAssignment) for i in results))

def test_list_os_policy_assignments_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_os_policy_assignments), '__call__') as call:
        call.side_effect = (os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()], next_page_token='abc'), os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[], next_page_token='def'), os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment()], next_page_token='ghi'), os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()]), RuntimeError)
        pages = list(client.list_os_policy_assignments(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_os_policy_assignments_async_pager():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_os_policy_assignments), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()], next_page_token='abc'), os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[], next_page_token='def'), os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment()], next_page_token='ghi'), os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()]), RuntimeError)
        async_pager = await client.list_os_policy_assignments(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, os_policy_assignments.OSPolicyAssignment) for i in responses))

@pytest.mark.asyncio
async def test_list_os_policy_assignments_async_pages():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_os_policy_assignments), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()], next_page_token='abc'), os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[], next_page_token='def'), os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment()], next_page_token='ghi'), os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_os_policy_assignments(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [os_policy_assignments.ListOSPolicyAssignmentRevisionsRequest, dict])
def test_list_os_policy_assignment_revisions(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_os_policy_assignment_revisions), '__call__') as call:
        call.return_value = os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(next_page_token='next_page_token_value')
        response = client.list_os_policy_assignment_revisions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignments.ListOSPolicyAssignmentRevisionsRequest()
    assert isinstance(response, pagers.ListOSPolicyAssignmentRevisionsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_os_policy_assignment_revisions_empty_call():
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_os_policy_assignment_revisions), '__call__') as call:
        client.list_os_policy_assignment_revisions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignments.ListOSPolicyAssignmentRevisionsRequest()

@pytest.mark.asyncio
async def test_list_os_policy_assignment_revisions_async(transport: str='grpc_asyncio', request_type=os_policy_assignments.ListOSPolicyAssignmentRevisionsRequest):
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_os_policy_assignment_revisions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(next_page_token='next_page_token_value'))
        response = await client.list_os_policy_assignment_revisions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignments.ListOSPolicyAssignmentRevisionsRequest()
    assert isinstance(response, pagers.ListOSPolicyAssignmentRevisionsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_os_policy_assignment_revisions_async_from_dict():
    await test_list_os_policy_assignment_revisions_async(request_type=dict)

def test_list_os_policy_assignment_revisions_field_headers():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = os_policy_assignments.ListOSPolicyAssignmentRevisionsRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.list_os_policy_assignment_revisions), '__call__') as call:
        call.return_value = os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse()
        client.list_os_policy_assignment_revisions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_os_policy_assignment_revisions_field_headers_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = os_policy_assignments.ListOSPolicyAssignmentRevisionsRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.list_os_policy_assignment_revisions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse())
        await client.list_os_policy_assignment_revisions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_list_os_policy_assignment_revisions_flattened():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_os_policy_assignment_revisions), '__call__') as call:
        call.return_value = os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse()
        client.list_os_policy_assignment_revisions(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_list_os_policy_assignment_revisions_flattened_error():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_os_policy_assignment_revisions(os_policy_assignments.ListOSPolicyAssignmentRevisionsRequest(), name='name_value')

@pytest.mark.asyncio
async def test_list_os_policy_assignment_revisions_flattened_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_os_policy_assignment_revisions), '__call__') as call:
        call.return_value = os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse())
        response = await client.list_os_policy_assignment_revisions(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_os_policy_assignment_revisions_flattened_error_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_os_policy_assignment_revisions(os_policy_assignments.ListOSPolicyAssignmentRevisionsRequest(), name='name_value')

def test_list_os_policy_assignment_revisions_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_os_policy_assignment_revisions), '__call__') as call:
        call.side_effect = (os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()], next_page_token='abc'), os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[], next_page_token='def'), os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment()], next_page_token='ghi'), os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', ''),)),)
        pager = client.list_os_policy_assignment_revisions(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, os_policy_assignments.OSPolicyAssignment) for i in results))

def test_list_os_policy_assignment_revisions_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_os_policy_assignment_revisions), '__call__') as call:
        call.side_effect = (os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()], next_page_token='abc'), os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[], next_page_token='def'), os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment()], next_page_token='ghi'), os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()]), RuntimeError)
        pages = list(client.list_os_policy_assignment_revisions(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_os_policy_assignment_revisions_async_pager():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_os_policy_assignment_revisions), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()], next_page_token='abc'), os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[], next_page_token='def'), os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment()], next_page_token='ghi'), os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()]), RuntimeError)
        async_pager = await client.list_os_policy_assignment_revisions(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, os_policy_assignments.OSPolicyAssignment) for i in responses))

@pytest.mark.asyncio
async def test_list_os_policy_assignment_revisions_async_pages():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_os_policy_assignment_revisions), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()], next_page_token='abc'), os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[], next_page_token='def'), os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment()], next_page_token='ghi'), os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_os_policy_assignment_revisions(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [os_policy_assignments.DeleteOSPolicyAssignmentRequest, dict])
def test_delete_os_policy_assignment(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_os_policy_assignment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_os_policy_assignment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignments.DeleteOSPolicyAssignmentRequest()
    assert isinstance(response, future.Future)

def test_delete_os_policy_assignment_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_os_policy_assignment), '__call__') as call:
        client.delete_os_policy_assignment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignments.DeleteOSPolicyAssignmentRequest()

@pytest.mark.asyncio
async def test_delete_os_policy_assignment_async(transport: str='grpc_asyncio', request_type=os_policy_assignments.DeleteOSPolicyAssignmentRequest):
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_os_policy_assignment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_os_policy_assignment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignments.DeleteOSPolicyAssignmentRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_os_policy_assignment_async_from_dict():
    await test_delete_os_policy_assignment_async(request_type=dict)

def test_delete_os_policy_assignment_field_headers():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = os_policy_assignments.DeleteOSPolicyAssignmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_os_policy_assignment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_os_policy_assignment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_os_policy_assignment_field_headers_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = os_policy_assignments.DeleteOSPolicyAssignmentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_os_policy_assignment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_os_policy_assignment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_os_policy_assignment_flattened():
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_os_policy_assignment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_os_policy_assignment(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_os_policy_assignment_flattened_error():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_os_policy_assignment(os_policy_assignments.DeleteOSPolicyAssignmentRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_os_policy_assignment_flattened_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_os_policy_assignment), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_os_policy_assignment(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_os_policy_assignment_flattened_error_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_os_policy_assignment(os_policy_assignments.DeleteOSPolicyAssignmentRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [instance_os_policies_compliance.GetInstanceOSPoliciesComplianceRequest, dict])
def test_get_instance_os_policies_compliance(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_instance_os_policies_compliance), '__call__') as call:
        call.return_value = instance_os_policies_compliance.InstanceOSPoliciesCompliance(name='name_value', instance='instance_value', state=config_common.OSPolicyComplianceState.COMPLIANT, detailed_state='detailed_state_value', detailed_state_reason='detailed_state_reason_value', last_compliance_run_id='last_compliance_run_id_value')
        response = client.get_instance_os_policies_compliance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance_os_policies_compliance.GetInstanceOSPoliciesComplianceRequest()
    assert isinstance(response, instance_os_policies_compliance.InstanceOSPoliciesCompliance)
    assert response.name == 'name_value'
    assert response.instance == 'instance_value'
    assert response.state == config_common.OSPolicyComplianceState.COMPLIANT
    assert response.detailed_state == 'detailed_state_value'
    assert response.detailed_state_reason == 'detailed_state_reason_value'
    assert response.last_compliance_run_id == 'last_compliance_run_id_value'

def test_get_instance_os_policies_compliance_empty_call():
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_instance_os_policies_compliance), '__call__') as call:
        client.get_instance_os_policies_compliance()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance_os_policies_compliance.GetInstanceOSPoliciesComplianceRequest()

@pytest.mark.asyncio
async def test_get_instance_os_policies_compliance_async(transport: str='grpc_asyncio', request_type=instance_os_policies_compliance.GetInstanceOSPoliciesComplianceRequest):
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_instance_os_policies_compliance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(instance_os_policies_compliance.InstanceOSPoliciesCompliance(name='name_value', instance='instance_value', state=config_common.OSPolicyComplianceState.COMPLIANT, detailed_state='detailed_state_value', detailed_state_reason='detailed_state_reason_value', last_compliance_run_id='last_compliance_run_id_value'))
        response = await client.get_instance_os_policies_compliance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance_os_policies_compliance.GetInstanceOSPoliciesComplianceRequest()
    assert isinstance(response, instance_os_policies_compliance.InstanceOSPoliciesCompliance)
    assert response.name == 'name_value'
    assert response.instance == 'instance_value'
    assert response.state == config_common.OSPolicyComplianceState.COMPLIANT
    assert response.detailed_state == 'detailed_state_value'
    assert response.detailed_state_reason == 'detailed_state_reason_value'
    assert response.last_compliance_run_id == 'last_compliance_run_id_value'

@pytest.mark.asyncio
async def test_get_instance_os_policies_compliance_async_from_dict():
    await test_get_instance_os_policies_compliance_async(request_type=dict)

def test_get_instance_os_policies_compliance_field_headers():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance_os_policies_compliance.GetInstanceOSPoliciesComplianceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_instance_os_policies_compliance), '__call__') as call:
        call.return_value = instance_os_policies_compliance.InstanceOSPoliciesCompliance()
        client.get_instance_os_policies_compliance(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_instance_os_policies_compliance_field_headers_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance_os_policies_compliance.GetInstanceOSPoliciesComplianceRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_instance_os_policies_compliance), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(instance_os_policies_compliance.InstanceOSPoliciesCompliance())
        await client.get_instance_os_policies_compliance(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_instance_os_policies_compliance_flattened():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_instance_os_policies_compliance), '__call__') as call:
        call.return_value = instance_os_policies_compliance.InstanceOSPoliciesCompliance()
        client.get_instance_os_policies_compliance(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_instance_os_policies_compliance_flattened_error():
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_instance_os_policies_compliance(instance_os_policies_compliance.GetInstanceOSPoliciesComplianceRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_instance_os_policies_compliance_flattened_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_instance_os_policies_compliance), '__call__') as call:
        call.return_value = instance_os_policies_compliance.InstanceOSPoliciesCompliance()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(instance_os_policies_compliance.InstanceOSPoliciesCompliance())
        response = await client.get_instance_os_policies_compliance(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_instance_os_policies_compliance_flattened_error_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_instance_os_policies_compliance(instance_os_policies_compliance.GetInstanceOSPoliciesComplianceRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesRequest, dict])
def test_list_instance_os_policies_compliances(request_type, transport: str='grpc'):
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_instance_os_policies_compliances), '__call__') as call:
        call.return_value = instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(next_page_token='next_page_token_value')
        response = client.list_instance_os_policies_compliances(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesRequest()
    assert isinstance(response, pagers.ListInstanceOSPoliciesCompliancesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_instance_os_policies_compliances_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_instance_os_policies_compliances), '__call__') as call:
        client.list_instance_os_policies_compliances()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesRequest()

@pytest.mark.asyncio
async def test_list_instance_os_policies_compliances_async(transport: str='grpc_asyncio', request_type=instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesRequest):
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_instance_os_policies_compliances), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(next_page_token='next_page_token_value'))
        response = await client.list_instance_os_policies_compliances(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesRequest()
    assert isinstance(response, pagers.ListInstanceOSPoliciesCompliancesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_instance_os_policies_compliances_async_from_dict():
    await test_list_instance_os_policies_compliances_async(request_type=dict)

def test_list_instance_os_policies_compliances_field_headers():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_instance_os_policies_compliances), '__call__') as call:
        call.return_value = instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse()
        client.list_instance_os_policies_compliances(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_instance_os_policies_compliances_field_headers_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_instance_os_policies_compliances), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse())
        await client.list_instance_os_policies_compliances(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_instance_os_policies_compliances_flattened():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_instance_os_policies_compliances), '__call__') as call:
        call.return_value = instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse()
        client.list_instance_os_policies_compliances(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_instance_os_policies_compliances_flattened_error():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_instance_os_policies_compliances(instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_instance_os_policies_compliances_flattened_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_instance_os_policies_compliances), '__call__') as call:
        call.return_value = instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse())
        response = await client.list_instance_os_policies_compliances(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_instance_os_policies_compliances_flattened_error_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_instance_os_policies_compliances(instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesRequest(), parent='parent_value')

def test_list_instance_os_policies_compliances_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_instance_os_policies_compliances), '__call__') as call:
        call.side_effect = (instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[instance_os_policies_compliance.InstanceOSPoliciesCompliance(), instance_os_policies_compliance.InstanceOSPoliciesCompliance(), instance_os_policies_compliance.InstanceOSPoliciesCompliance()], next_page_token='abc'), instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[], next_page_token='def'), instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[instance_os_policies_compliance.InstanceOSPoliciesCompliance()], next_page_token='ghi'), instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[instance_os_policies_compliance.InstanceOSPoliciesCompliance(), instance_os_policies_compliance.InstanceOSPoliciesCompliance()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_instance_os_policies_compliances(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, instance_os_policies_compliance.InstanceOSPoliciesCompliance) for i in results))

def test_list_instance_os_policies_compliances_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_instance_os_policies_compliances), '__call__') as call:
        call.side_effect = (instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[instance_os_policies_compliance.InstanceOSPoliciesCompliance(), instance_os_policies_compliance.InstanceOSPoliciesCompliance(), instance_os_policies_compliance.InstanceOSPoliciesCompliance()], next_page_token='abc'), instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[], next_page_token='def'), instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[instance_os_policies_compliance.InstanceOSPoliciesCompliance()], next_page_token='ghi'), instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[instance_os_policies_compliance.InstanceOSPoliciesCompliance(), instance_os_policies_compliance.InstanceOSPoliciesCompliance()]), RuntimeError)
        pages = list(client.list_instance_os_policies_compliances(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_instance_os_policies_compliances_async_pager():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_instance_os_policies_compliances), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[instance_os_policies_compliance.InstanceOSPoliciesCompliance(), instance_os_policies_compliance.InstanceOSPoliciesCompliance(), instance_os_policies_compliance.InstanceOSPoliciesCompliance()], next_page_token='abc'), instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[], next_page_token='def'), instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[instance_os_policies_compliance.InstanceOSPoliciesCompliance()], next_page_token='ghi'), instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[instance_os_policies_compliance.InstanceOSPoliciesCompliance(), instance_os_policies_compliance.InstanceOSPoliciesCompliance()]), RuntimeError)
        async_pager = await client.list_instance_os_policies_compliances(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, instance_os_policies_compliance.InstanceOSPoliciesCompliance) for i in responses))

@pytest.mark.asyncio
async def test_list_instance_os_policies_compliances_async_pages():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_instance_os_policies_compliances), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[instance_os_policies_compliance.InstanceOSPoliciesCompliance(), instance_os_policies_compliance.InstanceOSPoliciesCompliance(), instance_os_policies_compliance.InstanceOSPoliciesCompliance()], next_page_token='abc'), instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[], next_page_token='def'), instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[instance_os_policies_compliance.InstanceOSPoliciesCompliance()], next_page_token='ghi'), instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[instance_os_policies_compliance.InstanceOSPoliciesCompliance(), instance_os_policies_compliance.InstanceOSPoliciesCompliance()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_instance_os_policies_compliances(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [os_policy_assignment_reports.GetOSPolicyAssignmentReportRequest, dict])
def test_get_os_policy_assignment_report(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_os_policy_assignment_report), '__call__') as call:
        call.return_value = os_policy_assignment_reports.OSPolicyAssignmentReport(name='name_value', instance='instance_value', os_policy_assignment='os_policy_assignment_value', last_run_id='last_run_id_value')
        response = client.get_os_policy_assignment_report(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignment_reports.GetOSPolicyAssignmentReportRequest()
    assert isinstance(response, os_policy_assignment_reports.OSPolicyAssignmentReport)
    assert response.name == 'name_value'
    assert response.instance == 'instance_value'
    assert response.os_policy_assignment == 'os_policy_assignment_value'
    assert response.last_run_id == 'last_run_id_value'

def test_get_os_policy_assignment_report_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_os_policy_assignment_report), '__call__') as call:
        client.get_os_policy_assignment_report()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignment_reports.GetOSPolicyAssignmentReportRequest()

@pytest.mark.asyncio
async def test_get_os_policy_assignment_report_async(transport: str='grpc_asyncio', request_type=os_policy_assignment_reports.GetOSPolicyAssignmentReportRequest):
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_os_policy_assignment_report), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(os_policy_assignment_reports.OSPolicyAssignmentReport(name='name_value', instance='instance_value', os_policy_assignment='os_policy_assignment_value', last_run_id='last_run_id_value'))
        response = await client.get_os_policy_assignment_report(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignment_reports.GetOSPolicyAssignmentReportRequest()
    assert isinstance(response, os_policy_assignment_reports.OSPolicyAssignmentReport)
    assert response.name == 'name_value'
    assert response.instance == 'instance_value'
    assert response.os_policy_assignment == 'os_policy_assignment_value'
    assert response.last_run_id == 'last_run_id_value'

@pytest.mark.asyncio
async def test_get_os_policy_assignment_report_async_from_dict():
    await test_get_os_policy_assignment_report_async(request_type=dict)

def test_get_os_policy_assignment_report_field_headers():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = os_policy_assignment_reports.GetOSPolicyAssignmentReportRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_os_policy_assignment_report), '__call__') as call:
        call.return_value = os_policy_assignment_reports.OSPolicyAssignmentReport()
        client.get_os_policy_assignment_report(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_os_policy_assignment_report_field_headers_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = os_policy_assignment_reports.GetOSPolicyAssignmentReportRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_os_policy_assignment_report), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(os_policy_assignment_reports.OSPolicyAssignmentReport())
        await client.get_os_policy_assignment_report(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_os_policy_assignment_report_flattened():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_os_policy_assignment_report), '__call__') as call:
        call.return_value = os_policy_assignment_reports.OSPolicyAssignmentReport()
        client.get_os_policy_assignment_report(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_os_policy_assignment_report_flattened_error():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_os_policy_assignment_report(os_policy_assignment_reports.GetOSPolicyAssignmentReportRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_os_policy_assignment_report_flattened_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_os_policy_assignment_report), '__call__') as call:
        call.return_value = os_policy_assignment_reports.OSPolicyAssignmentReport()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(os_policy_assignment_reports.OSPolicyAssignmentReport())
        response = await client.get_os_policy_assignment_report(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_os_policy_assignment_report_flattened_error_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_os_policy_assignment_report(os_policy_assignment_reports.GetOSPolicyAssignmentReportRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [os_policy_assignment_reports.ListOSPolicyAssignmentReportsRequest, dict])
def test_list_os_policy_assignment_reports(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_os_policy_assignment_reports), '__call__') as call:
        call.return_value = os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(next_page_token='next_page_token_value')
        response = client.list_os_policy_assignment_reports(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignment_reports.ListOSPolicyAssignmentReportsRequest()
    assert isinstance(response, pagers.ListOSPolicyAssignmentReportsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_os_policy_assignment_reports_empty_call():
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_os_policy_assignment_reports), '__call__') as call:
        client.list_os_policy_assignment_reports()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignment_reports.ListOSPolicyAssignmentReportsRequest()

@pytest.mark.asyncio
async def test_list_os_policy_assignment_reports_async(transport: str='grpc_asyncio', request_type=os_policy_assignment_reports.ListOSPolicyAssignmentReportsRequest):
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_os_policy_assignment_reports), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(next_page_token='next_page_token_value'))
        response = await client.list_os_policy_assignment_reports(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == os_policy_assignment_reports.ListOSPolicyAssignmentReportsRequest()
    assert isinstance(response, pagers.ListOSPolicyAssignmentReportsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_os_policy_assignment_reports_async_from_dict():
    await test_list_os_policy_assignment_reports_async(request_type=dict)

def test_list_os_policy_assignment_reports_field_headers():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = os_policy_assignment_reports.ListOSPolicyAssignmentReportsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_os_policy_assignment_reports), '__call__') as call:
        call.return_value = os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse()
        client.list_os_policy_assignment_reports(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_os_policy_assignment_reports_field_headers_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = os_policy_assignment_reports.ListOSPolicyAssignmentReportsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_os_policy_assignment_reports), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse())
        await client.list_os_policy_assignment_reports(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_os_policy_assignment_reports_flattened():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_os_policy_assignment_reports), '__call__') as call:
        call.return_value = os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse()
        client.list_os_policy_assignment_reports(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_os_policy_assignment_reports_flattened_error():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_os_policy_assignment_reports(os_policy_assignment_reports.ListOSPolicyAssignmentReportsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_os_policy_assignment_reports_flattened_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_os_policy_assignment_reports), '__call__') as call:
        call.return_value = os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse())
        response = await client.list_os_policy_assignment_reports(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_os_policy_assignment_reports_flattened_error_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_os_policy_assignment_reports(os_policy_assignment_reports.ListOSPolicyAssignmentReportsRequest(), parent='parent_value')

def test_list_os_policy_assignment_reports_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_os_policy_assignment_reports), '__call__') as call:
        call.side_effect = (os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[os_policy_assignment_reports.OSPolicyAssignmentReport(), os_policy_assignment_reports.OSPolicyAssignmentReport(), os_policy_assignment_reports.OSPolicyAssignmentReport()], next_page_token='abc'), os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[], next_page_token='def'), os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[os_policy_assignment_reports.OSPolicyAssignmentReport()], next_page_token='ghi'), os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[os_policy_assignment_reports.OSPolicyAssignmentReport(), os_policy_assignment_reports.OSPolicyAssignmentReport()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_os_policy_assignment_reports(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, os_policy_assignment_reports.OSPolicyAssignmentReport) for i in results))

def test_list_os_policy_assignment_reports_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_os_policy_assignment_reports), '__call__') as call:
        call.side_effect = (os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[os_policy_assignment_reports.OSPolicyAssignmentReport(), os_policy_assignment_reports.OSPolicyAssignmentReport(), os_policy_assignment_reports.OSPolicyAssignmentReport()], next_page_token='abc'), os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[], next_page_token='def'), os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[os_policy_assignment_reports.OSPolicyAssignmentReport()], next_page_token='ghi'), os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[os_policy_assignment_reports.OSPolicyAssignmentReport(), os_policy_assignment_reports.OSPolicyAssignmentReport()]), RuntimeError)
        pages = list(client.list_os_policy_assignment_reports(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_os_policy_assignment_reports_async_pager():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_os_policy_assignment_reports), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[os_policy_assignment_reports.OSPolicyAssignmentReport(), os_policy_assignment_reports.OSPolicyAssignmentReport(), os_policy_assignment_reports.OSPolicyAssignmentReport()], next_page_token='abc'), os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[], next_page_token='def'), os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[os_policy_assignment_reports.OSPolicyAssignmentReport()], next_page_token='ghi'), os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[os_policy_assignment_reports.OSPolicyAssignmentReport(), os_policy_assignment_reports.OSPolicyAssignmentReport()]), RuntimeError)
        async_pager = await client.list_os_policy_assignment_reports(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, os_policy_assignment_reports.OSPolicyAssignmentReport) for i in responses))

@pytest.mark.asyncio
async def test_list_os_policy_assignment_reports_async_pages():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_os_policy_assignment_reports), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[os_policy_assignment_reports.OSPolicyAssignmentReport(), os_policy_assignment_reports.OSPolicyAssignmentReport(), os_policy_assignment_reports.OSPolicyAssignmentReport()], next_page_token='abc'), os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[], next_page_token='def'), os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[os_policy_assignment_reports.OSPolicyAssignmentReport()], next_page_token='ghi'), os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[os_policy_assignment_reports.OSPolicyAssignmentReport(), os_policy_assignment_reports.OSPolicyAssignmentReport()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_os_policy_assignment_reports(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [inventory.GetInventoryRequest, dict])
def test_get_inventory(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_inventory), '__call__') as call:
        call.return_value = inventory.Inventory(name='name_value')
        response = client.get_inventory(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == inventory.GetInventoryRequest()
    assert isinstance(response, inventory.Inventory)
    assert response.name == 'name_value'

def test_get_inventory_empty_call():
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_inventory), '__call__') as call:
        client.get_inventory()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == inventory.GetInventoryRequest()

@pytest.mark.asyncio
async def test_get_inventory_async(transport: str='grpc_asyncio', request_type=inventory.GetInventoryRequest):
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_inventory), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(inventory.Inventory(name='name_value'))
        response = await client.get_inventory(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == inventory.GetInventoryRequest()
    assert isinstance(response, inventory.Inventory)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_inventory_async_from_dict():
    await test_get_inventory_async(request_type=dict)

def test_get_inventory_field_headers():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = inventory.GetInventoryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_inventory), '__call__') as call:
        call.return_value = inventory.Inventory()
        client.get_inventory(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_inventory_field_headers_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = inventory.GetInventoryRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_inventory), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(inventory.Inventory())
        await client.get_inventory(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_inventory_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_inventory), '__call__') as call:
        call.return_value = inventory.Inventory()
        client.get_inventory(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_inventory_flattened_error():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_inventory(inventory.GetInventoryRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_inventory_flattened_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_inventory), '__call__') as call:
        call.return_value = inventory.Inventory()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(inventory.Inventory())
        response = await client.get_inventory(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_inventory_flattened_error_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_inventory(inventory.GetInventoryRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [inventory.ListInventoriesRequest, dict])
def test_list_inventories(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_inventories), '__call__') as call:
        call.return_value = inventory.ListInventoriesResponse(next_page_token='next_page_token_value')
        response = client.list_inventories(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == inventory.ListInventoriesRequest()
    assert isinstance(response, pagers.ListInventoriesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_inventories_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_inventories), '__call__') as call:
        client.list_inventories()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == inventory.ListInventoriesRequest()

@pytest.mark.asyncio
async def test_list_inventories_async(transport: str='grpc_asyncio', request_type=inventory.ListInventoriesRequest):
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_inventories), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(inventory.ListInventoriesResponse(next_page_token='next_page_token_value'))
        response = await client.list_inventories(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == inventory.ListInventoriesRequest()
    assert isinstance(response, pagers.ListInventoriesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_inventories_async_from_dict():
    await test_list_inventories_async(request_type=dict)

def test_list_inventories_field_headers():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = inventory.ListInventoriesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_inventories), '__call__') as call:
        call.return_value = inventory.ListInventoriesResponse()
        client.list_inventories(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_inventories_field_headers_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = inventory.ListInventoriesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_inventories), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(inventory.ListInventoriesResponse())
        await client.list_inventories(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_inventories_flattened():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_inventories), '__call__') as call:
        call.return_value = inventory.ListInventoriesResponse()
        client.list_inventories(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_inventories_flattened_error():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_inventories(inventory.ListInventoriesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_inventories_flattened_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_inventories), '__call__') as call:
        call.return_value = inventory.ListInventoriesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(inventory.ListInventoriesResponse())
        response = await client.list_inventories(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_inventories_flattened_error_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_inventories(inventory.ListInventoriesRequest(), parent='parent_value')

def test_list_inventories_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_inventories), '__call__') as call:
        call.side_effect = (inventory.ListInventoriesResponse(inventories=[inventory.Inventory(), inventory.Inventory(), inventory.Inventory()], next_page_token='abc'), inventory.ListInventoriesResponse(inventories=[], next_page_token='def'), inventory.ListInventoriesResponse(inventories=[inventory.Inventory()], next_page_token='ghi'), inventory.ListInventoriesResponse(inventories=[inventory.Inventory(), inventory.Inventory()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_inventories(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, inventory.Inventory) for i in results))

def test_list_inventories_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_inventories), '__call__') as call:
        call.side_effect = (inventory.ListInventoriesResponse(inventories=[inventory.Inventory(), inventory.Inventory(), inventory.Inventory()], next_page_token='abc'), inventory.ListInventoriesResponse(inventories=[], next_page_token='def'), inventory.ListInventoriesResponse(inventories=[inventory.Inventory()], next_page_token='ghi'), inventory.ListInventoriesResponse(inventories=[inventory.Inventory(), inventory.Inventory()]), RuntimeError)
        pages = list(client.list_inventories(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_inventories_async_pager():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_inventories), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (inventory.ListInventoriesResponse(inventories=[inventory.Inventory(), inventory.Inventory(), inventory.Inventory()], next_page_token='abc'), inventory.ListInventoriesResponse(inventories=[], next_page_token='def'), inventory.ListInventoriesResponse(inventories=[inventory.Inventory()], next_page_token='ghi'), inventory.ListInventoriesResponse(inventories=[inventory.Inventory(), inventory.Inventory()]), RuntimeError)
        async_pager = await client.list_inventories(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, inventory.Inventory) for i in responses))

@pytest.mark.asyncio
async def test_list_inventories_async_pages():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_inventories), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (inventory.ListInventoriesResponse(inventories=[inventory.Inventory(), inventory.Inventory(), inventory.Inventory()], next_page_token='abc'), inventory.ListInventoriesResponse(inventories=[], next_page_token='def'), inventory.ListInventoriesResponse(inventories=[inventory.Inventory()], next_page_token='ghi'), inventory.ListInventoriesResponse(inventories=[inventory.Inventory(), inventory.Inventory()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_inventories(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vulnerability.GetVulnerabilityReportRequest, dict])
def test_get_vulnerability_report(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_vulnerability_report), '__call__') as call:
        call.return_value = vulnerability.VulnerabilityReport(name='name_value')
        response = client.get_vulnerability_report(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vulnerability.GetVulnerabilityReportRequest()
    assert isinstance(response, vulnerability.VulnerabilityReport)
    assert response.name == 'name_value'

def test_get_vulnerability_report_empty_call():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_vulnerability_report), '__call__') as call:
        client.get_vulnerability_report()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vulnerability.GetVulnerabilityReportRequest()

@pytest.mark.asyncio
async def test_get_vulnerability_report_async(transport: str='grpc_asyncio', request_type=vulnerability.GetVulnerabilityReportRequest):
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_vulnerability_report), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vulnerability.VulnerabilityReport(name='name_value'))
        response = await client.get_vulnerability_report(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vulnerability.GetVulnerabilityReportRequest()
    assert isinstance(response, vulnerability.VulnerabilityReport)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_vulnerability_report_async_from_dict():
    await test_get_vulnerability_report_async(request_type=dict)

def test_get_vulnerability_report_field_headers():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = vulnerability.GetVulnerabilityReportRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_vulnerability_report), '__call__') as call:
        call.return_value = vulnerability.VulnerabilityReport()
        client.get_vulnerability_report(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_vulnerability_report_field_headers_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vulnerability.GetVulnerabilityReportRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_vulnerability_report), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vulnerability.VulnerabilityReport())
        await client.get_vulnerability_report(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_vulnerability_report_flattened():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_vulnerability_report), '__call__') as call:
        call.return_value = vulnerability.VulnerabilityReport()
        client.get_vulnerability_report(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_vulnerability_report_flattened_error():
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_vulnerability_report(vulnerability.GetVulnerabilityReportRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_vulnerability_report_flattened_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_vulnerability_report), '__call__') as call:
        call.return_value = vulnerability.VulnerabilityReport()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vulnerability.VulnerabilityReport())
        response = await client.get_vulnerability_report(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_vulnerability_report_flattened_error_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_vulnerability_report(vulnerability.GetVulnerabilityReportRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [vulnerability.ListVulnerabilityReportsRequest, dict])
def test_list_vulnerability_reports(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_vulnerability_reports), '__call__') as call:
        call.return_value = vulnerability.ListVulnerabilityReportsResponse(next_page_token='next_page_token_value')
        response = client.list_vulnerability_reports(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vulnerability.ListVulnerabilityReportsRequest()
    assert isinstance(response, pagers.ListVulnerabilityReportsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_vulnerability_reports_empty_call():
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_vulnerability_reports), '__call__') as call:
        client.list_vulnerability_reports()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vulnerability.ListVulnerabilityReportsRequest()

@pytest.mark.asyncio
async def test_list_vulnerability_reports_async(transport: str='grpc_asyncio', request_type=vulnerability.ListVulnerabilityReportsRequest):
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_vulnerability_reports), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vulnerability.ListVulnerabilityReportsResponse(next_page_token='next_page_token_value'))
        response = await client.list_vulnerability_reports(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == vulnerability.ListVulnerabilityReportsRequest()
    assert isinstance(response, pagers.ListVulnerabilityReportsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_vulnerability_reports_async_from_dict():
    await test_list_vulnerability_reports_async(request_type=dict)

def test_list_vulnerability_reports_field_headers():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = vulnerability.ListVulnerabilityReportsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_vulnerability_reports), '__call__') as call:
        call.return_value = vulnerability.ListVulnerabilityReportsResponse()
        client.list_vulnerability_reports(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_vulnerability_reports_field_headers_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = vulnerability.ListVulnerabilityReportsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_vulnerability_reports), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vulnerability.ListVulnerabilityReportsResponse())
        await client.list_vulnerability_reports(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_vulnerability_reports_flattened():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_vulnerability_reports), '__call__') as call:
        call.return_value = vulnerability.ListVulnerabilityReportsResponse()
        client.list_vulnerability_reports(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_vulnerability_reports_flattened_error():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_vulnerability_reports(vulnerability.ListVulnerabilityReportsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_vulnerability_reports_flattened_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_vulnerability_reports), '__call__') as call:
        call.return_value = vulnerability.ListVulnerabilityReportsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(vulnerability.ListVulnerabilityReportsResponse())
        response = await client.list_vulnerability_reports(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_vulnerability_reports_flattened_error_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_vulnerability_reports(vulnerability.ListVulnerabilityReportsRequest(), parent='parent_value')

def test_list_vulnerability_reports_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_vulnerability_reports), '__call__') as call:
        call.side_effect = (vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[vulnerability.VulnerabilityReport(), vulnerability.VulnerabilityReport(), vulnerability.VulnerabilityReport()], next_page_token='abc'), vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[], next_page_token='def'), vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[vulnerability.VulnerabilityReport()], next_page_token='ghi'), vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[vulnerability.VulnerabilityReport(), vulnerability.VulnerabilityReport()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_vulnerability_reports(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vulnerability.VulnerabilityReport) for i in results))

def test_list_vulnerability_reports_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_vulnerability_reports), '__call__') as call:
        call.side_effect = (vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[vulnerability.VulnerabilityReport(), vulnerability.VulnerabilityReport(), vulnerability.VulnerabilityReport()], next_page_token='abc'), vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[], next_page_token='def'), vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[vulnerability.VulnerabilityReport()], next_page_token='ghi'), vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[vulnerability.VulnerabilityReport(), vulnerability.VulnerabilityReport()]), RuntimeError)
        pages = list(client.list_vulnerability_reports(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_vulnerability_reports_async_pager():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_vulnerability_reports), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[vulnerability.VulnerabilityReport(), vulnerability.VulnerabilityReport(), vulnerability.VulnerabilityReport()], next_page_token='abc'), vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[], next_page_token='def'), vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[vulnerability.VulnerabilityReport()], next_page_token='ghi'), vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[vulnerability.VulnerabilityReport(), vulnerability.VulnerabilityReport()]), RuntimeError)
        async_pager = await client.list_vulnerability_reports(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, vulnerability.VulnerabilityReport) for i in responses))

@pytest.mark.asyncio
async def test_list_vulnerability_reports_async_pages():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_vulnerability_reports), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[vulnerability.VulnerabilityReport(), vulnerability.VulnerabilityReport(), vulnerability.VulnerabilityReport()], next_page_token='abc'), vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[], next_page_token='def'), vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[vulnerability.VulnerabilityReport()], next_page_token='ghi'), vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[vulnerability.VulnerabilityReport(), vulnerability.VulnerabilityReport()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_vulnerability_reports(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [os_policy_assignments.CreateOSPolicyAssignmentRequest, dict])
def test_create_os_policy_assignment_rest(request_type):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['os_policy_assignment'] = {'name': 'name_value', 'description': 'description_value', 'os_policies': [{'id': 'id_value', 'description': 'description_value', 'mode': 1, 'resource_groups': [{'os_filter': {'os_short_name': 'os_short_name_value', 'os_version': 'os_version_value'}, 'inventory_filters': [{'os_short_name': 'os_short_name_value', 'os_version': 'os_version_value'}], 'resources': [{'id': 'id_value', 'pkg': {'desired_state': 1, 'apt': {'name': 'name_value'}, 'deb': {'source': {'remote': {'uri': 'uri_value', 'sha256_checksum': 'sha256_checksum_value'}, 'gcs': {'bucket': 'bucket_value', 'object_': 'object__value', 'generation': 1068}, 'local_path': 'local_path_value', 'allow_insecure': True}, 'pull_deps': True}, 'yum': {'name': 'name_value'}, 'zypper': {'name': 'name_value'}, 'rpm': {'source': {}, 'pull_deps': True}, 'googet': {'name': 'name_value'}, 'msi': {'source': {}, 'properties': ['properties_value1', 'properties_value2']}}, 'repository': {'apt': {'archive_type': 1, 'uri': 'uri_value', 'distribution': 'distribution_value', 'components': ['components_value1', 'components_value2'], 'gpg_key': 'gpg_key_value'}, 'yum': {'id': 'id_value', 'display_name': 'display_name_value', 'base_url': 'base_url_value', 'gpg_keys': ['gpg_keys_value1', 'gpg_keys_value2']}, 'zypper': {'id': 'id_value', 'display_name': 'display_name_value', 'base_url': 'base_url_value', 'gpg_keys': ['gpg_keys_value1', 'gpg_keys_value2']}, 'goo': {'name': 'name_value', 'url': 'url_value'}}, 'exec_': {'validate': {'file': {}, 'script': 'script_value', 'args': ['args_value1', 'args_value2'], 'interpreter': 1, 'output_file_path': 'output_file_path_value'}, 'enforce': {}}, 'file': {'file': {}, 'content': 'content_value', 'path': 'path_value', 'state': 1, 'permissions': 'permissions_value'}}]}], 'allow_no_resource_group_match': True}], 'instance_filter': {'all_': True, 'os_short_names': ['os_short_names_value1', 'os_short_names_value2'], 'inclusion_labels': [{'labels': {}}], 'exclusion_labels': {}, 'inventories': [{'os_short_name': 'os_short_name_value', 'os_version': 'os_version_value'}]}, 'rollout': {'disruption_budget': {'fixed': 528, 'percent': 753}, 'min_wait_duration': {'seconds': 751, 'nanos': 543}}, 'revision_id': 'revision_id_value', 'revision_create_time': {'seconds': 751, 'nanos': 543}, 'etag': 'etag_value', 'rollout_state': 1, 'baseline': True, 'deleted': True, 'reconciling': True, 'uid': 'uid_value'}
    test_field = os_policy_assignments.CreateOSPolicyAssignmentRequest.meta.fields['os_policy_assignment']

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
    for (field, value) in request_init['os_policy_assignment'].items():
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
                for i in range(0, len(request_init['os_policy_assignment'][field])):
                    del request_init['os_policy_assignment'][field][i][subfield]
            else:
                del request_init['os_policy_assignment'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_os_policy_assignment(request)
    assert response.operation.name == 'operations/spam'

def test_create_os_policy_assignment_rest_required_fields(request_type=os_policy_assignments.CreateOSPolicyAssignmentRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.OsConfigZonalServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['os_policy_assignment_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'osPolicyAssignmentId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_os_policy_assignment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'osPolicyAssignmentId' in jsonified_request
    assert jsonified_request['osPolicyAssignmentId'] == request_init['os_policy_assignment_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['osPolicyAssignmentId'] = 'os_policy_assignment_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_os_policy_assignment._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('os_policy_assignment_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'osPolicyAssignmentId' in jsonified_request
    assert jsonified_request['osPolicyAssignmentId'] == 'os_policy_assignment_id_value'
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_os_policy_assignment(request)
            expected_params = [('osPolicyAssignmentId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_os_policy_assignment_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_os_policy_assignment._get_unset_required_fields({})
    assert set(unset_fields) == set(('osPolicyAssignmentId',)) & set(('parent', 'osPolicyAssignment', 'osPolicyAssignmentId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_os_policy_assignment_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigZonalServiceRestInterceptor())
    client = OsConfigZonalServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'post_create_os_policy_assignment') as post, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'pre_create_os_policy_assignment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = os_policy_assignments.CreateOSPolicyAssignmentRequest.pb(os_policy_assignments.CreateOSPolicyAssignmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = os_policy_assignments.CreateOSPolicyAssignmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_os_policy_assignment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_os_policy_assignment_rest_bad_request(transport: str='rest', request_type=os_policy_assignments.CreateOSPolicyAssignmentRequest):
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_os_policy_assignment(request)

def test_create_os_policy_assignment_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', os_policy_assignment=os_policy_assignments.OSPolicyAssignment(name='name_value'), os_policy_assignment_id='os_policy_assignment_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_os_policy_assignment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{parent=projects/*/locations/*}/osPolicyAssignments' % client.transport._host, args[1])

def test_create_os_policy_assignment_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_os_policy_assignment(os_policy_assignments.CreateOSPolicyAssignmentRequest(), parent='parent_value', os_policy_assignment=os_policy_assignments.OSPolicyAssignment(name='name_value'), os_policy_assignment_id='os_policy_assignment_id_value')

def test_create_os_policy_assignment_rest_error():
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [os_policy_assignments.UpdateOSPolicyAssignmentRequest, dict])
def test_update_os_policy_assignment_rest(request_type):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'os_policy_assignment': {'name': 'projects/sample1/locations/sample2/osPolicyAssignments/sample3'}}
    request_init['os_policy_assignment'] = {'name': 'projects/sample1/locations/sample2/osPolicyAssignments/sample3', 'description': 'description_value', 'os_policies': [{'id': 'id_value', 'description': 'description_value', 'mode': 1, 'resource_groups': [{'os_filter': {'os_short_name': 'os_short_name_value', 'os_version': 'os_version_value'}, 'inventory_filters': [{'os_short_name': 'os_short_name_value', 'os_version': 'os_version_value'}], 'resources': [{'id': 'id_value', 'pkg': {'desired_state': 1, 'apt': {'name': 'name_value'}, 'deb': {'source': {'remote': {'uri': 'uri_value', 'sha256_checksum': 'sha256_checksum_value'}, 'gcs': {'bucket': 'bucket_value', 'object_': 'object__value', 'generation': 1068}, 'local_path': 'local_path_value', 'allow_insecure': True}, 'pull_deps': True}, 'yum': {'name': 'name_value'}, 'zypper': {'name': 'name_value'}, 'rpm': {'source': {}, 'pull_deps': True}, 'googet': {'name': 'name_value'}, 'msi': {'source': {}, 'properties': ['properties_value1', 'properties_value2']}}, 'repository': {'apt': {'archive_type': 1, 'uri': 'uri_value', 'distribution': 'distribution_value', 'components': ['components_value1', 'components_value2'], 'gpg_key': 'gpg_key_value'}, 'yum': {'id': 'id_value', 'display_name': 'display_name_value', 'base_url': 'base_url_value', 'gpg_keys': ['gpg_keys_value1', 'gpg_keys_value2']}, 'zypper': {'id': 'id_value', 'display_name': 'display_name_value', 'base_url': 'base_url_value', 'gpg_keys': ['gpg_keys_value1', 'gpg_keys_value2']}, 'goo': {'name': 'name_value', 'url': 'url_value'}}, 'exec_': {'validate': {'file': {}, 'script': 'script_value', 'args': ['args_value1', 'args_value2'], 'interpreter': 1, 'output_file_path': 'output_file_path_value'}, 'enforce': {}}, 'file': {'file': {}, 'content': 'content_value', 'path': 'path_value', 'state': 1, 'permissions': 'permissions_value'}}]}], 'allow_no_resource_group_match': True}], 'instance_filter': {'all_': True, 'os_short_names': ['os_short_names_value1', 'os_short_names_value2'], 'inclusion_labels': [{'labels': {}}], 'exclusion_labels': {}, 'inventories': [{'os_short_name': 'os_short_name_value', 'os_version': 'os_version_value'}]}, 'rollout': {'disruption_budget': {'fixed': 528, 'percent': 753}, 'min_wait_duration': {'seconds': 751, 'nanos': 543}}, 'revision_id': 'revision_id_value', 'revision_create_time': {'seconds': 751, 'nanos': 543}, 'etag': 'etag_value', 'rollout_state': 1, 'baseline': True, 'deleted': True, 'reconciling': True, 'uid': 'uid_value'}
    test_field = os_policy_assignments.UpdateOSPolicyAssignmentRequest.meta.fields['os_policy_assignment']

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
    for (field, value) in request_init['os_policy_assignment'].items():
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
                for i in range(0, len(request_init['os_policy_assignment'][field])):
                    del request_init['os_policy_assignment'][field][i][subfield]
            else:
                del request_init['os_policy_assignment'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_os_policy_assignment(request)
    assert response.operation.name == 'operations/spam'

def test_update_os_policy_assignment_rest_required_fields(request_type=os_policy_assignments.UpdateOSPolicyAssignmentRequest):
    if False:
        print('Hello World!')
    transport_class = transports.OsConfigZonalServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_os_policy_assignment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_os_policy_assignment._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_os_policy_assignment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_os_policy_assignment_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_os_policy_assignment._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('osPolicyAssignment',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_os_policy_assignment_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigZonalServiceRestInterceptor())
    client = OsConfigZonalServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'post_update_os_policy_assignment') as post, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'pre_update_os_policy_assignment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = os_policy_assignments.UpdateOSPolicyAssignmentRequest.pb(os_policy_assignments.UpdateOSPolicyAssignmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = os_policy_assignments.UpdateOSPolicyAssignmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_os_policy_assignment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_os_policy_assignment_rest_bad_request(transport: str='rest', request_type=os_policy_assignments.UpdateOSPolicyAssignmentRequest):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'os_policy_assignment': {'name': 'projects/sample1/locations/sample2/osPolicyAssignments/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_os_policy_assignment(request)

def test_update_os_policy_assignment_rest_flattened():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'os_policy_assignment': {'name': 'projects/sample1/locations/sample2/osPolicyAssignments/sample3'}}
        mock_args = dict(os_policy_assignment=os_policy_assignments.OSPolicyAssignment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_os_policy_assignment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{os_policy_assignment.name=projects/*/locations/*/osPolicyAssignments/*}' % client.transport._host, args[1])

def test_update_os_policy_assignment_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_os_policy_assignment(os_policy_assignments.UpdateOSPolicyAssignmentRequest(), os_policy_assignment=os_policy_assignments.OSPolicyAssignment(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_os_policy_assignment_rest_error():
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [os_policy_assignments.GetOSPolicyAssignmentRequest, dict])
def test_get_os_policy_assignment_rest(request_type):
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/osPolicyAssignments/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = os_policy_assignments.OSPolicyAssignment(name='name_value', description='description_value', revision_id='revision_id_value', etag='etag_value', rollout_state=os_policy_assignments.OSPolicyAssignment.RolloutState.IN_PROGRESS, baseline=True, deleted=True, reconciling=True, uid='uid_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = os_policy_assignments.OSPolicyAssignment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_os_policy_assignment(request)
    assert isinstance(response, os_policy_assignments.OSPolicyAssignment)
    assert response.name == 'name_value'
    assert response.description == 'description_value'
    assert response.revision_id == 'revision_id_value'
    assert response.etag == 'etag_value'
    assert response.rollout_state == os_policy_assignments.OSPolicyAssignment.RolloutState.IN_PROGRESS
    assert response.baseline is True
    assert response.deleted is True
    assert response.reconciling is True
    assert response.uid == 'uid_value'

def test_get_os_policy_assignment_rest_required_fields(request_type=os_policy_assignments.GetOSPolicyAssignmentRequest):
    if False:
        return 10
    transport_class = transports.OsConfigZonalServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_os_policy_assignment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_os_policy_assignment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = os_policy_assignments.OSPolicyAssignment()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = os_policy_assignments.OSPolicyAssignment.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_os_policy_assignment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_os_policy_assignment_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_os_policy_assignment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_os_policy_assignment_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigZonalServiceRestInterceptor())
    client = OsConfigZonalServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'post_get_os_policy_assignment') as post, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'pre_get_os_policy_assignment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = os_policy_assignments.GetOSPolicyAssignmentRequest.pb(os_policy_assignments.GetOSPolicyAssignmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = os_policy_assignments.OSPolicyAssignment.to_json(os_policy_assignments.OSPolicyAssignment())
        request = os_policy_assignments.GetOSPolicyAssignmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = os_policy_assignments.OSPolicyAssignment()
        client.get_os_policy_assignment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_os_policy_assignment_rest_bad_request(transport: str='rest', request_type=os_policy_assignments.GetOSPolicyAssignmentRequest):
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/osPolicyAssignments/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_os_policy_assignment(request)

def test_get_os_policy_assignment_rest_flattened():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = os_policy_assignments.OSPolicyAssignment()
        sample_request = {'name': 'projects/sample1/locations/sample2/osPolicyAssignments/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = os_policy_assignments.OSPolicyAssignment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_os_policy_assignment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=projects/*/locations/*/osPolicyAssignments/*}' % client.transport._host, args[1])

def test_get_os_policy_assignment_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_os_policy_assignment(os_policy_assignments.GetOSPolicyAssignmentRequest(), name='name_value')

def test_get_os_policy_assignment_rest_error():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [os_policy_assignments.ListOSPolicyAssignmentsRequest, dict])
def test_list_os_policy_assignments_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = os_policy_assignments.ListOSPolicyAssignmentsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = os_policy_assignments.ListOSPolicyAssignmentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_os_policy_assignments(request)
    assert isinstance(response, pagers.ListOSPolicyAssignmentsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_os_policy_assignments_rest_required_fields(request_type=os_policy_assignments.ListOSPolicyAssignmentsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.OsConfigZonalServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_os_policy_assignments._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_os_policy_assignments._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = os_policy_assignments.ListOSPolicyAssignmentsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = os_policy_assignments.ListOSPolicyAssignmentsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_os_policy_assignments(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_os_policy_assignments_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_os_policy_assignments._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_os_policy_assignments_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigZonalServiceRestInterceptor())
    client = OsConfigZonalServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'post_list_os_policy_assignments') as post, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'pre_list_os_policy_assignments') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = os_policy_assignments.ListOSPolicyAssignmentsRequest.pb(os_policy_assignments.ListOSPolicyAssignmentsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = os_policy_assignments.ListOSPolicyAssignmentsResponse.to_json(os_policy_assignments.ListOSPolicyAssignmentsResponse())
        request = os_policy_assignments.ListOSPolicyAssignmentsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = os_policy_assignments.ListOSPolicyAssignmentsResponse()
        client.list_os_policy_assignments(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_os_policy_assignments_rest_bad_request(transport: str='rest', request_type=os_policy_assignments.ListOSPolicyAssignmentsRequest):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_os_policy_assignments(request)

def test_list_os_policy_assignments_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = os_policy_assignments.ListOSPolicyAssignmentsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = os_policy_assignments.ListOSPolicyAssignmentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_os_policy_assignments(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{parent=projects/*/locations/*}/osPolicyAssignments' % client.transport._host, args[1])

def test_list_os_policy_assignments_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_os_policy_assignments(os_policy_assignments.ListOSPolicyAssignmentsRequest(), parent='parent_value')

def test_list_os_policy_assignments_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()], next_page_token='abc'), os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[], next_page_token='def'), os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment()], next_page_token='ghi'), os_policy_assignments.ListOSPolicyAssignmentsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()]))
        response = response + response
        response = tuple((os_policy_assignments.ListOSPolicyAssignmentsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_os_policy_assignments(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, os_policy_assignments.OSPolicyAssignment) for i in results))
        pages = list(client.list_os_policy_assignments(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [os_policy_assignments.ListOSPolicyAssignmentRevisionsRequest, dict])
def test_list_os_policy_assignment_revisions_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/osPolicyAssignments/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_os_policy_assignment_revisions(request)
    assert isinstance(response, pagers.ListOSPolicyAssignmentRevisionsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_os_policy_assignment_revisions_rest_required_fields(request_type=os_policy_assignments.ListOSPolicyAssignmentRevisionsRequest):
    if False:
        return 10
    transport_class = transports.OsConfigZonalServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_os_policy_assignment_revisions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_os_policy_assignment_revisions._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_os_policy_assignment_revisions(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_os_policy_assignment_revisions_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_os_policy_assignment_revisions._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_os_policy_assignment_revisions_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigZonalServiceRestInterceptor())
    client = OsConfigZonalServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'post_list_os_policy_assignment_revisions') as post, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'pre_list_os_policy_assignment_revisions') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = os_policy_assignments.ListOSPolicyAssignmentRevisionsRequest.pb(os_policy_assignments.ListOSPolicyAssignmentRevisionsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse.to_json(os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse())
        request = os_policy_assignments.ListOSPolicyAssignmentRevisionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse()
        client.list_os_policy_assignment_revisions(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_os_policy_assignment_revisions_rest_bad_request(transport: str='rest', request_type=os_policy_assignments.ListOSPolicyAssignmentRevisionsRequest):
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/osPolicyAssignments/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_os_policy_assignment_revisions(request)

def test_list_os_policy_assignment_revisions_rest_flattened():
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse()
        sample_request = {'name': 'projects/sample1/locations/sample2/osPolicyAssignments/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_os_policy_assignment_revisions(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=projects/*/locations/*/osPolicyAssignments/*}:listRevisions' % client.transport._host, args[1])

def test_list_os_policy_assignment_revisions_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_os_policy_assignment_revisions(os_policy_assignments.ListOSPolicyAssignmentRevisionsRequest(), name='name_value')

def test_list_os_policy_assignment_revisions_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()], next_page_token='abc'), os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[], next_page_token='def'), os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment()], next_page_token='ghi'), os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse(os_policy_assignments=[os_policy_assignments.OSPolicyAssignment(), os_policy_assignments.OSPolicyAssignment()]))
        response = response + response
        response = tuple((os_policy_assignments.ListOSPolicyAssignmentRevisionsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'name': 'projects/sample1/locations/sample2/osPolicyAssignments/sample3'}
        pager = client.list_os_policy_assignment_revisions(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, os_policy_assignments.OSPolicyAssignment) for i in results))
        pages = list(client.list_os_policy_assignment_revisions(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [os_policy_assignments.DeleteOSPolicyAssignmentRequest, dict])
def test_delete_os_policy_assignment_rest(request_type):
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/osPolicyAssignments/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_os_policy_assignment(request)
    assert response.operation.name == 'operations/spam'

def test_delete_os_policy_assignment_rest_required_fields(request_type=os_policy_assignments.DeleteOSPolicyAssignmentRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.OsConfigZonalServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_os_policy_assignment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_os_policy_assignment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_os_policy_assignment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_os_policy_assignment_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_os_policy_assignment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_os_policy_assignment_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigZonalServiceRestInterceptor())
    client = OsConfigZonalServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'post_delete_os_policy_assignment') as post, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'pre_delete_os_policy_assignment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = os_policy_assignments.DeleteOSPolicyAssignmentRequest.pb(os_policy_assignments.DeleteOSPolicyAssignmentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = os_policy_assignments.DeleteOSPolicyAssignmentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_os_policy_assignment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_os_policy_assignment_rest_bad_request(transport: str='rest', request_type=os_policy_assignments.DeleteOSPolicyAssignmentRequest):
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/osPolicyAssignments/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_os_policy_assignment(request)

def test_delete_os_policy_assignment_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/osPolicyAssignments/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_os_policy_assignment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=projects/*/locations/*/osPolicyAssignments/*}' % client.transport._host, args[1])

def test_delete_os_policy_assignment_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_os_policy_assignment(os_policy_assignments.DeleteOSPolicyAssignmentRequest(), name='name_value')

def test_delete_os_policy_assignment_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [instance_os_policies_compliance.GetInstanceOSPoliciesComplianceRequest, dict])
def test_get_instance_os_policies_compliance_rest(request_type):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instanceOSPoliciesCompliances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = instance_os_policies_compliance.InstanceOSPoliciesCompliance(name='name_value', instance='instance_value', state=config_common.OSPolicyComplianceState.COMPLIANT, detailed_state='detailed_state_value', detailed_state_reason='detailed_state_reason_value', last_compliance_run_id='last_compliance_run_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = instance_os_policies_compliance.InstanceOSPoliciesCompliance.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_instance_os_policies_compliance(request)
    assert isinstance(response, instance_os_policies_compliance.InstanceOSPoliciesCompliance)
    assert response.name == 'name_value'
    assert response.instance == 'instance_value'
    assert response.state == config_common.OSPolicyComplianceState.COMPLIANT
    assert response.detailed_state == 'detailed_state_value'
    assert response.detailed_state_reason == 'detailed_state_reason_value'
    assert response.last_compliance_run_id == 'last_compliance_run_id_value'

def test_get_instance_os_policies_compliance_rest_required_fields(request_type=instance_os_policies_compliance.GetInstanceOSPoliciesComplianceRequest):
    if False:
        return 10
    transport_class = transports.OsConfigZonalServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_instance_os_policies_compliance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_instance_os_policies_compliance._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = instance_os_policies_compliance.InstanceOSPoliciesCompliance()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = instance_os_policies_compliance.InstanceOSPoliciesCompliance.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_instance_os_policies_compliance(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_instance_os_policies_compliance_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_instance_os_policies_compliance._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_instance_os_policies_compliance_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigZonalServiceRestInterceptor())
    client = OsConfigZonalServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'post_get_instance_os_policies_compliance') as post, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'pre_get_instance_os_policies_compliance') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = instance_os_policies_compliance.GetInstanceOSPoliciesComplianceRequest.pb(instance_os_policies_compliance.GetInstanceOSPoliciesComplianceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = instance_os_policies_compliance.InstanceOSPoliciesCompliance.to_json(instance_os_policies_compliance.InstanceOSPoliciesCompliance())
        request = instance_os_policies_compliance.GetInstanceOSPoliciesComplianceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = instance_os_policies_compliance.InstanceOSPoliciesCompliance()
        client.get_instance_os_policies_compliance(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_instance_os_policies_compliance_rest_bad_request(transport: str='rest', request_type=instance_os_policies_compliance.GetInstanceOSPoliciesComplianceRequest):
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instanceOSPoliciesCompliances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_instance_os_policies_compliance(request)

def test_get_instance_os_policies_compliance_rest_flattened():
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = instance_os_policies_compliance.InstanceOSPoliciesCompliance()
        sample_request = {'name': 'projects/sample1/locations/sample2/instanceOSPoliciesCompliances/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = instance_os_policies_compliance.InstanceOSPoliciesCompliance.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_instance_os_policies_compliance(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=projects/*/locations/*/instanceOSPoliciesCompliances/*}' % client.transport._host, args[1])

def test_get_instance_os_policies_compliance_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_instance_os_policies_compliance(instance_os_policies_compliance.GetInstanceOSPoliciesComplianceRequest(), name='name_value')

def test_get_instance_os_policies_compliance_rest_error():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesRequest, dict])
def test_list_instance_os_policies_compliances_rest(request_type):
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_instance_os_policies_compliances(request)
    assert isinstance(response, pagers.ListInstanceOSPoliciesCompliancesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_instance_os_policies_compliances_rest_required_fields(request_type=instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.OsConfigZonalServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_instance_os_policies_compliances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_instance_os_policies_compliances._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_instance_os_policies_compliances(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_instance_os_policies_compliances_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_instance_os_policies_compliances._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_instance_os_policies_compliances_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigZonalServiceRestInterceptor())
    client = OsConfigZonalServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'post_list_instance_os_policies_compliances') as post, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'pre_list_instance_os_policies_compliances') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesRequest.pb(instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse.to_json(instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse())
        request = instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse()
        client.list_instance_os_policies_compliances(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_instance_os_policies_compliances_rest_bad_request(transport: str='rest', request_type=instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesRequest):
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_instance_os_policies_compliances(request)

def test_list_instance_os_policies_compliances_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_instance_os_policies_compliances(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{parent=projects/*/locations/*}/instanceOSPoliciesCompliances' % client.transport._host, args[1])

def test_list_instance_os_policies_compliances_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_instance_os_policies_compliances(instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesRequest(), parent='parent_value')

def test_list_instance_os_policies_compliances_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[instance_os_policies_compliance.InstanceOSPoliciesCompliance(), instance_os_policies_compliance.InstanceOSPoliciesCompliance(), instance_os_policies_compliance.InstanceOSPoliciesCompliance()], next_page_token='abc'), instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[], next_page_token='def'), instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[instance_os_policies_compliance.InstanceOSPoliciesCompliance()], next_page_token='ghi'), instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse(instance_os_policies_compliances=[instance_os_policies_compliance.InstanceOSPoliciesCompliance(), instance_os_policies_compliance.InstanceOSPoliciesCompliance()]))
        response = response + response
        response = tuple((instance_os_policies_compliance.ListInstanceOSPoliciesCompliancesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_instance_os_policies_compliances(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, instance_os_policies_compliance.InstanceOSPoliciesCompliance) for i in results))
        pages = list(client.list_instance_os_policies_compliances(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [os_policy_assignment_reports.GetOSPolicyAssignmentReportRequest, dict])
def test_get_os_policy_assignment_report_rest(request_type):
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3/osPolicyAssignments/sample4/report'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = os_policy_assignment_reports.OSPolicyAssignmentReport(name='name_value', instance='instance_value', os_policy_assignment='os_policy_assignment_value', last_run_id='last_run_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = os_policy_assignment_reports.OSPolicyAssignmentReport.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_os_policy_assignment_report(request)
    assert isinstance(response, os_policy_assignment_reports.OSPolicyAssignmentReport)
    assert response.name == 'name_value'
    assert response.instance == 'instance_value'
    assert response.os_policy_assignment == 'os_policy_assignment_value'
    assert response.last_run_id == 'last_run_id_value'

def test_get_os_policy_assignment_report_rest_required_fields(request_type=os_policy_assignment_reports.GetOSPolicyAssignmentReportRequest):
    if False:
        print('Hello World!')
    transport_class = transports.OsConfigZonalServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_os_policy_assignment_report._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_os_policy_assignment_report._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = os_policy_assignment_reports.OSPolicyAssignmentReport()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = os_policy_assignment_reports.OSPolicyAssignmentReport.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_os_policy_assignment_report(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_os_policy_assignment_report_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_os_policy_assignment_report._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_os_policy_assignment_report_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigZonalServiceRestInterceptor())
    client = OsConfigZonalServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'post_get_os_policy_assignment_report') as post, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'pre_get_os_policy_assignment_report') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = os_policy_assignment_reports.GetOSPolicyAssignmentReportRequest.pb(os_policy_assignment_reports.GetOSPolicyAssignmentReportRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = os_policy_assignment_reports.OSPolicyAssignmentReport.to_json(os_policy_assignment_reports.OSPolicyAssignmentReport())
        request = os_policy_assignment_reports.GetOSPolicyAssignmentReportRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = os_policy_assignment_reports.OSPolicyAssignmentReport()
        client.get_os_policy_assignment_report(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_os_policy_assignment_report_rest_bad_request(transport: str='rest', request_type=os_policy_assignment_reports.GetOSPolicyAssignmentReportRequest):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3/osPolicyAssignments/sample4/report'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_os_policy_assignment_report(request)

def test_get_os_policy_assignment_report_rest_flattened():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = os_policy_assignment_reports.OSPolicyAssignmentReport()
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3/osPolicyAssignments/sample4/report'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = os_policy_assignment_reports.OSPolicyAssignmentReport.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_os_policy_assignment_report(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=projects/*/locations/*/instances/*/osPolicyAssignments/*/report}' % client.transport._host, args[1])

def test_get_os_policy_assignment_report_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_os_policy_assignment_report(os_policy_assignment_reports.GetOSPolicyAssignmentReportRequest(), name='name_value')

def test_get_os_policy_assignment_report_rest_error():
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [os_policy_assignment_reports.ListOSPolicyAssignmentReportsRequest, dict])
def test_list_os_policy_assignment_reports_rest(request_type):
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/instances/sample3/osPolicyAssignments/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_os_policy_assignment_reports(request)
    assert isinstance(response, pagers.ListOSPolicyAssignmentReportsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_os_policy_assignment_reports_rest_required_fields(request_type=os_policy_assignment_reports.ListOSPolicyAssignmentReportsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.OsConfigZonalServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_os_policy_assignment_reports._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_os_policy_assignment_reports._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_os_policy_assignment_reports(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_os_policy_assignment_reports_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_os_policy_assignment_reports._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_os_policy_assignment_reports_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigZonalServiceRestInterceptor())
    client = OsConfigZonalServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'post_list_os_policy_assignment_reports') as post, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'pre_list_os_policy_assignment_reports') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = os_policy_assignment_reports.ListOSPolicyAssignmentReportsRequest.pb(os_policy_assignment_reports.ListOSPolicyAssignmentReportsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse.to_json(os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse())
        request = os_policy_assignment_reports.ListOSPolicyAssignmentReportsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse()
        client.list_os_policy_assignment_reports(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_os_policy_assignment_reports_rest_bad_request(transport: str='rest', request_type=os_policy_assignment_reports.ListOSPolicyAssignmentReportsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/instances/sample3/osPolicyAssignments/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_os_policy_assignment_reports(request)

def test_list_os_policy_assignment_reports_rest_flattened():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/instances/sample3/osPolicyAssignments/sample4'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_os_policy_assignment_reports(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{parent=projects/*/locations/*/instances/*/osPolicyAssignments/*}/reports' % client.transport._host, args[1])

def test_list_os_policy_assignment_reports_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_os_policy_assignment_reports(os_policy_assignment_reports.ListOSPolicyAssignmentReportsRequest(), parent='parent_value')

def test_list_os_policy_assignment_reports_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[os_policy_assignment_reports.OSPolicyAssignmentReport(), os_policy_assignment_reports.OSPolicyAssignmentReport(), os_policy_assignment_reports.OSPolicyAssignmentReport()], next_page_token='abc'), os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[], next_page_token='def'), os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[os_policy_assignment_reports.OSPolicyAssignmentReport()], next_page_token='ghi'), os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse(os_policy_assignment_reports=[os_policy_assignment_reports.OSPolicyAssignmentReport(), os_policy_assignment_reports.OSPolicyAssignmentReport()]))
        response = response + response
        response = tuple((os_policy_assignment_reports.ListOSPolicyAssignmentReportsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/instances/sample3/osPolicyAssignments/sample4'}
        pager = client.list_os_policy_assignment_reports(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, os_policy_assignment_reports.OSPolicyAssignmentReport) for i in results))
        pages = list(client.list_os_policy_assignment_reports(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [inventory.GetInventoryRequest, dict])
def test_get_inventory_rest(request_type):
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3/inventory'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = inventory.Inventory(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = inventory.Inventory.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_inventory(request)
    assert isinstance(response, inventory.Inventory)
    assert response.name == 'name_value'

def test_get_inventory_rest_required_fields(request_type=inventory.GetInventoryRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.OsConfigZonalServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_inventory._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_inventory._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('view',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = inventory.Inventory()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = inventory.Inventory.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_inventory(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_inventory_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_inventory._get_unset_required_fields({})
    assert set(unset_fields) == set(('view',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_inventory_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigZonalServiceRestInterceptor())
    client = OsConfigZonalServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'post_get_inventory') as post, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'pre_get_inventory') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = inventory.GetInventoryRequest.pb(inventory.GetInventoryRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = inventory.Inventory.to_json(inventory.Inventory())
        request = inventory.GetInventoryRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = inventory.Inventory()
        client.get_inventory(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_inventory_rest_bad_request(transport: str='rest', request_type=inventory.GetInventoryRequest):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3/inventory'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_inventory(request)

def test_get_inventory_rest_flattened():
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = inventory.Inventory()
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3/inventory'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = inventory.Inventory.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_inventory(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=projects/*/locations/*/instances/*/inventory}' % client.transport._host, args[1])

def test_get_inventory_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_inventory(inventory.GetInventoryRequest(), name='name_value')

def test_get_inventory_rest_error():
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [inventory.ListInventoriesRequest, dict])
def test_list_inventories_rest(request_type):
    if False:
        return 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = inventory.ListInventoriesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = inventory.ListInventoriesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_inventories(request)
    assert isinstance(response, pagers.ListInventoriesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_inventories_rest_required_fields(request_type=inventory.ListInventoriesRequest):
    if False:
        print('Hello World!')
    transport_class = transports.OsConfigZonalServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_inventories._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_inventories._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token', 'view'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = inventory.ListInventoriesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = inventory.ListInventoriesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_inventories(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_inventories_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_inventories._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken', 'view')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_inventories_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigZonalServiceRestInterceptor())
    client = OsConfigZonalServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'post_list_inventories') as post, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'pre_list_inventories') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = inventory.ListInventoriesRequest.pb(inventory.ListInventoriesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = inventory.ListInventoriesResponse.to_json(inventory.ListInventoriesResponse())
        request = inventory.ListInventoriesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = inventory.ListInventoriesResponse()
        client.list_inventories(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_inventories_rest_bad_request(transport: str='rest', request_type=inventory.ListInventoriesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_inventories(request)

def test_list_inventories_rest_flattened():
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = inventory.ListInventoriesResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = inventory.ListInventoriesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_inventories(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{parent=projects/*/locations/*/instances/*}/inventories' % client.transport._host, args[1])

def test_list_inventories_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_inventories(inventory.ListInventoriesRequest(), parent='parent_value')

def test_list_inventories_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (inventory.ListInventoriesResponse(inventories=[inventory.Inventory(), inventory.Inventory(), inventory.Inventory()], next_page_token='abc'), inventory.ListInventoriesResponse(inventories=[], next_page_token='def'), inventory.ListInventoriesResponse(inventories=[inventory.Inventory()], next_page_token='ghi'), inventory.ListInventoriesResponse(inventories=[inventory.Inventory(), inventory.Inventory()]))
        response = response + response
        response = tuple((inventory.ListInventoriesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/instances/sample3'}
        pager = client.list_inventories(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, inventory.Inventory) for i in results))
        pages = list(client.list_inventories(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [vulnerability.GetVulnerabilityReportRequest, dict])
def test_get_vulnerability_report_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3/vulnerabilityReport'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vulnerability.VulnerabilityReport(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = vulnerability.VulnerabilityReport.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_vulnerability_report(request)
    assert isinstance(response, vulnerability.VulnerabilityReport)
    assert response.name == 'name_value'

def test_get_vulnerability_report_rest_required_fields(request_type=vulnerability.GetVulnerabilityReportRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.OsConfigZonalServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_vulnerability_report._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_vulnerability_report._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vulnerability.VulnerabilityReport()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vulnerability.VulnerabilityReport.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_vulnerability_report(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_vulnerability_report_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_vulnerability_report._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_vulnerability_report_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigZonalServiceRestInterceptor())
    client = OsConfigZonalServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'post_get_vulnerability_report') as post, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'pre_get_vulnerability_report') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vulnerability.GetVulnerabilityReportRequest.pb(vulnerability.GetVulnerabilityReportRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vulnerability.VulnerabilityReport.to_json(vulnerability.VulnerabilityReport())
        request = vulnerability.GetVulnerabilityReportRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vulnerability.VulnerabilityReport()
        client.get_vulnerability_report(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_vulnerability_report_rest_bad_request(transport: str='rest', request_type=vulnerability.GetVulnerabilityReportRequest):
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/instances/sample3/vulnerabilityReport'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_vulnerability_report(request)

def test_get_vulnerability_report_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vulnerability.VulnerabilityReport()
        sample_request = {'name': 'projects/sample1/locations/sample2/instances/sample3/vulnerabilityReport'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vulnerability.VulnerabilityReport.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_vulnerability_report(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{name=projects/*/locations/*/instances/*/vulnerabilityReport}' % client.transport._host, args[1])

def test_get_vulnerability_report_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_vulnerability_report(vulnerability.GetVulnerabilityReportRequest(), name='name_value')

def test_get_vulnerability_report_rest_error():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [vulnerability.ListVulnerabilityReportsRequest, dict])
def test_list_vulnerability_reports_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vulnerability.ListVulnerabilityReportsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = vulnerability.ListVulnerabilityReportsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_vulnerability_reports(request)
    assert isinstance(response, pagers.ListVulnerabilityReportsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_vulnerability_reports_rest_required_fields(request_type=vulnerability.ListVulnerabilityReportsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.OsConfigZonalServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_vulnerability_reports._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_vulnerability_reports._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = vulnerability.ListVulnerabilityReportsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = vulnerability.ListVulnerabilityReportsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_vulnerability_reports(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_vulnerability_reports_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_vulnerability_reports._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_vulnerability_reports_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.OsConfigZonalServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.OsConfigZonalServiceRestInterceptor())
    client = OsConfigZonalServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'post_list_vulnerability_reports') as post, mock.patch.object(transports.OsConfigZonalServiceRestInterceptor, 'pre_list_vulnerability_reports') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = vulnerability.ListVulnerabilityReportsRequest.pb(vulnerability.ListVulnerabilityReportsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = vulnerability.ListVulnerabilityReportsResponse.to_json(vulnerability.ListVulnerabilityReportsResponse())
        request = vulnerability.ListVulnerabilityReportsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = vulnerability.ListVulnerabilityReportsResponse()
        client.list_vulnerability_reports(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_vulnerability_reports_rest_bad_request(transport: str='rest', request_type=vulnerability.ListVulnerabilityReportsRequest):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/instances/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_vulnerability_reports(request)

def test_list_vulnerability_reports_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = vulnerability.ListVulnerabilityReportsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/instances/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = vulnerability.ListVulnerabilityReportsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_vulnerability_reports(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1alpha/{parent=projects/*/locations/*/instances/*}/vulnerabilityReports' % client.transport._host, args[1])

def test_list_vulnerability_reports_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_vulnerability_reports(vulnerability.ListVulnerabilityReportsRequest(), parent='parent_value')

def test_list_vulnerability_reports_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[vulnerability.VulnerabilityReport(), vulnerability.VulnerabilityReport(), vulnerability.VulnerabilityReport()], next_page_token='abc'), vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[], next_page_token='def'), vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[vulnerability.VulnerabilityReport()], next_page_token='ghi'), vulnerability.ListVulnerabilityReportsResponse(vulnerability_reports=[vulnerability.VulnerabilityReport(), vulnerability.VulnerabilityReport()]))
        response = response + response
        response = tuple((vulnerability.ListVulnerabilityReportsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/instances/sample3'}
        pager = client.list_vulnerability_reports(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, vulnerability.VulnerabilityReport) for i in results))
        pages = list(client.list_vulnerability_reports(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.OsConfigZonalServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.OsConfigZonalServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = OsConfigZonalServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.OsConfigZonalServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = OsConfigZonalServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = OsConfigZonalServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.OsConfigZonalServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = OsConfigZonalServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.OsConfigZonalServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = OsConfigZonalServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.OsConfigZonalServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.OsConfigZonalServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.OsConfigZonalServiceGrpcTransport, transports.OsConfigZonalServiceGrpcAsyncIOTransport, transports.OsConfigZonalServiceRestTransport])
def test_transport_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        i = 10
        return i + 15
    transport = OsConfigZonalServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.OsConfigZonalServiceGrpcTransport)

def test_os_config_zonal_service_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.OsConfigZonalServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_os_config_zonal_service_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.osconfig_v1alpha.services.os_config_zonal_service.transports.OsConfigZonalServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.OsConfigZonalServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_os_policy_assignment', 'update_os_policy_assignment', 'get_os_policy_assignment', 'list_os_policy_assignments', 'list_os_policy_assignment_revisions', 'delete_os_policy_assignment', 'get_instance_os_policies_compliance', 'list_instance_os_policies_compliances', 'get_os_policy_assignment_report', 'list_os_policy_assignment_reports', 'get_inventory', 'list_inventories', 'get_vulnerability_report', 'list_vulnerability_reports')
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

def test_os_config_zonal_service_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.osconfig_v1alpha.services.os_config_zonal_service.transports.OsConfigZonalServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.OsConfigZonalServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_os_config_zonal_service_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.osconfig_v1alpha.services.os_config_zonal_service.transports.OsConfigZonalServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.OsConfigZonalServiceTransport()
        adc.assert_called_once()

def test_os_config_zonal_service_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        OsConfigZonalServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.OsConfigZonalServiceGrpcTransport, transports.OsConfigZonalServiceGrpcAsyncIOTransport])
def test_os_config_zonal_service_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.OsConfigZonalServiceGrpcTransport, transports.OsConfigZonalServiceGrpcAsyncIOTransport, transports.OsConfigZonalServiceRestTransport])
def test_os_config_zonal_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.OsConfigZonalServiceGrpcTransport, grpc_helpers), (transports.OsConfigZonalServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_os_config_zonal_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('osconfig.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='osconfig.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.OsConfigZonalServiceGrpcTransport, transports.OsConfigZonalServiceGrpcAsyncIOTransport])
def test_os_config_zonal_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_os_config_zonal_service_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.OsConfigZonalServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_os_config_zonal_service_rest_lro_client():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_os_config_zonal_service_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='osconfig.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('osconfig.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://osconfig.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_os_config_zonal_service_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='osconfig.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('osconfig.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://osconfig.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_os_config_zonal_service_client_transport_session_collision(transport_name):
    if False:
        while True:
            i = 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = OsConfigZonalServiceClient(credentials=creds1, transport=transport_name)
    client2 = OsConfigZonalServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_os_policy_assignment._session
    session2 = client2.transport.create_os_policy_assignment._session
    assert session1 != session2
    session1 = client1.transport.update_os_policy_assignment._session
    session2 = client2.transport.update_os_policy_assignment._session
    assert session1 != session2
    session1 = client1.transport.get_os_policy_assignment._session
    session2 = client2.transport.get_os_policy_assignment._session
    assert session1 != session2
    session1 = client1.transport.list_os_policy_assignments._session
    session2 = client2.transport.list_os_policy_assignments._session
    assert session1 != session2
    session1 = client1.transport.list_os_policy_assignment_revisions._session
    session2 = client2.transport.list_os_policy_assignment_revisions._session
    assert session1 != session2
    session1 = client1.transport.delete_os_policy_assignment._session
    session2 = client2.transport.delete_os_policy_assignment._session
    assert session1 != session2
    session1 = client1.transport.get_instance_os_policies_compliance._session
    session2 = client2.transport.get_instance_os_policies_compliance._session
    assert session1 != session2
    session1 = client1.transport.list_instance_os_policies_compliances._session
    session2 = client2.transport.list_instance_os_policies_compliances._session
    assert session1 != session2
    session1 = client1.transport.get_os_policy_assignment_report._session
    session2 = client2.transport.get_os_policy_assignment_report._session
    assert session1 != session2
    session1 = client1.transport.list_os_policy_assignment_reports._session
    session2 = client2.transport.list_os_policy_assignment_reports._session
    assert session1 != session2
    session1 = client1.transport.get_inventory._session
    session2 = client2.transport.get_inventory._session
    assert session1 != session2
    session1 = client1.transport.list_inventories._session
    session2 = client2.transport.list_inventories._session
    assert session1 != session2
    session1 = client1.transport.get_vulnerability_report._session
    session2 = client2.transport.get_vulnerability_report._session
    assert session1 != session2
    session1 = client1.transport.list_vulnerability_reports._session
    session2 = client2.transport.list_vulnerability_reports._session
    assert session1 != session2

def test_os_config_zonal_service_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.OsConfigZonalServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_os_config_zonal_service_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.OsConfigZonalServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.OsConfigZonalServiceGrpcTransport, transports.OsConfigZonalServiceGrpcAsyncIOTransport])
def test_os_config_zonal_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.OsConfigZonalServiceGrpcTransport, transports.OsConfigZonalServiceGrpcAsyncIOTransport])
def test_os_config_zonal_service_transport_channel_mtls_with_adc(transport_class):
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

def test_os_config_zonal_service_grpc_lro_client():
    if False:
        while True:
            i = 10
    client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_os_config_zonal_service_grpc_lro_async_client():
    if False:
        for i in range(10):
            print('nop')
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_instance_path():
    if False:
        return 10
    project = 'squid'
    location = 'clam'
    instance = 'whelk'
    expected = 'projects/{project}/locations/{location}/instances/{instance}'.format(project=project, location=location, instance=instance)
    actual = OsConfigZonalServiceClient.instance_path(project, location, instance)
    assert expected == actual

def test_parse_instance_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus', 'location': 'oyster', 'instance': 'nudibranch'}
    path = OsConfigZonalServiceClient.instance_path(**expected)
    actual = OsConfigZonalServiceClient.parse_instance_path(path)
    assert expected == actual

def test_instance_os_policies_compliance_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    location = 'mussel'
    instance = 'winkle'
    expected = 'projects/{project}/locations/{location}/instanceOSPoliciesCompliances/{instance}'.format(project=project, location=location, instance=instance)
    actual = OsConfigZonalServiceClient.instance_os_policies_compliance_path(project, location, instance)
    assert expected == actual

def test_parse_instance_os_policies_compliance_path():
    if False:
        return 10
    expected = {'project': 'nautilus', 'location': 'scallop', 'instance': 'abalone'}
    path = OsConfigZonalServiceClient.instance_os_policies_compliance_path(**expected)
    actual = OsConfigZonalServiceClient.parse_instance_os_policies_compliance_path(path)
    assert expected == actual

def test_instance_os_policy_assignment_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    instance = 'whelk'
    assignment = 'octopus'
    expected = 'projects/{project}/locations/{location}/instances/{instance}/osPolicyAssignments/{assignment}'.format(project=project, location=location, instance=instance, assignment=assignment)
    actual = OsConfigZonalServiceClient.instance_os_policy_assignment_path(project, location, instance, assignment)
    assert expected == actual

def test_parse_instance_os_policy_assignment_path():
    if False:
        return 10
    expected = {'project': 'oyster', 'location': 'nudibranch', 'instance': 'cuttlefish', 'assignment': 'mussel'}
    path = OsConfigZonalServiceClient.instance_os_policy_assignment_path(**expected)
    actual = OsConfigZonalServiceClient.parse_instance_os_policy_assignment_path(path)
    assert expected == actual

def test_inventory_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    instance = 'scallop'
    expected = 'projects/{project}/locations/{location}/instances/{instance}/inventory'.format(project=project, location=location, instance=instance)
    actual = OsConfigZonalServiceClient.inventory_path(project, location, instance)
    assert expected == actual

def test_parse_inventory_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'abalone', 'location': 'squid', 'instance': 'clam'}
    path = OsConfigZonalServiceClient.inventory_path(**expected)
    actual = OsConfigZonalServiceClient.parse_inventory_path(path)
    assert expected == actual

def test_os_policy_assignment_path():
    if False:
        print('Hello World!')
    project = 'whelk'
    location = 'octopus'
    os_policy_assignment = 'oyster'
    expected = 'projects/{project}/locations/{location}/osPolicyAssignments/{os_policy_assignment}'.format(project=project, location=location, os_policy_assignment=os_policy_assignment)
    actual = OsConfigZonalServiceClient.os_policy_assignment_path(project, location, os_policy_assignment)
    assert expected == actual

def test_parse_os_policy_assignment_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nudibranch', 'location': 'cuttlefish', 'os_policy_assignment': 'mussel'}
    path = OsConfigZonalServiceClient.os_policy_assignment_path(**expected)
    actual = OsConfigZonalServiceClient.parse_os_policy_assignment_path(path)
    assert expected == actual

def test_os_policy_assignment_report_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    instance = 'scallop'
    assignment = 'abalone'
    expected = 'projects/{project}/locations/{location}/instances/{instance}/osPolicyAssignments/{assignment}/report'.format(project=project, location=location, instance=instance, assignment=assignment)
    actual = OsConfigZonalServiceClient.os_policy_assignment_report_path(project, location, instance, assignment)
    assert expected == actual

def test_parse_os_policy_assignment_report_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'squid', 'location': 'clam', 'instance': 'whelk', 'assignment': 'octopus'}
    path = OsConfigZonalServiceClient.os_policy_assignment_report_path(**expected)
    actual = OsConfigZonalServiceClient.parse_os_policy_assignment_report_path(path)
    assert expected == actual

def test_vulnerability_report_path():
    if False:
        while True:
            i = 10
    project = 'oyster'
    location = 'nudibranch'
    instance = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/instances/{instance}/vulnerabilityReport'.format(project=project, location=location, instance=instance)
    actual = OsConfigZonalServiceClient.vulnerability_report_path(project, location, instance)
    assert expected == actual

def test_parse_vulnerability_report_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'mussel', 'location': 'winkle', 'instance': 'nautilus'}
    path = OsConfigZonalServiceClient.vulnerability_report_path(**expected)
    actual = OsConfigZonalServiceClient.parse_vulnerability_report_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'scallop'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = OsConfigZonalServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'abalone'}
    path = OsConfigZonalServiceClient.common_billing_account_path(**expected)
    actual = OsConfigZonalServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'squid'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = OsConfigZonalServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'clam'}
    path = OsConfigZonalServiceClient.common_folder_path(**expected)
    actual = OsConfigZonalServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'whelk'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = OsConfigZonalServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'octopus'}
    path = OsConfigZonalServiceClient.common_organization_path(**expected)
    actual = OsConfigZonalServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'oyster'
    expected = 'projects/{project}'.format(project=project)
    actual = OsConfigZonalServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'nudibranch'}
    path = OsConfigZonalServiceClient.common_project_path(**expected)
    actual = OsConfigZonalServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'cuttlefish'
    location = 'mussel'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = OsConfigZonalServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'winkle', 'location': 'nautilus'}
    path = OsConfigZonalServiceClient.common_location_path(**expected)
    actual = OsConfigZonalServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.OsConfigZonalServiceTransport, '_prep_wrapped_messages') as prep:
        client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.OsConfigZonalServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = OsConfigZonalServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = OsConfigZonalServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = OsConfigZonalServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(OsConfigZonalServiceClient, transports.OsConfigZonalServiceGrpcTransport), (OsConfigZonalServiceAsyncClient, transports.OsConfigZonalServiceGrpcAsyncIOTransport)])
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