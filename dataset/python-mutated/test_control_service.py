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
from google.protobuf import timestamp_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.retail_v2.services.control_service import ControlServiceAsyncClient, ControlServiceClient, pagers, transports
from google.cloud.retail_v2.types import common
from google.cloud.retail_v2.types import control
from google.cloud.retail_v2.types import control as gcr_control
from google.cloud.retail_v2.types import control_service

def client_cert_source_callback():
    if False:
        print('Hello World!')
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
    assert ControlServiceClient._get_default_mtls_endpoint(None) is None
    assert ControlServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert ControlServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert ControlServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert ControlServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert ControlServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(ControlServiceClient, 'grpc'), (ControlServiceAsyncClient, 'grpc_asyncio'), (ControlServiceClient, 'rest')])
def test_control_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('retail.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://retail.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.ControlServiceGrpcTransport, 'grpc'), (transports.ControlServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.ControlServiceRestTransport, 'rest')])
def test_control_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(ControlServiceClient, 'grpc'), (ControlServiceAsyncClient, 'grpc_asyncio'), (ControlServiceClient, 'rest')])
def test_control_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('retail.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://retail.googleapis.com')

def test_control_service_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = ControlServiceClient.get_transport_class()
    available_transports = [transports.ControlServiceGrpcTransport, transports.ControlServiceRestTransport]
    assert transport in available_transports
    transport = ControlServiceClient.get_transport_class('grpc')
    assert transport == transports.ControlServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ControlServiceClient, transports.ControlServiceGrpcTransport, 'grpc'), (ControlServiceAsyncClient, transports.ControlServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (ControlServiceClient, transports.ControlServiceRestTransport, 'rest')])
@mock.patch.object(ControlServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ControlServiceClient))
@mock.patch.object(ControlServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ControlServiceAsyncClient))
def test_control_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(ControlServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(ControlServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(ControlServiceClient, transports.ControlServiceGrpcTransport, 'grpc', 'true'), (ControlServiceAsyncClient, transports.ControlServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (ControlServiceClient, transports.ControlServiceGrpcTransport, 'grpc', 'false'), (ControlServiceAsyncClient, transports.ControlServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (ControlServiceClient, transports.ControlServiceRestTransport, 'rest', 'true'), (ControlServiceClient, transports.ControlServiceRestTransport, 'rest', 'false')])
@mock.patch.object(ControlServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ControlServiceClient))
@mock.patch.object(ControlServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ControlServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_control_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [ControlServiceClient, ControlServiceAsyncClient])
@mock.patch.object(ControlServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ControlServiceClient))
@mock.patch.object(ControlServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ControlServiceAsyncClient))
def test_control_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ControlServiceClient, transports.ControlServiceGrpcTransport, 'grpc'), (ControlServiceAsyncClient, transports.ControlServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (ControlServiceClient, transports.ControlServiceRestTransport, 'rest')])
def test_control_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ControlServiceClient, transports.ControlServiceGrpcTransport, 'grpc', grpc_helpers), (ControlServiceAsyncClient, transports.ControlServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (ControlServiceClient, transports.ControlServiceRestTransport, 'rest', None)])
def test_control_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_control_service_client_client_options_from_dict():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.retail_v2.services.control_service.transports.ControlServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = ControlServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ControlServiceClient, transports.ControlServiceGrpcTransport, 'grpc', grpc_helpers), (ControlServiceAsyncClient, transports.ControlServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_control_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('retail.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='retail.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [control_service.CreateControlRequest, dict])
def test_create_control(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_control), '__call__') as call:
        call.return_value = gcr_control.Control(name='name_value', display_name='display_name_value', associated_serving_config_ids=['associated_serving_config_ids_value'], solution_types=[common.SolutionType.SOLUTION_TYPE_RECOMMENDATION], search_solution_use_case=[common.SearchSolutionUseCase.SEARCH_SOLUTION_USE_CASE_SEARCH])
        response = client.create_control(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == control_service.CreateControlRequest()
    assert isinstance(response, gcr_control.Control)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.associated_serving_config_ids == ['associated_serving_config_ids_value']
    assert response.solution_types == [common.SolutionType.SOLUTION_TYPE_RECOMMENDATION]
    assert response.search_solution_use_case == [common.SearchSolutionUseCase.SEARCH_SOLUTION_USE_CASE_SEARCH]

def test_create_control_empty_call():
    if False:
        return 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_control), '__call__') as call:
        client.create_control()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == control_service.CreateControlRequest()

@pytest.mark.asyncio
async def test_create_control_async(transport: str='grpc_asyncio', request_type=control_service.CreateControlRequest):
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_control), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcr_control.Control(name='name_value', display_name='display_name_value', associated_serving_config_ids=['associated_serving_config_ids_value'], solution_types=[common.SolutionType.SOLUTION_TYPE_RECOMMENDATION], search_solution_use_case=[common.SearchSolutionUseCase.SEARCH_SOLUTION_USE_CASE_SEARCH]))
        response = await client.create_control(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == control_service.CreateControlRequest()
    assert isinstance(response, gcr_control.Control)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.associated_serving_config_ids == ['associated_serving_config_ids_value']
    assert response.solution_types == [common.SolutionType.SOLUTION_TYPE_RECOMMENDATION]
    assert response.search_solution_use_case == [common.SearchSolutionUseCase.SEARCH_SOLUTION_USE_CASE_SEARCH]

@pytest.mark.asyncio
async def test_create_control_async_from_dict():
    await test_create_control_async(request_type=dict)

def test_create_control_field_headers():
    if False:
        print('Hello World!')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = control_service.CreateControlRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_control), '__call__') as call:
        call.return_value = gcr_control.Control()
        client.create_control(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_control_field_headers_async():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = control_service.CreateControlRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_control), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcr_control.Control())
        await client.create_control(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_control_flattened():
    if False:
        while True:
            i = 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_control), '__call__') as call:
        call.return_value = gcr_control.Control()
        client.create_control(parent='parent_value', control=gcr_control.Control(rule=common.Rule(boost_action=common.Rule.BoostAction(boost=0.551))), control_id='control_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].control
        mock_val = gcr_control.Control(rule=common.Rule(boost_action=common.Rule.BoostAction(boost=0.551)))
        assert arg == mock_val
        arg = args[0].control_id
        mock_val = 'control_id_value'
        assert arg == mock_val

def test_create_control_flattened_error():
    if False:
        print('Hello World!')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_control(control_service.CreateControlRequest(), parent='parent_value', control=gcr_control.Control(rule=common.Rule(boost_action=common.Rule.BoostAction(boost=0.551))), control_id='control_id_value')

@pytest.mark.asyncio
async def test_create_control_flattened_async():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_control), '__call__') as call:
        call.return_value = gcr_control.Control()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcr_control.Control())
        response = await client.create_control(parent='parent_value', control=gcr_control.Control(rule=common.Rule(boost_action=common.Rule.BoostAction(boost=0.551))), control_id='control_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].control
        mock_val = gcr_control.Control(rule=common.Rule(boost_action=common.Rule.BoostAction(boost=0.551)))
        assert arg == mock_val
        arg = args[0].control_id
        mock_val = 'control_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_control_flattened_error_async():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_control(control_service.CreateControlRequest(), parent='parent_value', control=gcr_control.Control(rule=common.Rule(boost_action=common.Rule.BoostAction(boost=0.551))), control_id='control_id_value')

@pytest.mark.parametrize('request_type', [control_service.DeleteControlRequest, dict])
def test_delete_control(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_control), '__call__') as call:
        call.return_value = None
        response = client.delete_control(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == control_service.DeleteControlRequest()
    assert response is None

def test_delete_control_empty_call():
    if False:
        i = 10
        return i + 15
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_control), '__call__') as call:
        client.delete_control()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == control_service.DeleteControlRequest()

@pytest.mark.asyncio
async def test_delete_control_async(transport: str='grpc_asyncio', request_type=control_service.DeleteControlRequest):
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_control), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_control(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == control_service.DeleteControlRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_control_async_from_dict():
    await test_delete_control_async(request_type=dict)

def test_delete_control_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = control_service.DeleteControlRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_control), '__call__') as call:
        call.return_value = None
        client.delete_control(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_control_field_headers_async():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = control_service.DeleteControlRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_control), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_control(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_control_flattened():
    if False:
        i = 10
        return i + 15
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_control), '__call__') as call:
        call.return_value = None
        client.delete_control(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_control_flattened_error():
    if False:
        print('Hello World!')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_control(control_service.DeleteControlRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_control_flattened_async():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_control), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_control(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_control_flattened_error_async():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_control(control_service.DeleteControlRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [control_service.UpdateControlRequest, dict])
def test_update_control(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_control), '__call__') as call:
        call.return_value = gcr_control.Control(name='name_value', display_name='display_name_value', associated_serving_config_ids=['associated_serving_config_ids_value'], solution_types=[common.SolutionType.SOLUTION_TYPE_RECOMMENDATION], search_solution_use_case=[common.SearchSolutionUseCase.SEARCH_SOLUTION_USE_CASE_SEARCH])
        response = client.update_control(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == control_service.UpdateControlRequest()
    assert isinstance(response, gcr_control.Control)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.associated_serving_config_ids == ['associated_serving_config_ids_value']
    assert response.solution_types == [common.SolutionType.SOLUTION_TYPE_RECOMMENDATION]
    assert response.search_solution_use_case == [common.SearchSolutionUseCase.SEARCH_SOLUTION_USE_CASE_SEARCH]

def test_update_control_empty_call():
    if False:
        while True:
            i = 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_control), '__call__') as call:
        client.update_control()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == control_service.UpdateControlRequest()

@pytest.mark.asyncio
async def test_update_control_async(transport: str='grpc_asyncio', request_type=control_service.UpdateControlRequest):
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_control), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcr_control.Control(name='name_value', display_name='display_name_value', associated_serving_config_ids=['associated_serving_config_ids_value'], solution_types=[common.SolutionType.SOLUTION_TYPE_RECOMMENDATION], search_solution_use_case=[common.SearchSolutionUseCase.SEARCH_SOLUTION_USE_CASE_SEARCH]))
        response = await client.update_control(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == control_service.UpdateControlRequest()
    assert isinstance(response, gcr_control.Control)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.associated_serving_config_ids == ['associated_serving_config_ids_value']
    assert response.solution_types == [common.SolutionType.SOLUTION_TYPE_RECOMMENDATION]
    assert response.search_solution_use_case == [common.SearchSolutionUseCase.SEARCH_SOLUTION_USE_CASE_SEARCH]

@pytest.mark.asyncio
async def test_update_control_async_from_dict():
    await test_update_control_async(request_type=dict)

def test_update_control_field_headers():
    if False:
        print('Hello World!')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = control_service.UpdateControlRequest()
    request.control.name = 'name_value'
    with mock.patch.object(type(client.transport.update_control), '__call__') as call:
        call.return_value = gcr_control.Control()
        client.update_control(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'control.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_control_field_headers_async():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = control_service.UpdateControlRequest()
    request.control.name = 'name_value'
    with mock.patch.object(type(client.transport.update_control), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcr_control.Control())
        await client.update_control(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'control.name=name_value') in kw['metadata']

def test_update_control_flattened():
    if False:
        return 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_control), '__call__') as call:
        call.return_value = gcr_control.Control()
        client.update_control(control=gcr_control.Control(rule=common.Rule(boost_action=common.Rule.BoostAction(boost=0.551))), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].control
        mock_val = gcr_control.Control(rule=common.Rule(boost_action=common.Rule.BoostAction(boost=0.551)))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_control_flattened_error():
    if False:
        return 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_control(control_service.UpdateControlRequest(), control=gcr_control.Control(rule=common.Rule(boost_action=common.Rule.BoostAction(boost=0.551))), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_control_flattened_async():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_control), '__call__') as call:
        call.return_value = gcr_control.Control()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcr_control.Control())
        response = await client.update_control(control=gcr_control.Control(rule=common.Rule(boost_action=common.Rule.BoostAction(boost=0.551))), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].control
        mock_val = gcr_control.Control(rule=common.Rule(boost_action=common.Rule.BoostAction(boost=0.551)))
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_control_flattened_error_async():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_control(control_service.UpdateControlRequest(), control=gcr_control.Control(rule=common.Rule(boost_action=common.Rule.BoostAction(boost=0.551))), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [control_service.GetControlRequest, dict])
def test_get_control(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_control), '__call__') as call:
        call.return_value = control.Control(name='name_value', display_name='display_name_value', associated_serving_config_ids=['associated_serving_config_ids_value'], solution_types=[common.SolutionType.SOLUTION_TYPE_RECOMMENDATION], search_solution_use_case=[common.SearchSolutionUseCase.SEARCH_SOLUTION_USE_CASE_SEARCH])
        response = client.get_control(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == control_service.GetControlRequest()
    assert isinstance(response, control.Control)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.associated_serving_config_ids == ['associated_serving_config_ids_value']
    assert response.solution_types == [common.SolutionType.SOLUTION_TYPE_RECOMMENDATION]
    assert response.search_solution_use_case == [common.SearchSolutionUseCase.SEARCH_SOLUTION_USE_CASE_SEARCH]

def test_get_control_empty_call():
    if False:
        return 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_control), '__call__') as call:
        client.get_control()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == control_service.GetControlRequest()

@pytest.mark.asyncio
async def test_get_control_async(transport: str='grpc_asyncio', request_type=control_service.GetControlRequest):
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_control), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(control.Control(name='name_value', display_name='display_name_value', associated_serving_config_ids=['associated_serving_config_ids_value'], solution_types=[common.SolutionType.SOLUTION_TYPE_RECOMMENDATION], search_solution_use_case=[common.SearchSolutionUseCase.SEARCH_SOLUTION_USE_CASE_SEARCH]))
        response = await client.get_control(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == control_service.GetControlRequest()
    assert isinstance(response, control.Control)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.associated_serving_config_ids == ['associated_serving_config_ids_value']
    assert response.solution_types == [common.SolutionType.SOLUTION_TYPE_RECOMMENDATION]
    assert response.search_solution_use_case == [common.SearchSolutionUseCase.SEARCH_SOLUTION_USE_CASE_SEARCH]

@pytest.mark.asyncio
async def test_get_control_async_from_dict():
    await test_get_control_async(request_type=dict)

def test_get_control_field_headers():
    if False:
        while True:
            i = 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = control_service.GetControlRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_control), '__call__') as call:
        call.return_value = control.Control()
        client.get_control(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_control_field_headers_async():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = control_service.GetControlRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_control), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(control.Control())
        await client.get_control(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_control_flattened():
    if False:
        i = 10
        return i + 15
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_control), '__call__') as call:
        call.return_value = control.Control()
        client.get_control(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_control_flattened_error():
    if False:
        while True:
            i = 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_control(control_service.GetControlRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_control_flattened_async():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_control), '__call__') as call:
        call.return_value = control.Control()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(control.Control())
        response = await client.get_control(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_control_flattened_error_async():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_control(control_service.GetControlRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [control_service.ListControlsRequest, dict])
def test_list_controls(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_controls), '__call__') as call:
        call.return_value = control_service.ListControlsResponse(next_page_token='next_page_token_value')
        response = client.list_controls(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == control_service.ListControlsRequest()
    assert isinstance(response, pagers.ListControlsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_controls_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_controls), '__call__') as call:
        client.list_controls()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == control_service.ListControlsRequest()

@pytest.mark.asyncio
async def test_list_controls_async(transport: str='grpc_asyncio', request_type=control_service.ListControlsRequest):
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_controls), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(control_service.ListControlsResponse(next_page_token='next_page_token_value'))
        response = await client.list_controls(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == control_service.ListControlsRequest()
    assert isinstance(response, pagers.ListControlsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_controls_async_from_dict():
    await test_list_controls_async(request_type=dict)

def test_list_controls_field_headers():
    if False:
        while True:
            i = 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = control_service.ListControlsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_controls), '__call__') as call:
        call.return_value = control_service.ListControlsResponse()
        client.list_controls(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_controls_field_headers_async():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = control_service.ListControlsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_controls), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(control_service.ListControlsResponse())
        await client.list_controls(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_controls_flattened():
    if False:
        while True:
            i = 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_controls), '__call__') as call:
        call.return_value = control_service.ListControlsResponse()
        client.list_controls(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_controls_flattened_error():
    if False:
        print('Hello World!')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_controls(control_service.ListControlsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_controls_flattened_async():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_controls), '__call__') as call:
        call.return_value = control_service.ListControlsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(control_service.ListControlsResponse())
        response = await client.list_controls(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_controls_flattened_error_async():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_controls(control_service.ListControlsRequest(), parent='parent_value')

def test_list_controls_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_controls), '__call__') as call:
        call.side_effect = (control_service.ListControlsResponse(controls=[control.Control(), control.Control(), control.Control()], next_page_token='abc'), control_service.ListControlsResponse(controls=[], next_page_token='def'), control_service.ListControlsResponse(controls=[control.Control()], next_page_token='ghi'), control_service.ListControlsResponse(controls=[control.Control(), control.Control()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_controls(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, control.Control) for i in results))

def test_list_controls_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_controls), '__call__') as call:
        call.side_effect = (control_service.ListControlsResponse(controls=[control.Control(), control.Control(), control.Control()], next_page_token='abc'), control_service.ListControlsResponse(controls=[], next_page_token='def'), control_service.ListControlsResponse(controls=[control.Control()], next_page_token='ghi'), control_service.ListControlsResponse(controls=[control.Control(), control.Control()]), RuntimeError)
        pages = list(client.list_controls(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_controls_async_pager():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_controls), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (control_service.ListControlsResponse(controls=[control.Control(), control.Control(), control.Control()], next_page_token='abc'), control_service.ListControlsResponse(controls=[], next_page_token='def'), control_service.ListControlsResponse(controls=[control.Control()], next_page_token='ghi'), control_service.ListControlsResponse(controls=[control.Control(), control.Control()]), RuntimeError)
        async_pager = await client.list_controls(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, control.Control) for i in responses))

@pytest.mark.asyncio
async def test_list_controls_async_pages():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_controls), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (control_service.ListControlsResponse(controls=[control.Control(), control.Control(), control.Control()], next_page_token='abc'), control_service.ListControlsResponse(controls=[], next_page_token='def'), control_service.ListControlsResponse(controls=[control.Control()], next_page_token='ghi'), control_service.ListControlsResponse(controls=[control.Control(), control.Control()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_controls(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [control_service.CreateControlRequest, dict])
def test_create_control_rest(request_type):
    if False:
        return 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3'}
    request_init['control'] = {'rule': {'boost_action': {'boost': 0.551, 'products_filter': 'products_filter_value'}, 'redirect_action': {'redirect_uri': 'redirect_uri_value'}, 'oneway_synonyms_action': {'query_terms': ['query_terms_value1', 'query_terms_value2'], 'synonyms': ['synonyms_value1', 'synonyms_value2'], 'oneway_terms': ['oneway_terms_value1', 'oneway_terms_value2']}, 'do_not_associate_action': {'query_terms': ['query_terms_value1', 'query_terms_value2'], 'do_not_associate_terms': ['do_not_associate_terms_value1', 'do_not_associate_terms_value2'], 'terms': ['terms_value1', 'terms_value2']}, 'replacement_action': {'query_terms': ['query_terms_value1', 'query_terms_value2'], 'replacement_term': 'replacement_term_value', 'term': 'term_value'}, 'ignore_action': {'ignore_terms': ['ignore_terms_value1', 'ignore_terms_value2']}, 'filter_action': {'filter': 'filter_value'}, 'twoway_synonyms_action': {'synonyms': ['synonyms_value1', 'synonyms_value2']}, 'condition': {'query_terms': [{'value': 'value_value', 'full_match': True}], 'active_time_range': [{'start_time': {'seconds': 751, 'nanos': 543}, 'end_time': {}}]}}, 'name': 'name_value', 'display_name': 'display_name_value', 'associated_serving_config_ids': ['associated_serving_config_ids_value1', 'associated_serving_config_ids_value2'], 'solution_types': [1], 'search_solution_use_case': [1]}
    test_field = control_service.CreateControlRequest.meta.fields['control']

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
    for (field, value) in request_init['control'].items():
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
                for i in range(0, len(request_init['control'][field])):
                    del request_init['control'][field][i][subfield]
            else:
                del request_init['control'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcr_control.Control(name='name_value', display_name='display_name_value', associated_serving_config_ids=['associated_serving_config_ids_value'], solution_types=[common.SolutionType.SOLUTION_TYPE_RECOMMENDATION], search_solution_use_case=[common.SearchSolutionUseCase.SEARCH_SOLUTION_USE_CASE_SEARCH])
        response_value = Response()
        response_value.status_code = 200
        return_value = gcr_control.Control.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_control(request)
    assert isinstance(response, gcr_control.Control)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.associated_serving_config_ids == ['associated_serving_config_ids_value']
    assert response.solution_types == [common.SolutionType.SOLUTION_TYPE_RECOMMENDATION]
    assert response.search_solution_use_case == [common.SearchSolutionUseCase.SEARCH_SOLUTION_USE_CASE_SEARCH]

def test_create_control_rest_required_fields(request_type=control_service.CreateControlRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ControlServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['control_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'controlId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_control._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'controlId' in jsonified_request
    assert jsonified_request['controlId'] == request_init['control_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['controlId'] = 'control_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_control._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('control_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'controlId' in jsonified_request
    assert jsonified_request['controlId'] == 'control_id_value'
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcr_control.Control()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcr_control.Control.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_control(request)
            expected_params = [('controlId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_control_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ControlServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_control._get_unset_required_fields({})
    assert set(unset_fields) == set(('controlId',)) & set(('parent', 'control', 'controlId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_control_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ControlServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ControlServiceRestInterceptor())
    client = ControlServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ControlServiceRestInterceptor, 'post_create_control') as post, mock.patch.object(transports.ControlServiceRestInterceptor, 'pre_create_control') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = control_service.CreateControlRequest.pb(control_service.CreateControlRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcr_control.Control.to_json(gcr_control.Control())
        request = control_service.CreateControlRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcr_control.Control()
        client.create_control(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_control_rest_bad_request(transport: str='rest', request_type=control_service.CreateControlRequest):
    if False:
        print('Hello World!')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_control(request)

def test_create_control_rest_flattened():
    if False:
        return 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcr_control.Control()
        sample_request = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3'}
        mock_args = dict(parent='parent_value', control=gcr_control.Control(rule=common.Rule(boost_action=common.Rule.BoostAction(boost=0.551))), control_id='control_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcr_control.Control.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_control(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*/catalogs/*}/controls' % client.transport._host, args[1])

def test_create_control_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_control(control_service.CreateControlRequest(), parent='parent_value', control=gcr_control.Control(rule=common.Rule(boost_action=common.Rule.BoostAction(boost=0.551))), control_id='control_id_value')

def test_create_control_rest_error():
    if False:
        print('Hello World!')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [control_service.DeleteControlRequest, dict])
def test_delete_control_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/controls/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_control(request)
    assert response is None

def test_delete_control_rest_required_fields(request_type=control_service.DeleteControlRequest):
    if False:
        return 10
    transport_class = transports.ControlServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_control._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_control._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_control(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_control_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ControlServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_control._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_control_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ControlServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ControlServiceRestInterceptor())
    client = ControlServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ControlServiceRestInterceptor, 'pre_delete_control') as pre:
        pre.assert_not_called()
        pb_message = control_service.DeleteControlRequest.pb(control_service.DeleteControlRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = control_service.DeleteControlRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_control(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_control_rest_bad_request(transport: str='rest', request_type=control_service.DeleteControlRequest):
    if False:
        print('Hello World!')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/controls/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_control(request)

def test_delete_control_rest_flattened():
    if False:
        print('Hello World!')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/controls/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_control(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/catalogs/*/controls/*}' % client.transport._host, args[1])

def test_delete_control_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_control(control_service.DeleteControlRequest(), name='name_value')

def test_delete_control_rest_error():
    if False:
        print('Hello World!')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [control_service.UpdateControlRequest, dict])
def test_update_control_rest(request_type):
    if False:
        print('Hello World!')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'control': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/controls/sample4'}}
    request_init['control'] = {'rule': {'boost_action': {'boost': 0.551, 'products_filter': 'products_filter_value'}, 'redirect_action': {'redirect_uri': 'redirect_uri_value'}, 'oneway_synonyms_action': {'query_terms': ['query_terms_value1', 'query_terms_value2'], 'synonyms': ['synonyms_value1', 'synonyms_value2'], 'oneway_terms': ['oneway_terms_value1', 'oneway_terms_value2']}, 'do_not_associate_action': {'query_terms': ['query_terms_value1', 'query_terms_value2'], 'do_not_associate_terms': ['do_not_associate_terms_value1', 'do_not_associate_terms_value2'], 'terms': ['terms_value1', 'terms_value2']}, 'replacement_action': {'query_terms': ['query_terms_value1', 'query_terms_value2'], 'replacement_term': 'replacement_term_value', 'term': 'term_value'}, 'ignore_action': {'ignore_terms': ['ignore_terms_value1', 'ignore_terms_value2']}, 'filter_action': {'filter': 'filter_value'}, 'twoway_synonyms_action': {'synonyms': ['synonyms_value1', 'synonyms_value2']}, 'condition': {'query_terms': [{'value': 'value_value', 'full_match': True}], 'active_time_range': [{'start_time': {'seconds': 751, 'nanos': 543}, 'end_time': {}}]}}, 'name': 'projects/sample1/locations/sample2/catalogs/sample3/controls/sample4', 'display_name': 'display_name_value', 'associated_serving_config_ids': ['associated_serving_config_ids_value1', 'associated_serving_config_ids_value2'], 'solution_types': [1], 'search_solution_use_case': [1]}
    test_field = control_service.UpdateControlRequest.meta.fields['control']

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
    for (field, value) in request_init['control'].items():
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
                for i in range(0, len(request_init['control'][field])):
                    del request_init['control'][field][i][subfield]
            else:
                del request_init['control'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcr_control.Control(name='name_value', display_name='display_name_value', associated_serving_config_ids=['associated_serving_config_ids_value'], solution_types=[common.SolutionType.SOLUTION_TYPE_RECOMMENDATION], search_solution_use_case=[common.SearchSolutionUseCase.SEARCH_SOLUTION_USE_CASE_SEARCH])
        response_value = Response()
        response_value.status_code = 200
        return_value = gcr_control.Control.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_control(request)
    assert isinstance(response, gcr_control.Control)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.associated_serving_config_ids == ['associated_serving_config_ids_value']
    assert response.solution_types == [common.SolutionType.SOLUTION_TYPE_RECOMMENDATION]
    assert response.search_solution_use_case == [common.SearchSolutionUseCase.SEARCH_SOLUTION_USE_CASE_SEARCH]

def test_update_control_rest_required_fields(request_type=control_service.UpdateControlRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ControlServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_control._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_control._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcr_control.Control()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcr_control.Control.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_control(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_control_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ControlServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_control._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('control',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_control_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ControlServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ControlServiceRestInterceptor())
    client = ControlServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ControlServiceRestInterceptor, 'post_update_control') as post, mock.patch.object(transports.ControlServiceRestInterceptor, 'pre_update_control') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = control_service.UpdateControlRequest.pb(control_service.UpdateControlRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcr_control.Control.to_json(gcr_control.Control())
        request = control_service.UpdateControlRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcr_control.Control()
        client.update_control(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_control_rest_bad_request(transport: str='rest', request_type=control_service.UpdateControlRequest):
    if False:
        print('Hello World!')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'control': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/controls/sample4'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_control(request)

def test_update_control_rest_flattened():
    if False:
        print('Hello World!')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcr_control.Control()
        sample_request = {'control': {'name': 'projects/sample1/locations/sample2/catalogs/sample3/controls/sample4'}}
        mock_args = dict(control=gcr_control.Control(rule=common.Rule(boost_action=common.Rule.BoostAction(boost=0.551))), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcr_control.Control.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_control(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{control.name=projects/*/locations/*/catalogs/*/controls/*}' % client.transport._host, args[1])

def test_update_control_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_control(control_service.UpdateControlRequest(), control=gcr_control.Control(rule=common.Rule(boost_action=common.Rule.BoostAction(boost=0.551))), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_control_rest_error():
    if False:
        print('Hello World!')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [control_service.GetControlRequest, dict])
def test_get_control_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/controls/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = control.Control(name='name_value', display_name='display_name_value', associated_serving_config_ids=['associated_serving_config_ids_value'], solution_types=[common.SolutionType.SOLUTION_TYPE_RECOMMENDATION], search_solution_use_case=[common.SearchSolutionUseCase.SEARCH_SOLUTION_USE_CASE_SEARCH])
        response_value = Response()
        response_value.status_code = 200
        return_value = control.Control.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_control(request)
    assert isinstance(response, control.Control)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.associated_serving_config_ids == ['associated_serving_config_ids_value']
    assert response.solution_types == [common.SolutionType.SOLUTION_TYPE_RECOMMENDATION]
    assert response.search_solution_use_case == [common.SearchSolutionUseCase.SEARCH_SOLUTION_USE_CASE_SEARCH]

def test_get_control_rest_required_fields(request_type=control_service.GetControlRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ControlServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_control._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_control._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = control.Control()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = control.Control.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_control(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_control_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ControlServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_control._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_control_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ControlServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ControlServiceRestInterceptor())
    client = ControlServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ControlServiceRestInterceptor, 'post_get_control') as post, mock.patch.object(transports.ControlServiceRestInterceptor, 'pre_get_control') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = control_service.GetControlRequest.pb(control_service.GetControlRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = control.Control.to_json(control.Control())
        request = control_service.GetControlRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = control.Control()
        client.get_control(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_control_rest_bad_request(transport: str='rest', request_type=control_service.GetControlRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/controls/sample4'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_control(request)

def test_get_control_rest_flattened():
    if False:
        while True:
            i = 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = control.Control()
        sample_request = {'name': 'projects/sample1/locations/sample2/catalogs/sample3/controls/sample4'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = control.Control.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_control(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/locations/*/catalogs/*/controls/*}' % client.transport._host, args[1])

def test_get_control_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_control(control_service.GetControlRequest(), name='name_value')

def test_get_control_rest_error():
    if False:
        i = 10
        return i + 15
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [control_service.ListControlsRequest, dict])
def test_list_controls_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = control_service.ListControlsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = control_service.ListControlsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_controls(request)
    assert isinstance(response, pagers.ListControlsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_controls_rest_required_fields(request_type=control_service.ListControlsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ControlServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_controls._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_controls._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = control_service.ListControlsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = control_service.ListControlsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_controls(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_controls_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ControlServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_controls._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_controls_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ControlServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ControlServiceRestInterceptor())
    client = ControlServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ControlServiceRestInterceptor, 'post_list_controls') as post, mock.patch.object(transports.ControlServiceRestInterceptor, 'pre_list_controls') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = control_service.ListControlsRequest.pb(control_service.ListControlsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = control_service.ListControlsResponse.to_json(control_service.ListControlsResponse())
        request = control_service.ListControlsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = control_service.ListControlsResponse()
        client.list_controls(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_controls_rest_bad_request(transport: str='rest', request_type=control_service.ListControlsRequest):
    if False:
        i = 10
        return i + 15
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_controls(request)

def test_list_controls_rest_flattened():
    if False:
        while True:
            i = 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = control_service.ListControlsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = control_service.ListControlsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_controls(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*/locations/*/catalogs/*}/controls' % client.transport._host, args[1])

def test_list_controls_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_controls(control_service.ListControlsRequest(), parent='parent_value')

def test_list_controls_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (control_service.ListControlsResponse(controls=[control.Control(), control.Control(), control.Control()], next_page_token='abc'), control_service.ListControlsResponse(controls=[], next_page_token='def'), control_service.ListControlsResponse(controls=[control.Control()], next_page_token='ghi'), control_service.ListControlsResponse(controls=[control.Control(), control.Control()]))
        response = response + response
        response = tuple((control_service.ListControlsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/catalogs/sample3'}
        pager = client.list_controls(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, control.Control) for i in results))
        pages = list(client.list_controls(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        print('Hello World!')
    transport = transports.ControlServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.ControlServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ControlServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.ControlServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ControlServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ControlServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.ControlServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ControlServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.ControlServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = ControlServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        print('Hello World!')
    transport = transports.ControlServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.ControlServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.ControlServiceGrpcTransport, transports.ControlServiceGrpcAsyncIOTransport, transports.ControlServiceRestTransport])
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
        for i in range(10):
            print('nop')
    transport = ControlServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.ControlServiceGrpcTransport)

def test_control_service_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.ControlServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_control_service_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.retail_v2.services.control_service.transports.ControlServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.ControlServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_control', 'delete_control', 'update_control', 'get_control', 'list_controls', 'get_operation', 'list_operations')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_control_service_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.retail_v2.services.control_service.transports.ControlServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ControlServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_control_service_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.retail_v2.services.control_service.transports.ControlServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ControlServiceTransport()
        adc.assert_called_once()

def test_control_service_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        ControlServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.ControlServiceGrpcTransport, transports.ControlServiceGrpcAsyncIOTransport])
def test_control_service_transport_auth_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.ControlServiceGrpcTransport, transports.ControlServiceGrpcAsyncIOTransport, transports.ControlServiceRestTransport])
def test_control_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.ControlServiceGrpcTransport, grpc_helpers), (transports.ControlServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_control_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('retail.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='retail.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.ControlServiceGrpcTransport, transports.ControlServiceGrpcAsyncIOTransport])
def test_control_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_control_service_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.ControlServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_control_service_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='retail.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('retail.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://retail.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_control_service_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='retail.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('retail.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://retail.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_control_service_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = ControlServiceClient(credentials=creds1, transport=transport_name)
    client2 = ControlServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_control._session
    session2 = client2.transport.create_control._session
    assert session1 != session2
    session1 = client1.transport.delete_control._session
    session2 = client2.transport.delete_control._session
    assert session1 != session2
    session1 = client1.transport.update_control._session
    session2 = client2.transport.update_control._session
    assert session1 != session2
    session1 = client1.transport.get_control._session
    session2 = client2.transport.get_control._session
    assert session1 != session2
    session1 = client1.transport.list_controls._session
    session2 = client2.transport.list_controls._session
    assert session1 != session2

def test_control_service_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ControlServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_control_service_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ControlServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.ControlServiceGrpcTransport, transports.ControlServiceGrpcAsyncIOTransport])
def test_control_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.ControlServiceGrpcTransport, transports.ControlServiceGrpcAsyncIOTransport])
def test_control_service_transport_channel_mtls_with_adc(transport_class):
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

def test_catalog_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    catalog = 'whelk'
    expected = 'projects/{project}/locations/{location}/catalogs/{catalog}'.format(project=project, location=location, catalog=catalog)
    actual = ControlServiceClient.catalog_path(project, location, catalog)
    assert expected == actual

def test_parse_catalog_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'octopus', 'location': 'oyster', 'catalog': 'nudibranch'}
    path = ControlServiceClient.catalog_path(**expected)
    actual = ControlServiceClient.parse_catalog_path(path)
    assert expected == actual

def test_control_path():
    if False:
        i = 10
        return i + 15
    project = 'cuttlefish'
    location = 'mussel'
    catalog = 'winkle'
    control = 'nautilus'
    expected = 'projects/{project}/locations/{location}/catalogs/{catalog}/controls/{control}'.format(project=project, location=location, catalog=catalog, control=control)
    actual = ControlServiceClient.control_path(project, location, catalog, control)
    assert expected == actual

def test_parse_control_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'scallop', 'location': 'abalone', 'catalog': 'squid', 'control': 'clam'}
    path = ControlServiceClient.control_path(**expected)
    actual = ControlServiceClient.parse_control_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = ControlServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'octopus'}
    path = ControlServiceClient.common_billing_account_path(**expected)
    actual = ControlServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = ControlServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'nudibranch'}
    path = ControlServiceClient.common_folder_path(**expected)
    actual = ControlServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = ControlServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'mussel'}
    path = ControlServiceClient.common_organization_path(**expected)
    actual = ControlServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = ControlServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'nautilus'}
    path = ControlServiceClient.common_project_path(**expected)
    actual = ControlServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = ControlServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'squid', 'location': 'clam'}
    path = ControlServiceClient.common_location_path(**expected)
    actual = ControlServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        i = 10
        return i + 15
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.ControlServiceTransport, '_prep_wrapped_messages') as prep:
        client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.ControlServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = ControlServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        while True:
            i = 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        return 10
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = ControlServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        while True:
            i = 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = ControlServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(ControlServiceClient, transports.ControlServiceGrpcTransport), (ControlServiceAsyncClient, transports.ControlServiceGrpcAsyncIOTransport)])
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