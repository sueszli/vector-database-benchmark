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
from google.cloud.support_v2.services.case_service import CaseServiceAsyncClient, CaseServiceClient, pagers, transports
from google.cloud.support_v2.types import actor
from google.cloud.support_v2.types import case
from google.cloud.support_v2.types import case as gcs_case
from google.cloud.support_v2.types import case_service, escalation

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
        return 10
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert CaseServiceClient._get_default_mtls_endpoint(None) is None
    assert CaseServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert CaseServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert CaseServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert CaseServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert CaseServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(CaseServiceClient, 'grpc'), (CaseServiceAsyncClient, 'grpc_asyncio'), (CaseServiceClient, 'rest')])
def test_case_service_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('cloudsupport.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudsupport.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.CaseServiceGrpcTransport, 'grpc'), (transports.CaseServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.CaseServiceRestTransport, 'rest')])
def test_case_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(CaseServiceClient, 'grpc'), (CaseServiceAsyncClient, 'grpc_asyncio'), (CaseServiceClient, 'rest')])
def test_case_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('cloudsupport.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudsupport.googleapis.com')

def test_case_service_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = CaseServiceClient.get_transport_class()
    available_transports = [transports.CaseServiceGrpcTransport, transports.CaseServiceRestTransport]
    assert transport in available_transports
    transport = CaseServiceClient.get_transport_class('grpc')
    assert transport == transports.CaseServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CaseServiceClient, transports.CaseServiceGrpcTransport, 'grpc'), (CaseServiceAsyncClient, transports.CaseServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (CaseServiceClient, transports.CaseServiceRestTransport, 'rest')])
@mock.patch.object(CaseServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CaseServiceClient))
@mock.patch.object(CaseServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CaseServiceAsyncClient))
def test_case_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(CaseServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(CaseServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(CaseServiceClient, transports.CaseServiceGrpcTransport, 'grpc', 'true'), (CaseServiceAsyncClient, transports.CaseServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (CaseServiceClient, transports.CaseServiceGrpcTransport, 'grpc', 'false'), (CaseServiceAsyncClient, transports.CaseServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (CaseServiceClient, transports.CaseServiceRestTransport, 'rest', 'true'), (CaseServiceClient, transports.CaseServiceRestTransport, 'rest', 'false')])
@mock.patch.object(CaseServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CaseServiceClient))
@mock.patch.object(CaseServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CaseServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_case_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [CaseServiceClient, CaseServiceAsyncClient])
@mock.patch.object(CaseServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CaseServiceClient))
@mock.patch.object(CaseServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CaseServiceAsyncClient))
def test_case_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CaseServiceClient, transports.CaseServiceGrpcTransport, 'grpc'), (CaseServiceAsyncClient, transports.CaseServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (CaseServiceClient, transports.CaseServiceRestTransport, 'rest')])
def test_case_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CaseServiceClient, transports.CaseServiceGrpcTransport, 'grpc', grpc_helpers), (CaseServiceAsyncClient, transports.CaseServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (CaseServiceClient, transports.CaseServiceRestTransport, 'rest', None)])
def test_case_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_case_service_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.support_v2.services.case_service.transports.CaseServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = CaseServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CaseServiceClient, transports.CaseServiceGrpcTransport, 'grpc', grpc_helpers), (CaseServiceAsyncClient, transports.CaseServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_case_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('cloudsupport.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='cloudsupport.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [case_service.GetCaseRequest, dict])
def test_get_case(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_case), '__call__') as call:
        call.return_value = case.Case(name='name_value', display_name='display_name_value', description='description_value', time_zone='time_zone_value', subscriber_email_addresses=['subscriber_email_addresses_value'], state=case.Case.State.NEW, contact_email='contact_email_value', escalated=True, test_case=True, language_code='language_code_value', priority=case.Case.Priority.P0)
        response = client.get_case(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.GetCaseRequest()
    assert isinstance(response, case.Case)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.time_zone == 'time_zone_value'
    assert response.subscriber_email_addresses == ['subscriber_email_addresses_value']
    assert response.state == case.Case.State.NEW
    assert response.contact_email == 'contact_email_value'
    assert response.escalated is True
    assert response.test_case is True
    assert response.language_code == 'language_code_value'
    assert response.priority == case.Case.Priority.P0

def test_get_case_empty_call():
    if False:
        while True:
            i = 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_case), '__call__') as call:
        client.get_case()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.GetCaseRequest()

@pytest.mark.asyncio
async def test_get_case_async(transport: str='grpc_asyncio', request_type=case_service.GetCaseRequest):
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_case), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(case.Case(name='name_value', display_name='display_name_value', description='description_value', time_zone='time_zone_value', subscriber_email_addresses=['subscriber_email_addresses_value'], state=case.Case.State.NEW, contact_email='contact_email_value', escalated=True, test_case=True, language_code='language_code_value', priority=case.Case.Priority.P0))
        response = await client.get_case(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.GetCaseRequest()
    assert isinstance(response, case.Case)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.time_zone == 'time_zone_value'
    assert response.subscriber_email_addresses == ['subscriber_email_addresses_value']
    assert response.state == case.Case.State.NEW
    assert response.contact_email == 'contact_email_value'
    assert response.escalated is True
    assert response.test_case is True
    assert response.language_code == 'language_code_value'
    assert response.priority == case.Case.Priority.P0

@pytest.mark.asyncio
async def test_get_case_async_from_dict():
    await test_get_case_async(request_type=dict)

def test_get_case_field_headers():
    if False:
        print('Hello World!')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = case_service.GetCaseRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_case), '__call__') as call:
        call.return_value = case.Case()
        client.get_case(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_case_field_headers_async():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = case_service.GetCaseRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_case), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(case.Case())
        await client.get_case(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_case_flattened():
    if False:
        return 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_case), '__call__') as call:
        call.return_value = case.Case()
        client.get_case(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_case_flattened_error():
    if False:
        return 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_case(case_service.GetCaseRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_case_flattened_async():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_case), '__call__') as call:
        call.return_value = case.Case()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(case.Case())
        response = await client.get_case(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_case_flattened_error_async():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_case(case_service.GetCaseRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [case_service.ListCasesRequest, dict])
def test_list_cases(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_cases), '__call__') as call:
        call.return_value = case_service.ListCasesResponse(next_page_token='next_page_token_value')
        response = client.list_cases(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.ListCasesRequest()
    assert isinstance(response, pagers.ListCasesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_cases_empty_call():
    if False:
        while True:
            i = 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_cases), '__call__') as call:
        client.list_cases()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.ListCasesRequest()

@pytest.mark.asyncio
async def test_list_cases_async(transport: str='grpc_asyncio', request_type=case_service.ListCasesRequest):
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_cases), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(case_service.ListCasesResponse(next_page_token='next_page_token_value'))
        response = await client.list_cases(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.ListCasesRequest()
    assert isinstance(response, pagers.ListCasesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_cases_async_from_dict():
    await test_list_cases_async(request_type=dict)

def test_list_cases_field_headers():
    if False:
        while True:
            i = 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = case_service.ListCasesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_cases), '__call__') as call:
        call.return_value = case_service.ListCasesResponse()
        client.list_cases(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_cases_field_headers_async():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = case_service.ListCasesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_cases), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(case_service.ListCasesResponse())
        await client.list_cases(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_cases_flattened():
    if False:
        print('Hello World!')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_cases), '__call__') as call:
        call.return_value = case_service.ListCasesResponse()
        client.list_cases(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_cases_flattened_error():
    if False:
        print('Hello World!')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_cases(case_service.ListCasesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_cases_flattened_async():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_cases), '__call__') as call:
        call.return_value = case_service.ListCasesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(case_service.ListCasesResponse())
        response = await client.list_cases(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_cases_flattened_error_async():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_cases(case_service.ListCasesRequest(), parent='parent_value')

def test_list_cases_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_cases), '__call__') as call:
        call.side_effect = (case_service.ListCasesResponse(cases=[case.Case(), case.Case(), case.Case()], next_page_token='abc'), case_service.ListCasesResponse(cases=[], next_page_token='def'), case_service.ListCasesResponse(cases=[case.Case()], next_page_token='ghi'), case_service.ListCasesResponse(cases=[case.Case(), case.Case()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_cases(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, case.Case) for i in results))

def test_list_cases_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_cases), '__call__') as call:
        call.side_effect = (case_service.ListCasesResponse(cases=[case.Case(), case.Case(), case.Case()], next_page_token='abc'), case_service.ListCasesResponse(cases=[], next_page_token='def'), case_service.ListCasesResponse(cases=[case.Case()], next_page_token='ghi'), case_service.ListCasesResponse(cases=[case.Case(), case.Case()]), RuntimeError)
        pages = list(client.list_cases(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_cases_async_pager():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_cases), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (case_service.ListCasesResponse(cases=[case.Case(), case.Case(), case.Case()], next_page_token='abc'), case_service.ListCasesResponse(cases=[], next_page_token='def'), case_service.ListCasesResponse(cases=[case.Case()], next_page_token='ghi'), case_service.ListCasesResponse(cases=[case.Case(), case.Case()]), RuntimeError)
        async_pager = await client.list_cases(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, case.Case) for i in responses))

@pytest.mark.asyncio
async def test_list_cases_async_pages():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_cases), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (case_service.ListCasesResponse(cases=[case.Case(), case.Case(), case.Case()], next_page_token='abc'), case_service.ListCasesResponse(cases=[], next_page_token='def'), case_service.ListCasesResponse(cases=[case.Case()], next_page_token='ghi'), case_service.ListCasesResponse(cases=[case.Case(), case.Case()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_cases(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [case_service.SearchCasesRequest, dict])
def test_search_cases(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_cases), '__call__') as call:
        call.return_value = case_service.SearchCasesResponse(next_page_token='next_page_token_value')
        response = client.search_cases(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.SearchCasesRequest()
    assert isinstance(response, pagers.SearchCasesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_search_cases_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search_cases), '__call__') as call:
        client.search_cases()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.SearchCasesRequest()

@pytest.mark.asyncio
async def test_search_cases_async(transport: str='grpc_asyncio', request_type=case_service.SearchCasesRequest):
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_cases), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(case_service.SearchCasesResponse(next_page_token='next_page_token_value'))
        response = await client.search_cases(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.SearchCasesRequest()
    assert isinstance(response, pagers.SearchCasesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_search_cases_async_from_dict():
    await test_search_cases_async(request_type=dict)

def test_search_cases_field_headers():
    if False:
        while True:
            i = 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = case_service.SearchCasesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.search_cases), '__call__') as call:
        call.return_value = case_service.SearchCasesResponse()
        client.search_cases(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_search_cases_field_headers_async():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = case_service.SearchCasesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.search_cases), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(case_service.SearchCasesResponse())
        await client.search_cases(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_search_cases_pager(transport_name: str='grpc'):
    if False:
        return 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_cases), '__call__') as call:
        call.side_effect = (case_service.SearchCasesResponse(cases=[case.Case(), case.Case(), case.Case()], next_page_token='abc'), case_service.SearchCasesResponse(cases=[], next_page_token='def'), case_service.SearchCasesResponse(cases=[case.Case()], next_page_token='ghi'), case_service.SearchCasesResponse(cases=[case.Case(), case.Case()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.search_cases(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, case.Case) for i in results))

def test_search_cases_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_cases), '__call__') as call:
        call.side_effect = (case_service.SearchCasesResponse(cases=[case.Case(), case.Case(), case.Case()], next_page_token='abc'), case_service.SearchCasesResponse(cases=[], next_page_token='def'), case_service.SearchCasesResponse(cases=[case.Case()], next_page_token='ghi'), case_service.SearchCasesResponse(cases=[case.Case(), case.Case()]), RuntimeError)
        pages = list(client.search_cases(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_search_cases_async_pager():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_cases), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (case_service.SearchCasesResponse(cases=[case.Case(), case.Case(), case.Case()], next_page_token='abc'), case_service.SearchCasesResponse(cases=[], next_page_token='def'), case_service.SearchCasesResponse(cases=[case.Case()], next_page_token='ghi'), case_service.SearchCasesResponse(cases=[case.Case(), case.Case()]), RuntimeError)
        async_pager = await client.search_cases(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, case.Case) for i in responses))

@pytest.mark.asyncio
async def test_search_cases_async_pages():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_cases), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (case_service.SearchCasesResponse(cases=[case.Case(), case.Case(), case.Case()], next_page_token='abc'), case_service.SearchCasesResponse(cases=[], next_page_token='def'), case_service.SearchCasesResponse(cases=[case.Case()], next_page_token='ghi'), case_service.SearchCasesResponse(cases=[case.Case(), case.Case()]), RuntimeError)
        pages = []
        async for page_ in (await client.search_cases(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [case_service.CreateCaseRequest, dict])
def test_create_case(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_case), '__call__') as call:
        call.return_value = gcs_case.Case(name='name_value', display_name='display_name_value', description='description_value', time_zone='time_zone_value', subscriber_email_addresses=['subscriber_email_addresses_value'], state=gcs_case.Case.State.NEW, contact_email='contact_email_value', escalated=True, test_case=True, language_code='language_code_value', priority=gcs_case.Case.Priority.P0)
        response = client.create_case(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.CreateCaseRequest()
    assert isinstance(response, gcs_case.Case)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.time_zone == 'time_zone_value'
    assert response.subscriber_email_addresses == ['subscriber_email_addresses_value']
    assert response.state == gcs_case.Case.State.NEW
    assert response.contact_email == 'contact_email_value'
    assert response.escalated is True
    assert response.test_case is True
    assert response.language_code == 'language_code_value'
    assert response.priority == gcs_case.Case.Priority.P0

def test_create_case_empty_call():
    if False:
        return 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_case), '__call__') as call:
        client.create_case()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.CreateCaseRequest()

@pytest.mark.asyncio
async def test_create_case_async(transport: str='grpc_asyncio', request_type=case_service.CreateCaseRequest):
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_case), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_case.Case(name='name_value', display_name='display_name_value', description='description_value', time_zone='time_zone_value', subscriber_email_addresses=['subscriber_email_addresses_value'], state=gcs_case.Case.State.NEW, contact_email='contact_email_value', escalated=True, test_case=True, language_code='language_code_value', priority=gcs_case.Case.Priority.P0))
        response = await client.create_case(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.CreateCaseRequest()
    assert isinstance(response, gcs_case.Case)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.time_zone == 'time_zone_value'
    assert response.subscriber_email_addresses == ['subscriber_email_addresses_value']
    assert response.state == gcs_case.Case.State.NEW
    assert response.contact_email == 'contact_email_value'
    assert response.escalated is True
    assert response.test_case is True
    assert response.language_code == 'language_code_value'
    assert response.priority == gcs_case.Case.Priority.P0

@pytest.mark.asyncio
async def test_create_case_async_from_dict():
    await test_create_case_async(request_type=dict)

def test_create_case_field_headers():
    if False:
        return 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = case_service.CreateCaseRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_case), '__call__') as call:
        call.return_value = gcs_case.Case()
        client.create_case(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_case_field_headers_async():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = case_service.CreateCaseRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_case), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_case.Case())
        await client.create_case(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_case_flattened():
    if False:
        return 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_case), '__call__') as call:
        call.return_value = gcs_case.Case()
        client.create_case(parent='parent_value', case=gcs_case.Case(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].case
        mock_val = gcs_case.Case(name='name_value')
        assert arg == mock_val

def test_create_case_flattened_error():
    if False:
        return 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_case(case_service.CreateCaseRequest(), parent='parent_value', case=gcs_case.Case(name='name_value'))

@pytest.mark.asyncio
async def test_create_case_flattened_async():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_case), '__call__') as call:
        call.return_value = gcs_case.Case()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_case.Case())
        response = await client.create_case(parent='parent_value', case=gcs_case.Case(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].case
        mock_val = gcs_case.Case(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_case_flattened_error_async():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_case(case_service.CreateCaseRequest(), parent='parent_value', case=gcs_case.Case(name='name_value'))

@pytest.mark.parametrize('request_type', [case_service.UpdateCaseRequest, dict])
def test_update_case(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_case), '__call__') as call:
        call.return_value = gcs_case.Case(name='name_value', display_name='display_name_value', description='description_value', time_zone='time_zone_value', subscriber_email_addresses=['subscriber_email_addresses_value'], state=gcs_case.Case.State.NEW, contact_email='contact_email_value', escalated=True, test_case=True, language_code='language_code_value', priority=gcs_case.Case.Priority.P0)
        response = client.update_case(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.UpdateCaseRequest()
    assert isinstance(response, gcs_case.Case)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.time_zone == 'time_zone_value'
    assert response.subscriber_email_addresses == ['subscriber_email_addresses_value']
    assert response.state == gcs_case.Case.State.NEW
    assert response.contact_email == 'contact_email_value'
    assert response.escalated is True
    assert response.test_case is True
    assert response.language_code == 'language_code_value'
    assert response.priority == gcs_case.Case.Priority.P0

def test_update_case_empty_call():
    if False:
        return 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_case), '__call__') as call:
        client.update_case()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.UpdateCaseRequest()

@pytest.mark.asyncio
async def test_update_case_async(transport: str='grpc_asyncio', request_type=case_service.UpdateCaseRequest):
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_case), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_case.Case(name='name_value', display_name='display_name_value', description='description_value', time_zone='time_zone_value', subscriber_email_addresses=['subscriber_email_addresses_value'], state=gcs_case.Case.State.NEW, contact_email='contact_email_value', escalated=True, test_case=True, language_code='language_code_value', priority=gcs_case.Case.Priority.P0))
        response = await client.update_case(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.UpdateCaseRequest()
    assert isinstance(response, gcs_case.Case)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.time_zone == 'time_zone_value'
    assert response.subscriber_email_addresses == ['subscriber_email_addresses_value']
    assert response.state == gcs_case.Case.State.NEW
    assert response.contact_email == 'contact_email_value'
    assert response.escalated is True
    assert response.test_case is True
    assert response.language_code == 'language_code_value'
    assert response.priority == gcs_case.Case.Priority.P0

@pytest.mark.asyncio
async def test_update_case_async_from_dict():
    await test_update_case_async(request_type=dict)

def test_update_case_field_headers():
    if False:
        print('Hello World!')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = case_service.UpdateCaseRequest()
    request.case.name = 'name_value'
    with mock.patch.object(type(client.transport.update_case), '__call__') as call:
        call.return_value = gcs_case.Case()
        client.update_case(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'case.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_case_field_headers_async():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = case_service.UpdateCaseRequest()
    request.case.name = 'name_value'
    with mock.patch.object(type(client.transport.update_case), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_case.Case())
        await client.update_case(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'case.name=name_value') in kw['metadata']

def test_update_case_flattened():
    if False:
        print('Hello World!')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_case), '__call__') as call:
        call.return_value = gcs_case.Case()
        client.update_case(case=gcs_case.Case(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].case
        mock_val = gcs_case.Case(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_case_flattened_error():
    if False:
        i = 10
        return i + 15
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_case(case_service.UpdateCaseRequest(), case=gcs_case.Case(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_case_flattened_async():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_case), '__call__') as call:
        call.return_value = gcs_case.Case()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcs_case.Case())
        response = await client.update_case(case=gcs_case.Case(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].case
        mock_val = gcs_case.Case(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_case_flattened_error_async():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_case(case_service.UpdateCaseRequest(), case=gcs_case.Case(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [case_service.EscalateCaseRequest, dict])
def test_escalate_case(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.escalate_case), '__call__') as call:
        call.return_value = case.Case(name='name_value', display_name='display_name_value', description='description_value', time_zone='time_zone_value', subscriber_email_addresses=['subscriber_email_addresses_value'], state=case.Case.State.NEW, contact_email='contact_email_value', escalated=True, test_case=True, language_code='language_code_value', priority=case.Case.Priority.P0)
        response = client.escalate_case(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.EscalateCaseRequest()
    assert isinstance(response, case.Case)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.time_zone == 'time_zone_value'
    assert response.subscriber_email_addresses == ['subscriber_email_addresses_value']
    assert response.state == case.Case.State.NEW
    assert response.contact_email == 'contact_email_value'
    assert response.escalated is True
    assert response.test_case is True
    assert response.language_code == 'language_code_value'
    assert response.priority == case.Case.Priority.P0

def test_escalate_case_empty_call():
    if False:
        i = 10
        return i + 15
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.escalate_case), '__call__') as call:
        client.escalate_case()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.EscalateCaseRequest()

@pytest.mark.asyncio
async def test_escalate_case_async(transport: str='grpc_asyncio', request_type=case_service.EscalateCaseRequest):
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.escalate_case), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(case.Case(name='name_value', display_name='display_name_value', description='description_value', time_zone='time_zone_value', subscriber_email_addresses=['subscriber_email_addresses_value'], state=case.Case.State.NEW, contact_email='contact_email_value', escalated=True, test_case=True, language_code='language_code_value', priority=case.Case.Priority.P0))
        response = await client.escalate_case(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.EscalateCaseRequest()
    assert isinstance(response, case.Case)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.time_zone == 'time_zone_value'
    assert response.subscriber_email_addresses == ['subscriber_email_addresses_value']
    assert response.state == case.Case.State.NEW
    assert response.contact_email == 'contact_email_value'
    assert response.escalated is True
    assert response.test_case is True
    assert response.language_code == 'language_code_value'
    assert response.priority == case.Case.Priority.P0

@pytest.mark.asyncio
async def test_escalate_case_async_from_dict():
    await test_escalate_case_async(request_type=dict)

def test_escalate_case_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = case_service.EscalateCaseRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.escalate_case), '__call__') as call:
        call.return_value = case.Case()
        client.escalate_case(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_escalate_case_field_headers_async():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = case_service.EscalateCaseRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.escalate_case), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(case.Case())
        await client.escalate_case(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [case_service.CloseCaseRequest, dict])
def test_close_case(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.close_case), '__call__') as call:
        call.return_value = case.Case(name='name_value', display_name='display_name_value', description='description_value', time_zone='time_zone_value', subscriber_email_addresses=['subscriber_email_addresses_value'], state=case.Case.State.NEW, contact_email='contact_email_value', escalated=True, test_case=True, language_code='language_code_value', priority=case.Case.Priority.P0)
        response = client.close_case(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.CloseCaseRequest()
    assert isinstance(response, case.Case)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.time_zone == 'time_zone_value'
    assert response.subscriber_email_addresses == ['subscriber_email_addresses_value']
    assert response.state == case.Case.State.NEW
    assert response.contact_email == 'contact_email_value'
    assert response.escalated is True
    assert response.test_case is True
    assert response.language_code == 'language_code_value'
    assert response.priority == case.Case.Priority.P0

def test_close_case_empty_call():
    if False:
        i = 10
        return i + 15
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.close_case), '__call__') as call:
        client.close_case()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.CloseCaseRequest()

@pytest.mark.asyncio
async def test_close_case_async(transport: str='grpc_asyncio', request_type=case_service.CloseCaseRequest):
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.close_case), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(case.Case(name='name_value', display_name='display_name_value', description='description_value', time_zone='time_zone_value', subscriber_email_addresses=['subscriber_email_addresses_value'], state=case.Case.State.NEW, contact_email='contact_email_value', escalated=True, test_case=True, language_code='language_code_value', priority=case.Case.Priority.P0))
        response = await client.close_case(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.CloseCaseRequest()
    assert isinstance(response, case.Case)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.time_zone == 'time_zone_value'
    assert response.subscriber_email_addresses == ['subscriber_email_addresses_value']
    assert response.state == case.Case.State.NEW
    assert response.contact_email == 'contact_email_value'
    assert response.escalated is True
    assert response.test_case is True
    assert response.language_code == 'language_code_value'
    assert response.priority == case.Case.Priority.P0

@pytest.mark.asyncio
async def test_close_case_async_from_dict():
    await test_close_case_async(request_type=dict)

def test_close_case_field_headers():
    if False:
        i = 10
        return i + 15
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = case_service.CloseCaseRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.close_case), '__call__') as call:
        call.return_value = case.Case()
        client.close_case(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_close_case_field_headers_async():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = case_service.CloseCaseRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.close_case), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(case.Case())
        await client.close_case(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.parametrize('request_type', [case_service.SearchCaseClassificationsRequest, dict])
def test_search_case_classifications(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_case_classifications), '__call__') as call:
        call.return_value = case_service.SearchCaseClassificationsResponse(next_page_token='next_page_token_value')
        response = client.search_case_classifications(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.SearchCaseClassificationsRequest()
    assert isinstance(response, pagers.SearchCaseClassificationsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_search_case_classifications_empty_call():
    if False:
        return 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.search_case_classifications), '__call__') as call:
        client.search_case_classifications()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.SearchCaseClassificationsRequest()

@pytest.mark.asyncio
async def test_search_case_classifications_async(transport: str='grpc_asyncio', request_type=case_service.SearchCaseClassificationsRequest):
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.search_case_classifications), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(case_service.SearchCaseClassificationsResponse(next_page_token='next_page_token_value'))
        response = await client.search_case_classifications(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == case_service.SearchCaseClassificationsRequest()
    assert isinstance(response, pagers.SearchCaseClassificationsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_search_case_classifications_async_from_dict():
    await test_search_case_classifications_async(request_type=dict)

def test_search_case_classifications_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_case_classifications), '__call__') as call:
        call.side_effect = (case_service.SearchCaseClassificationsResponse(case_classifications=[case.CaseClassification(), case.CaseClassification(), case.CaseClassification()], next_page_token='abc'), case_service.SearchCaseClassificationsResponse(case_classifications=[], next_page_token='def'), case_service.SearchCaseClassificationsResponse(case_classifications=[case.CaseClassification()], next_page_token='ghi'), case_service.SearchCaseClassificationsResponse(case_classifications=[case.CaseClassification(), case.CaseClassification()]), RuntimeError)
        metadata = ()
        pager = client.search_case_classifications(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, case.CaseClassification) for i in results))

def test_search_case_classifications_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.search_case_classifications), '__call__') as call:
        call.side_effect = (case_service.SearchCaseClassificationsResponse(case_classifications=[case.CaseClassification(), case.CaseClassification(), case.CaseClassification()], next_page_token='abc'), case_service.SearchCaseClassificationsResponse(case_classifications=[], next_page_token='def'), case_service.SearchCaseClassificationsResponse(case_classifications=[case.CaseClassification()], next_page_token='ghi'), case_service.SearchCaseClassificationsResponse(case_classifications=[case.CaseClassification(), case.CaseClassification()]), RuntimeError)
        pages = list(client.search_case_classifications(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_search_case_classifications_async_pager():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_case_classifications), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (case_service.SearchCaseClassificationsResponse(case_classifications=[case.CaseClassification(), case.CaseClassification(), case.CaseClassification()], next_page_token='abc'), case_service.SearchCaseClassificationsResponse(case_classifications=[], next_page_token='def'), case_service.SearchCaseClassificationsResponse(case_classifications=[case.CaseClassification()], next_page_token='ghi'), case_service.SearchCaseClassificationsResponse(case_classifications=[case.CaseClassification(), case.CaseClassification()]), RuntimeError)
        async_pager = await client.search_case_classifications(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, case.CaseClassification) for i in responses))

@pytest.mark.asyncio
async def test_search_case_classifications_async_pages():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.search_case_classifications), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (case_service.SearchCaseClassificationsResponse(case_classifications=[case.CaseClassification(), case.CaseClassification(), case.CaseClassification()], next_page_token='abc'), case_service.SearchCaseClassificationsResponse(case_classifications=[], next_page_token='def'), case_service.SearchCaseClassificationsResponse(case_classifications=[case.CaseClassification()], next_page_token='ghi'), case_service.SearchCaseClassificationsResponse(case_classifications=[case.CaseClassification(), case.CaseClassification()]), RuntimeError)
        pages = []
        async for page_ in (await client.search_case_classifications(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [case_service.GetCaseRequest, dict])
def test_get_case_rest(request_type):
    if False:
        print('Hello World!')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/cases/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = case.Case(name='name_value', display_name='display_name_value', description='description_value', time_zone='time_zone_value', subscriber_email_addresses=['subscriber_email_addresses_value'], state=case.Case.State.NEW, contact_email='contact_email_value', escalated=True, test_case=True, language_code='language_code_value', priority=case.Case.Priority.P0)
        response_value = Response()
        response_value.status_code = 200
        return_value = case.Case.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_case(request)
    assert isinstance(response, case.Case)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.time_zone == 'time_zone_value'
    assert response.subscriber_email_addresses == ['subscriber_email_addresses_value']
    assert response.state == case.Case.State.NEW
    assert response.contact_email == 'contact_email_value'
    assert response.escalated is True
    assert response.test_case is True
    assert response.language_code == 'language_code_value'
    assert response.priority == case.Case.Priority.P0

def test_get_case_rest_required_fields(request_type=case_service.GetCaseRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CaseServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_case._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_case._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = case.Case()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = case.Case.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_case(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_case_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CaseServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_case._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_case_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CaseServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CaseServiceRestInterceptor())
    client = CaseServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CaseServiceRestInterceptor, 'post_get_case') as post, mock.patch.object(transports.CaseServiceRestInterceptor, 'pre_get_case') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = case_service.GetCaseRequest.pb(case_service.GetCaseRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = case.Case.to_json(case.Case())
        request = case_service.GetCaseRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = case.Case()
        client.get_case(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_case_rest_bad_request(transport: str='rest', request_type=case_service.GetCaseRequest):
    if False:
        print('Hello World!')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/cases/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_case(request)

def test_get_case_rest_flattened():
    if False:
        print('Hello World!')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = case.Case()
        sample_request = {'name': 'projects/sample1/cases/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = case.Case.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_case(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{name=projects/*/cases/*}' % client.transport._host, args[1])

def test_get_case_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_case(case_service.GetCaseRequest(), name='name_value')

def test_get_case_rest_error():
    if False:
        return 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [case_service.ListCasesRequest, dict])
def test_list_cases_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = case_service.ListCasesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = case_service.ListCasesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_cases(request)
    assert isinstance(response, pagers.ListCasesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_cases_rest_required_fields(request_type=case_service.ListCasesRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CaseServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_cases._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_cases._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = case_service.ListCasesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = case_service.ListCasesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_cases(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_cases_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CaseServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_cases._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_cases_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CaseServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CaseServiceRestInterceptor())
    client = CaseServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CaseServiceRestInterceptor, 'post_list_cases') as post, mock.patch.object(transports.CaseServiceRestInterceptor, 'pre_list_cases') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = case_service.ListCasesRequest.pb(case_service.ListCasesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = case_service.ListCasesResponse.to_json(case_service.ListCasesResponse())
        request = case_service.ListCasesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = case_service.ListCasesResponse()
        client.list_cases(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_cases_rest_bad_request(transport: str='rest', request_type=case_service.ListCasesRequest):
    if False:
        i = 10
        return i + 15
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_cases(request)

def test_list_cases_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = case_service.ListCasesResponse()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = case_service.ListCasesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_cases(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*}/cases' % client.transport._host, args[1])

def test_list_cases_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_cases(case_service.ListCasesRequest(), parent='parent_value')

def test_list_cases_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (case_service.ListCasesResponse(cases=[case.Case(), case.Case(), case.Case()], next_page_token='abc'), case_service.ListCasesResponse(cases=[], next_page_token='def'), case_service.ListCasesResponse(cases=[case.Case()], next_page_token='ghi'), case_service.ListCasesResponse(cases=[case.Case(), case.Case()]))
        response = response + response
        response = tuple((case_service.ListCasesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1'}
        pager = client.list_cases(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, case.Case) for i in results))
        pages = list(client.list_cases(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [case_service.SearchCasesRequest, dict])
def test_search_cases_rest(request_type):
    if False:
        print('Hello World!')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = case_service.SearchCasesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = case_service.SearchCasesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.search_cases(request)
    assert isinstance(response, pagers.SearchCasesPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_search_cases_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CaseServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CaseServiceRestInterceptor())
    client = CaseServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CaseServiceRestInterceptor, 'post_search_cases') as post, mock.patch.object(transports.CaseServiceRestInterceptor, 'pre_search_cases') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = case_service.SearchCasesRequest.pb(case_service.SearchCasesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = case_service.SearchCasesResponse.to_json(case_service.SearchCasesResponse())
        request = case_service.SearchCasesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = case_service.SearchCasesResponse()
        client.search_cases(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_search_cases_rest_bad_request(transport: str='rest', request_type=case_service.SearchCasesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.search_cases(request)

def test_search_cases_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (case_service.SearchCasesResponse(cases=[case.Case(), case.Case(), case.Case()], next_page_token='abc'), case_service.SearchCasesResponse(cases=[], next_page_token='def'), case_service.SearchCasesResponse(cases=[case.Case()], next_page_token='ghi'), case_service.SearchCasesResponse(cases=[case.Case(), case.Case()]))
        response = response + response
        response = tuple((case_service.SearchCasesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1'}
        pager = client.search_cases(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, case.Case) for i in results))
        pages = list(client.search_cases(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [case_service.CreateCaseRequest, dict])
def test_create_case_rest(request_type):
    if False:
        while True:
            i = 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request_init['case'] = {'name': 'name_value', 'display_name': 'display_name_value', 'description': 'description_value', 'classification': {'id': 'id_value', 'display_name': 'display_name_value'}, 'time_zone': 'time_zone_value', 'subscriber_email_addresses': ['subscriber_email_addresses_value1', 'subscriber_email_addresses_value2'], 'state': 1, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'creator': {'display_name': 'display_name_value', 'email': 'email_value', 'google_support': True}, 'contact_email': 'contact_email_value', 'escalated': True, 'test_case': True, 'language_code': 'language_code_value', 'priority': 1}
    test_field = case_service.CreateCaseRequest.meta.fields['case']

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
    for (field, value) in request_init['case'].items():
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
                for i in range(0, len(request_init['case'][field])):
                    del request_init['case'][field][i][subfield]
            else:
                del request_init['case'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_case.Case(name='name_value', display_name='display_name_value', description='description_value', time_zone='time_zone_value', subscriber_email_addresses=['subscriber_email_addresses_value'], state=gcs_case.Case.State.NEW, contact_email='contact_email_value', escalated=True, test_case=True, language_code='language_code_value', priority=gcs_case.Case.Priority.P0)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_case.Case.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_case(request)
    assert isinstance(response, gcs_case.Case)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.time_zone == 'time_zone_value'
    assert response.subscriber_email_addresses == ['subscriber_email_addresses_value']
    assert response.state == gcs_case.Case.State.NEW
    assert response.contact_email == 'contact_email_value'
    assert response.escalated is True
    assert response.test_case is True
    assert response.language_code == 'language_code_value'
    assert response.priority == gcs_case.Case.Priority.P0

def test_create_case_rest_required_fields(request_type=case_service.CreateCaseRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CaseServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_case._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_case._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcs_case.Case()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcs_case.Case.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_case(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_case_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CaseServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_case._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'case'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_case_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CaseServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CaseServiceRestInterceptor())
    client = CaseServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CaseServiceRestInterceptor, 'post_create_case') as post, mock.patch.object(transports.CaseServiceRestInterceptor, 'pre_create_case') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = case_service.CreateCaseRequest.pb(case_service.CreateCaseRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcs_case.Case.to_json(gcs_case.Case())
        request = case_service.CreateCaseRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcs_case.Case()
        client.create_case(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_case_rest_bad_request(transport: str='rest', request_type=case_service.CreateCaseRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_case(request)

def test_create_case_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_case.Case()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value', case=gcs_case.Case(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_case.Case.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_case(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*}/cases' % client.transport._host, args[1])

def test_create_case_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_case(case_service.CreateCaseRequest(), parent='parent_value', case=gcs_case.Case(name='name_value'))

def test_create_case_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [case_service.UpdateCaseRequest, dict])
def test_update_case_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'case': {'name': 'projects/sample1/cases/sample2'}}
    request_init['case'] = {'name': 'projects/sample1/cases/sample2', 'display_name': 'display_name_value', 'description': 'description_value', 'classification': {'id': 'id_value', 'display_name': 'display_name_value'}, 'time_zone': 'time_zone_value', 'subscriber_email_addresses': ['subscriber_email_addresses_value1', 'subscriber_email_addresses_value2'], 'state': 1, 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'creator': {'display_name': 'display_name_value', 'email': 'email_value', 'google_support': True}, 'contact_email': 'contact_email_value', 'escalated': True, 'test_case': True, 'language_code': 'language_code_value', 'priority': 1}
    test_field = case_service.UpdateCaseRequest.meta.fields['case']

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
    for (field, value) in request_init['case'].items():
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
                for i in range(0, len(request_init['case'][field])):
                    del request_init['case'][field][i][subfield]
            else:
                del request_init['case'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_case.Case(name='name_value', display_name='display_name_value', description='description_value', time_zone='time_zone_value', subscriber_email_addresses=['subscriber_email_addresses_value'], state=gcs_case.Case.State.NEW, contact_email='contact_email_value', escalated=True, test_case=True, language_code='language_code_value', priority=gcs_case.Case.Priority.P0)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_case.Case.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_case(request)
    assert isinstance(response, gcs_case.Case)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.time_zone == 'time_zone_value'
    assert response.subscriber_email_addresses == ['subscriber_email_addresses_value']
    assert response.state == gcs_case.Case.State.NEW
    assert response.contact_email == 'contact_email_value'
    assert response.escalated is True
    assert response.test_case is True
    assert response.language_code == 'language_code_value'
    assert response.priority == gcs_case.Case.Priority.P0

def test_update_case_rest_required_fields(request_type=case_service.UpdateCaseRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CaseServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_case._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_case._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcs_case.Case()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcs_case.Case.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_case(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_case_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CaseServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_case._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('case',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_case_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CaseServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CaseServiceRestInterceptor())
    client = CaseServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CaseServiceRestInterceptor, 'post_update_case') as post, mock.patch.object(transports.CaseServiceRestInterceptor, 'pre_update_case') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = case_service.UpdateCaseRequest.pb(case_service.UpdateCaseRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcs_case.Case.to_json(gcs_case.Case())
        request = case_service.UpdateCaseRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcs_case.Case()
        client.update_case(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_case_rest_bad_request(transport: str='rest', request_type=case_service.UpdateCaseRequest):
    if False:
        return 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'case': {'name': 'projects/sample1/cases/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_case(request)

def test_update_case_rest_flattened():
    if False:
        print('Hello World!')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcs_case.Case()
        sample_request = {'case': {'name': 'projects/sample1/cases/sample2'}}
        mock_args = dict(case=gcs_case.Case(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcs_case.Case.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_case(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{case.name=projects/*/cases/*}' % client.transport._host, args[1])

def test_update_case_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_case(case_service.UpdateCaseRequest(), case=gcs_case.Case(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_case_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [case_service.EscalateCaseRequest, dict])
def test_escalate_case_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/cases/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = case.Case(name='name_value', display_name='display_name_value', description='description_value', time_zone='time_zone_value', subscriber_email_addresses=['subscriber_email_addresses_value'], state=case.Case.State.NEW, contact_email='contact_email_value', escalated=True, test_case=True, language_code='language_code_value', priority=case.Case.Priority.P0)
        response_value = Response()
        response_value.status_code = 200
        return_value = case.Case.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.escalate_case(request)
    assert isinstance(response, case.Case)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.time_zone == 'time_zone_value'
    assert response.subscriber_email_addresses == ['subscriber_email_addresses_value']
    assert response.state == case.Case.State.NEW
    assert response.contact_email == 'contact_email_value'
    assert response.escalated is True
    assert response.test_case is True
    assert response.language_code == 'language_code_value'
    assert response.priority == case.Case.Priority.P0

def test_escalate_case_rest_required_fields(request_type=case_service.EscalateCaseRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CaseServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).escalate_case._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).escalate_case._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = case.Case()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = case.Case.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.escalate_case(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_escalate_case_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CaseServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.escalate_case._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_escalate_case_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CaseServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CaseServiceRestInterceptor())
    client = CaseServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CaseServiceRestInterceptor, 'post_escalate_case') as post, mock.patch.object(transports.CaseServiceRestInterceptor, 'pre_escalate_case') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = case_service.EscalateCaseRequest.pb(case_service.EscalateCaseRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = case.Case.to_json(case.Case())
        request = case_service.EscalateCaseRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = case.Case()
        client.escalate_case(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_escalate_case_rest_bad_request(transport: str='rest', request_type=case_service.EscalateCaseRequest):
    if False:
        while True:
            i = 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/cases/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.escalate_case(request)

def test_escalate_case_rest_error():
    if False:
        i = 10
        return i + 15
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [case_service.CloseCaseRequest, dict])
def test_close_case_rest(request_type):
    if False:
        return 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/cases/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = case.Case(name='name_value', display_name='display_name_value', description='description_value', time_zone='time_zone_value', subscriber_email_addresses=['subscriber_email_addresses_value'], state=case.Case.State.NEW, contact_email='contact_email_value', escalated=True, test_case=True, language_code='language_code_value', priority=case.Case.Priority.P0)
        response_value = Response()
        response_value.status_code = 200
        return_value = case.Case.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.close_case(request)
    assert isinstance(response, case.Case)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.description == 'description_value'
    assert response.time_zone == 'time_zone_value'
    assert response.subscriber_email_addresses == ['subscriber_email_addresses_value']
    assert response.state == case.Case.State.NEW
    assert response.contact_email == 'contact_email_value'
    assert response.escalated is True
    assert response.test_case is True
    assert response.language_code == 'language_code_value'
    assert response.priority == case.Case.Priority.P0

def test_close_case_rest_required_fields(request_type=case_service.CloseCaseRequest):
    if False:
        return 10
    transport_class = transports.CaseServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).close_case._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).close_case._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = case.Case()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = case.Case.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.close_case(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_close_case_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CaseServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.close_case._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_close_case_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CaseServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CaseServiceRestInterceptor())
    client = CaseServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CaseServiceRestInterceptor, 'post_close_case') as post, mock.patch.object(transports.CaseServiceRestInterceptor, 'pre_close_case') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = case_service.CloseCaseRequest.pb(case_service.CloseCaseRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = case.Case.to_json(case.Case())
        request = case_service.CloseCaseRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = case.Case()
        client.close_case(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_close_case_rest_bad_request(transport: str='rest', request_type=case_service.CloseCaseRequest):
    if False:
        while True:
            i = 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/cases/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.close_case(request)

def test_close_case_rest_error():
    if False:
        while True:
            i = 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [case_service.SearchCaseClassificationsRequest, dict])
def test_search_case_classifications_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = case_service.SearchCaseClassificationsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = case_service.SearchCaseClassificationsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.search_case_classifications(request)
    assert isinstance(response, pagers.SearchCaseClassificationsPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_search_case_classifications_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CaseServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CaseServiceRestInterceptor())
    client = CaseServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CaseServiceRestInterceptor, 'post_search_case_classifications') as post, mock.patch.object(transports.CaseServiceRestInterceptor, 'pre_search_case_classifications') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = case_service.SearchCaseClassificationsRequest.pb(case_service.SearchCaseClassificationsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = case_service.SearchCaseClassificationsResponse.to_json(case_service.SearchCaseClassificationsResponse())
        request = case_service.SearchCaseClassificationsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = case_service.SearchCaseClassificationsResponse()
        client.search_case_classifications(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_search_case_classifications_rest_bad_request(transport: str='rest', request_type=case_service.SearchCaseClassificationsRequest):
    if False:
        return 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.search_case_classifications(request)

def test_search_case_classifications_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (case_service.SearchCaseClassificationsResponse(case_classifications=[case.CaseClassification(), case.CaseClassification(), case.CaseClassification()], next_page_token='abc'), case_service.SearchCaseClassificationsResponse(case_classifications=[], next_page_token='def'), case_service.SearchCaseClassificationsResponse(case_classifications=[case.CaseClassification()], next_page_token='ghi'), case_service.SearchCaseClassificationsResponse(case_classifications=[case.CaseClassification(), case.CaseClassification()]))
        response = response + response
        response = tuple((case_service.SearchCaseClassificationsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {}
        pager = client.search_case_classifications(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, case.CaseClassification) for i in results))
        pages = list(client.search_case_classifications(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CaseServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.CaseServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CaseServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.CaseServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CaseServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CaseServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.CaseServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CaseServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.CaseServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = CaseServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        return 10
    transport = transports.CaseServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.CaseServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.CaseServiceGrpcTransport, transports.CaseServiceGrpcAsyncIOTransport, transports.CaseServiceRestTransport])
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
        return 10
    transport = CaseServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.CaseServiceGrpcTransport)

def test_case_service_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.CaseServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_case_service_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.support_v2.services.case_service.transports.CaseServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.CaseServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('get_case', 'list_cases', 'search_cases', 'create_case', 'update_case', 'escalate_case', 'close_case', 'search_case_classifications')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_case_service_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.support_v2.services.case_service.transports.CaseServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CaseServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_case_service_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.support_v2.services.case_service.transports.CaseServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CaseServiceTransport()
        adc.assert_called_once()

def test_case_service_auth_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        CaseServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.CaseServiceGrpcTransport, transports.CaseServiceGrpcAsyncIOTransport])
def test_case_service_transport_auth_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.CaseServiceGrpcTransport, transports.CaseServiceGrpcAsyncIOTransport, transports.CaseServiceRestTransport])
def test_case_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.CaseServiceGrpcTransport, grpc_helpers), (transports.CaseServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_case_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('cloudsupport.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='cloudsupport.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.CaseServiceGrpcTransport, transports.CaseServiceGrpcAsyncIOTransport])
def test_case_service_grpc_transport_client_cert_source_for_mtls(transport_class):
    if False:
        for i in range(10):
            print('nop')
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

def test_case_service_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.CaseServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_case_service_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudsupport.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('cloudsupport.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudsupport.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_case_service_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudsupport.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('cloudsupport.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudsupport.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_case_service_client_transport_session_collision(transport_name):
    if False:
        while True:
            i = 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = CaseServiceClient(credentials=creds1, transport=transport_name)
    client2 = CaseServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.get_case._session
    session2 = client2.transport.get_case._session
    assert session1 != session2
    session1 = client1.transport.list_cases._session
    session2 = client2.transport.list_cases._session
    assert session1 != session2
    session1 = client1.transport.search_cases._session
    session2 = client2.transport.search_cases._session
    assert session1 != session2
    session1 = client1.transport.create_case._session
    session2 = client2.transport.create_case._session
    assert session1 != session2
    session1 = client1.transport.update_case._session
    session2 = client2.transport.update_case._session
    assert session1 != session2
    session1 = client1.transport.escalate_case._session
    session2 = client2.transport.escalate_case._session
    assert session1 != session2
    session1 = client1.transport.close_case._session
    session2 = client2.transport.close_case._session
    assert session1 != session2
    session1 = client1.transport.search_case_classifications._session
    session2 = client2.transport.search_case_classifications._session
    assert session1 != session2

def test_case_service_grpc_transport_channel():
    if False:
        while True:
            i = 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CaseServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_case_service_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CaseServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.CaseServiceGrpcTransport, transports.CaseServiceGrpcAsyncIOTransport])
def test_case_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.CaseServiceGrpcTransport, transports.CaseServiceGrpcAsyncIOTransport])
def test_case_service_transport_channel_mtls_with_adc(transport_class):
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

def test_case_path():
    if False:
        while True:
            i = 10
    organization = 'squid'
    case = 'clam'
    expected = 'organizations/{organization}/cases/{case}'.format(organization=organization, case=case)
    actual = CaseServiceClient.case_path(organization, case)
    assert expected == actual

def test_parse_case_path():
    if False:
        return 10
    expected = {'organization': 'whelk', 'case': 'octopus'}
    path = CaseServiceClient.case_path(**expected)
    actual = CaseServiceClient.parse_case_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = CaseServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'nudibranch'}
    path = CaseServiceClient.common_billing_account_path(**expected)
    actual = CaseServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = CaseServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'folder': 'mussel'}
    path = CaseServiceClient.common_folder_path(**expected)
    actual = CaseServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = CaseServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'nautilus'}
    path = CaseServiceClient.common_organization_path(**expected)
    actual = CaseServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = CaseServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'abalone'}
    path = CaseServiceClient.common_project_path(**expected)
    actual = CaseServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = CaseServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = CaseServiceClient.common_location_path(**expected)
    actual = CaseServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.CaseServiceTransport, '_prep_wrapped_messages') as prep:
        client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.CaseServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = CaseServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = CaseServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        return 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = CaseServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(CaseServiceClient, transports.CaseServiceGrpcTransport), (CaseServiceAsyncClient, transports.CaseServiceGrpcAsyncIOTransport)])
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