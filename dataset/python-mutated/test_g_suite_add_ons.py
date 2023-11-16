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
from google.apps.script.type.calendar.types import calendar_addon_manifest
from google.apps.script.type.docs.types import docs_addon_manifest
from google.apps.script.type.drive.types import drive_addon_manifest
from google.apps.script.type.gmail.types import gmail_addon_manifest
from google.apps.script.type.sheets.types import sheets_addon_manifest
from google.apps.script.type.slides.types import slides_addon_manifest
from google.apps.script.type.types import addon_widget_set, extension_point, script_manifest
import google.auth
from google.auth import credentials as ga_credentials
from google.auth.exceptions import MutualTLSChannelError
from google.oauth2 import service_account
from google.protobuf import json_format
from google.protobuf import struct_pb2
from google.protobuf import wrappers_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.gsuiteaddons_v1.services.g_suite_add_ons import GSuiteAddOnsAsyncClient, GSuiteAddOnsClient, pagers, transports
from google.cloud.gsuiteaddons_v1.types import gsuiteaddons

def client_cert_source_callback():
    if False:
        print('Hello World!')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        print('Hello World!')
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
    assert GSuiteAddOnsClient._get_default_mtls_endpoint(None) is None
    assert GSuiteAddOnsClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert GSuiteAddOnsClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert GSuiteAddOnsClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert GSuiteAddOnsClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert GSuiteAddOnsClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(GSuiteAddOnsClient, 'grpc'), (GSuiteAddOnsAsyncClient, 'grpc_asyncio'), (GSuiteAddOnsClient, 'rest')])
def test_g_suite_add_ons_client_from_service_account_info(client_class, transport_name):
    if False:
        print('Hello World!')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('gsuiteaddons.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://gsuiteaddons.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.GSuiteAddOnsGrpcTransport, 'grpc'), (transports.GSuiteAddOnsGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.GSuiteAddOnsRestTransport, 'rest')])
def test_g_suite_add_ons_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(GSuiteAddOnsClient, 'grpc'), (GSuiteAddOnsAsyncClient, 'grpc_asyncio'), (GSuiteAddOnsClient, 'rest')])
def test_g_suite_add_ons_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('gsuiteaddons.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://gsuiteaddons.googleapis.com')

def test_g_suite_add_ons_client_get_transport_class():
    if False:
        for i in range(10):
            print('nop')
    transport = GSuiteAddOnsClient.get_transport_class()
    available_transports = [transports.GSuiteAddOnsGrpcTransport, transports.GSuiteAddOnsRestTransport]
    assert transport in available_transports
    transport = GSuiteAddOnsClient.get_transport_class('grpc')
    assert transport == transports.GSuiteAddOnsGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(GSuiteAddOnsClient, transports.GSuiteAddOnsGrpcTransport, 'grpc'), (GSuiteAddOnsAsyncClient, transports.GSuiteAddOnsGrpcAsyncIOTransport, 'grpc_asyncio'), (GSuiteAddOnsClient, transports.GSuiteAddOnsRestTransport, 'rest')])
@mock.patch.object(GSuiteAddOnsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(GSuiteAddOnsClient))
@mock.patch.object(GSuiteAddOnsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(GSuiteAddOnsAsyncClient))
def test_g_suite_add_ons_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(GSuiteAddOnsClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(GSuiteAddOnsClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(GSuiteAddOnsClient, transports.GSuiteAddOnsGrpcTransport, 'grpc', 'true'), (GSuiteAddOnsAsyncClient, transports.GSuiteAddOnsGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (GSuiteAddOnsClient, transports.GSuiteAddOnsGrpcTransport, 'grpc', 'false'), (GSuiteAddOnsAsyncClient, transports.GSuiteAddOnsGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (GSuiteAddOnsClient, transports.GSuiteAddOnsRestTransport, 'rest', 'true'), (GSuiteAddOnsClient, transports.GSuiteAddOnsRestTransport, 'rest', 'false')])
@mock.patch.object(GSuiteAddOnsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(GSuiteAddOnsClient))
@mock.patch.object(GSuiteAddOnsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(GSuiteAddOnsAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_g_suite_add_ons_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [GSuiteAddOnsClient, GSuiteAddOnsAsyncClient])
@mock.patch.object(GSuiteAddOnsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(GSuiteAddOnsClient))
@mock.patch.object(GSuiteAddOnsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(GSuiteAddOnsAsyncClient))
def test_g_suite_add_ons_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(GSuiteAddOnsClient, transports.GSuiteAddOnsGrpcTransport, 'grpc'), (GSuiteAddOnsAsyncClient, transports.GSuiteAddOnsGrpcAsyncIOTransport, 'grpc_asyncio'), (GSuiteAddOnsClient, transports.GSuiteAddOnsRestTransport, 'rest')])
def test_g_suite_add_ons_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(GSuiteAddOnsClient, transports.GSuiteAddOnsGrpcTransport, 'grpc', grpc_helpers), (GSuiteAddOnsAsyncClient, transports.GSuiteAddOnsGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (GSuiteAddOnsClient, transports.GSuiteAddOnsRestTransport, 'rest', None)])
def test_g_suite_add_ons_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_g_suite_add_ons_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.gsuiteaddons_v1.services.g_suite_add_ons.transports.GSuiteAddOnsGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = GSuiteAddOnsClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(GSuiteAddOnsClient, transports.GSuiteAddOnsGrpcTransport, 'grpc', grpc_helpers), (GSuiteAddOnsAsyncClient, transports.GSuiteAddOnsGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_g_suite_add_ons_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('gsuiteaddons.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='gsuiteaddons.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [gsuiteaddons.GetAuthorizationRequest, dict])
def test_get_authorization(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_authorization), '__call__') as call:
        call.return_value = gsuiteaddons.Authorization(name='name_value', service_account_email='service_account_email_value', oauth_client_id='oauth_client_id_value')
        response = client.get_authorization(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.GetAuthorizationRequest()
    assert isinstance(response, gsuiteaddons.Authorization)
    assert response.name == 'name_value'
    assert response.service_account_email == 'service_account_email_value'
    assert response.oauth_client_id == 'oauth_client_id_value'

def test_get_authorization_empty_call():
    if False:
        return 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_authorization), '__call__') as call:
        client.get_authorization()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.GetAuthorizationRequest()

@pytest.mark.asyncio
async def test_get_authorization_async(transport: str='grpc_asyncio', request_type=gsuiteaddons.GetAuthorizationRequest):
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_authorization), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gsuiteaddons.Authorization(name='name_value', service_account_email='service_account_email_value', oauth_client_id='oauth_client_id_value'))
        response = await client.get_authorization(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.GetAuthorizationRequest()
    assert isinstance(response, gsuiteaddons.Authorization)
    assert response.name == 'name_value'
    assert response.service_account_email == 'service_account_email_value'
    assert response.oauth_client_id == 'oauth_client_id_value'

@pytest.mark.asyncio
async def test_get_authorization_async_from_dict():
    await test_get_authorization_async(request_type=dict)

def test_get_authorization_field_headers():
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    request = gsuiteaddons.GetAuthorizationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_authorization), '__call__') as call:
        call.return_value = gsuiteaddons.Authorization()
        client.get_authorization(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_authorization_field_headers_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gsuiteaddons.GetAuthorizationRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_authorization), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gsuiteaddons.Authorization())
        await client.get_authorization(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_authorization_flattened():
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_authorization), '__call__') as call:
        call.return_value = gsuiteaddons.Authorization()
        client.get_authorization(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_authorization_flattened_error():
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_authorization(gsuiteaddons.GetAuthorizationRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_authorization_flattened_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_authorization), '__call__') as call:
        call.return_value = gsuiteaddons.Authorization()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gsuiteaddons.Authorization())
        response = await client.get_authorization(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_authorization_flattened_error_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_authorization(gsuiteaddons.GetAuthorizationRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gsuiteaddons.CreateDeploymentRequest, dict])
def test_create_deployment(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_deployment), '__call__') as call:
        call.return_value = gsuiteaddons.Deployment(name='name_value', oauth_scopes=['oauth_scopes_value'], etag='etag_value')
        response = client.create_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.CreateDeploymentRequest()
    assert isinstance(response, gsuiteaddons.Deployment)
    assert response.name == 'name_value'
    assert response.oauth_scopes == ['oauth_scopes_value']
    assert response.etag == 'etag_value'

def test_create_deployment_empty_call():
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_deployment), '__call__') as call:
        client.create_deployment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.CreateDeploymentRequest()

@pytest.mark.asyncio
async def test_create_deployment_async(transport: str='grpc_asyncio', request_type=gsuiteaddons.CreateDeploymentRequest):
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gsuiteaddons.Deployment(name='name_value', oauth_scopes=['oauth_scopes_value'], etag='etag_value'))
        response = await client.create_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.CreateDeploymentRequest()
    assert isinstance(response, gsuiteaddons.Deployment)
    assert response.name == 'name_value'
    assert response.oauth_scopes == ['oauth_scopes_value']
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_create_deployment_async_from_dict():
    await test_create_deployment_async(request_type=dict)

def test_create_deployment_field_headers():
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    request = gsuiteaddons.CreateDeploymentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_deployment), '__call__') as call:
        call.return_value = gsuiteaddons.Deployment()
        client.create_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_deployment_field_headers_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gsuiteaddons.CreateDeploymentRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gsuiteaddons.Deployment())
        await client.create_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_deployment_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_deployment), '__call__') as call:
        call.return_value = gsuiteaddons.Deployment()
        client.create_deployment(parent='parent_value', deployment=gsuiteaddons.Deployment(name='name_value'), deployment_id='deployment_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].deployment
        mock_val = gsuiteaddons.Deployment(name='name_value')
        assert arg == mock_val
        arg = args[0].deployment_id
        mock_val = 'deployment_id_value'
        assert arg == mock_val

def test_create_deployment_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_deployment(gsuiteaddons.CreateDeploymentRequest(), parent='parent_value', deployment=gsuiteaddons.Deployment(name='name_value'), deployment_id='deployment_id_value')

@pytest.mark.asyncio
async def test_create_deployment_flattened_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_deployment), '__call__') as call:
        call.return_value = gsuiteaddons.Deployment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gsuiteaddons.Deployment())
        response = await client.create_deployment(parent='parent_value', deployment=gsuiteaddons.Deployment(name='name_value'), deployment_id='deployment_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].deployment
        mock_val = gsuiteaddons.Deployment(name='name_value')
        assert arg == mock_val
        arg = args[0].deployment_id
        mock_val = 'deployment_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_deployment_flattened_error_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_deployment(gsuiteaddons.CreateDeploymentRequest(), parent='parent_value', deployment=gsuiteaddons.Deployment(name='name_value'), deployment_id='deployment_id_value')

@pytest.mark.parametrize('request_type', [gsuiteaddons.ReplaceDeploymentRequest, dict])
def test_replace_deployment(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.replace_deployment), '__call__') as call:
        call.return_value = gsuiteaddons.Deployment(name='name_value', oauth_scopes=['oauth_scopes_value'], etag='etag_value')
        response = client.replace_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.ReplaceDeploymentRequest()
    assert isinstance(response, gsuiteaddons.Deployment)
    assert response.name == 'name_value'
    assert response.oauth_scopes == ['oauth_scopes_value']
    assert response.etag == 'etag_value'

def test_replace_deployment_empty_call():
    if False:
        return 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.replace_deployment), '__call__') as call:
        client.replace_deployment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.ReplaceDeploymentRequest()

@pytest.mark.asyncio
async def test_replace_deployment_async(transport: str='grpc_asyncio', request_type=gsuiteaddons.ReplaceDeploymentRequest):
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.replace_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gsuiteaddons.Deployment(name='name_value', oauth_scopes=['oauth_scopes_value'], etag='etag_value'))
        response = await client.replace_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.ReplaceDeploymentRequest()
    assert isinstance(response, gsuiteaddons.Deployment)
    assert response.name == 'name_value'
    assert response.oauth_scopes == ['oauth_scopes_value']
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_replace_deployment_async_from_dict():
    await test_replace_deployment_async(request_type=dict)

def test_replace_deployment_field_headers():
    if False:
        return 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    request = gsuiteaddons.ReplaceDeploymentRequest()
    request.deployment.name = 'name_value'
    with mock.patch.object(type(client.transport.replace_deployment), '__call__') as call:
        call.return_value = gsuiteaddons.Deployment()
        client.replace_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'deployment.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_replace_deployment_field_headers_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gsuiteaddons.ReplaceDeploymentRequest()
    request.deployment.name = 'name_value'
    with mock.patch.object(type(client.transport.replace_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gsuiteaddons.Deployment())
        await client.replace_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'deployment.name=name_value') in kw['metadata']

def test_replace_deployment_flattened():
    if False:
        while True:
            i = 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.replace_deployment), '__call__') as call:
        call.return_value = gsuiteaddons.Deployment()
        client.replace_deployment(deployment=gsuiteaddons.Deployment(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].deployment
        mock_val = gsuiteaddons.Deployment(name='name_value')
        assert arg == mock_val

def test_replace_deployment_flattened_error():
    if False:
        while True:
            i = 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.replace_deployment(gsuiteaddons.ReplaceDeploymentRequest(), deployment=gsuiteaddons.Deployment(name='name_value'))

@pytest.mark.asyncio
async def test_replace_deployment_flattened_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.replace_deployment), '__call__') as call:
        call.return_value = gsuiteaddons.Deployment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gsuiteaddons.Deployment())
        response = await client.replace_deployment(deployment=gsuiteaddons.Deployment(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].deployment
        mock_val = gsuiteaddons.Deployment(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_replace_deployment_flattened_error_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.replace_deployment(gsuiteaddons.ReplaceDeploymentRequest(), deployment=gsuiteaddons.Deployment(name='name_value'))

@pytest.mark.parametrize('request_type', [gsuiteaddons.GetDeploymentRequest, dict])
def test_get_deployment(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_deployment), '__call__') as call:
        call.return_value = gsuiteaddons.Deployment(name='name_value', oauth_scopes=['oauth_scopes_value'], etag='etag_value')
        response = client.get_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.GetDeploymentRequest()
    assert isinstance(response, gsuiteaddons.Deployment)
    assert response.name == 'name_value'
    assert response.oauth_scopes == ['oauth_scopes_value']
    assert response.etag == 'etag_value'

def test_get_deployment_empty_call():
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_deployment), '__call__') as call:
        client.get_deployment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.GetDeploymentRequest()

@pytest.mark.asyncio
async def test_get_deployment_async(transport: str='grpc_asyncio', request_type=gsuiteaddons.GetDeploymentRequest):
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gsuiteaddons.Deployment(name='name_value', oauth_scopes=['oauth_scopes_value'], etag='etag_value'))
        response = await client.get_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.GetDeploymentRequest()
    assert isinstance(response, gsuiteaddons.Deployment)
    assert response.name == 'name_value'
    assert response.oauth_scopes == ['oauth_scopes_value']
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_get_deployment_async_from_dict():
    await test_get_deployment_async(request_type=dict)

def test_get_deployment_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    request = gsuiteaddons.GetDeploymentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_deployment), '__call__') as call:
        call.return_value = gsuiteaddons.Deployment()
        client.get_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_deployment_field_headers_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gsuiteaddons.GetDeploymentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gsuiteaddons.Deployment())
        await client.get_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_deployment_flattened():
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_deployment), '__call__') as call:
        call.return_value = gsuiteaddons.Deployment()
        client.get_deployment(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_deployment_flattened_error():
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_deployment(gsuiteaddons.GetDeploymentRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_deployment_flattened_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_deployment), '__call__') as call:
        call.return_value = gsuiteaddons.Deployment()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gsuiteaddons.Deployment())
        response = await client.get_deployment(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_deployment_flattened_error_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_deployment(gsuiteaddons.GetDeploymentRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gsuiteaddons.ListDeploymentsRequest, dict])
def test_list_deployments(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_deployments), '__call__') as call:
        call.return_value = gsuiteaddons.ListDeploymentsResponse(next_page_token='next_page_token_value')
        response = client.list_deployments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.ListDeploymentsRequest()
    assert isinstance(response, pagers.ListDeploymentsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_deployments_empty_call():
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_deployments), '__call__') as call:
        client.list_deployments()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.ListDeploymentsRequest()

@pytest.mark.asyncio
async def test_list_deployments_async(transport: str='grpc_asyncio', request_type=gsuiteaddons.ListDeploymentsRequest):
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_deployments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gsuiteaddons.ListDeploymentsResponse(next_page_token='next_page_token_value'))
        response = await client.list_deployments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.ListDeploymentsRequest()
    assert isinstance(response, pagers.ListDeploymentsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_deployments_async_from_dict():
    await test_list_deployments_async(request_type=dict)

def test_list_deployments_field_headers():
    if False:
        while True:
            i = 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    request = gsuiteaddons.ListDeploymentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_deployments), '__call__') as call:
        call.return_value = gsuiteaddons.ListDeploymentsResponse()
        client.list_deployments(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_deployments_field_headers_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gsuiteaddons.ListDeploymentsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_deployments), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gsuiteaddons.ListDeploymentsResponse())
        await client.list_deployments(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_deployments_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_deployments), '__call__') as call:
        call.return_value = gsuiteaddons.ListDeploymentsResponse()
        client.list_deployments(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_deployments_flattened_error():
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_deployments(gsuiteaddons.ListDeploymentsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_deployments_flattened_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_deployments), '__call__') as call:
        call.return_value = gsuiteaddons.ListDeploymentsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gsuiteaddons.ListDeploymentsResponse())
        response = await client.list_deployments(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_deployments_flattened_error_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_deployments(gsuiteaddons.ListDeploymentsRequest(), parent='parent_value')

def test_list_deployments_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_deployments), '__call__') as call:
        call.side_effect = (gsuiteaddons.ListDeploymentsResponse(deployments=[gsuiteaddons.Deployment(), gsuiteaddons.Deployment(), gsuiteaddons.Deployment()], next_page_token='abc'), gsuiteaddons.ListDeploymentsResponse(deployments=[], next_page_token='def'), gsuiteaddons.ListDeploymentsResponse(deployments=[gsuiteaddons.Deployment()], next_page_token='ghi'), gsuiteaddons.ListDeploymentsResponse(deployments=[gsuiteaddons.Deployment(), gsuiteaddons.Deployment()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_deployments(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, gsuiteaddons.Deployment) for i in results))

def test_list_deployments_pages(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_deployments), '__call__') as call:
        call.side_effect = (gsuiteaddons.ListDeploymentsResponse(deployments=[gsuiteaddons.Deployment(), gsuiteaddons.Deployment(), gsuiteaddons.Deployment()], next_page_token='abc'), gsuiteaddons.ListDeploymentsResponse(deployments=[], next_page_token='def'), gsuiteaddons.ListDeploymentsResponse(deployments=[gsuiteaddons.Deployment()], next_page_token='ghi'), gsuiteaddons.ListDeploymentsResponse(deployments=[gsuiteaddons.Deployment(), gsuiteaddons.Deployment()]), RuntimeError)
        pages = list(client.list_deployments(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_deployments_async_pager():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_deployments), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (gsuiteaddons.ListDeploymentsResponse(deployments=[gsuiteaddons.Deployment(), gsuiteaddons.Deployment(), gsuiteaddons.Deployment()], next_page_token='abc'), gsuiteaddons.ListDeploymentsResponse(deployments=[], next_page_token='def'), gsuiteaddons.ListDeploymentsResponse(deployments=[gsuiteaddons.Deployment()], next_page_token='ghi'), gsuiteaddons.ListDeploymentsResponse(deployments=[gsuiteaddons.Deployment(), gsuiteaddons.Deployment()]), RuntimeError)
        async_pager = await client.list_deployments(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, gsuiteaddons.Deployment) for i in responses))

@pytest.mark.asyncio
async def test_list_deployments_async_pages():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_deployments), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (gsuiteaddons.ListDeploymentsResponse(deployments=[gsuiteaddons.Deployment(), gsuiteaddons.Deployment(), gsuiteaddons.Deployment()], next_page_token='abc'), gsuiteaddons.ListDeploymentsResponse(deployments=[], next_page_token='def'), gsuiteaddons.ListDeploymentsResponse(deployments=[gsuiteaddons.Deployment()], next_page_token='ghi'), gsuiteaddons.ListDeploymentsResponse(deployments=[gsuiteaddons.Deployment(), gsuiteaddons.Deployment()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_deployments(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [gsuiteaddons.DeleteDeploymentRequest, dict])
def test_delete_deployment(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_deployment), '__call__') as call:
        call.return_value = None
        response = client.delete_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.DeleteDeploymentRequest()
    assert response is None

def test_delete_deployment_empty_call():
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_deployment), '__call__') as call:
        client.delete_deployment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.DeleteDeploymentRequest()

@pytest.mark.asyncio
async def test_delete_deployment_async(transport: str='grpc_asyncio', request_type=gsuiteaddons.DeleteDeploymentRequest):
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.DeleteDeploymentRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_deployment_async_from_dict():
    await test_delete_deployment_async(request_type=dict)

def test_delete_deployment_field_headers():
    if False:
        return 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    request = gsuiteaddons.DeleteDeploymentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_deployment), '__call__') as call:
        call.return_value = None
        client.delete_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_deployment_field_headers_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gsuiteaddons.DeleteDeploymentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_deployment_flattened():
    if False:
        return 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_deployment), '__call__') as call:
        call.return_value = None
        client.delete_deployment(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_deployment_flattened_error():
    if False:
        while True:
            i = 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_deployment(gsuiteaddons.DeleteDeploymentRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_deployment_flattened_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_deployment), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_deployment(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_deployment_flattened_error_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_deployment(gsuiteaddons.DeleteDeploymentRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gsuiteaddons.InstallDeploymentRequest, dict])
def test_install_deployment(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.install_deployment), '__call__') as call:
        call.return_value = None
        response = client.install_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.InstallDeploymentRequest()
    assert response is None

def test_install_deployment_empty_call():
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.install_deployment), '__call__') as call:
        client.install_deployment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.InstallDeploymentRequest()

@pytest.mark.asyncio
async def test_install_deployment_async(transport: str='grpc_asyncio', request_type=gsuiteaddons.InstallDeploymentRequest):
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.install_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.install_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.InstallDeploymentRequest()
    assert response is None

@pytest.mark.asyncio
async def test_install_deployment_async_from_dict():
    await test_install_deployment_async(request_type=dict)

def test_install_deployment_field_headers():
    if False:
        return 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    request = gsuiteaddons.InstallDeploymentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.install_deployment), '__call__') as call:
        call.return_value = None
        client.install_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_install_deployment_field_headers_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gsuiteaddons.InstallDeploymentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.install_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.install_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_install_deployment_flattened():
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.install_deployment), '__call__') as call:
        call.return_value = None
        client.install_deployment(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_install_deployment_flattened_error():
    if False:
        while True:
            i = 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.install_deployment(gsuiteaddons.InstallDeploymentRequest(), name='name_value')

@pytest.mark.asyncio
async def test_install_deployment_flattened_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.install_deployment), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.install_deployment(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_install_deployment_flattened_error_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.install_deployment(gsuiteaddons.InstallDeploymentRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gsuiteaddons.UninstallDeploymentRequest, dict])
def test_uninstall_deployment(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.uninstall_deployment), '__call__') as call:
        call.return_value = None
        response = client.uninstall_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.UninstallDeploymentRequest()
    assert response is None

def test_uninstall_deployment_empty_call():
    if False:
        while True:
            i = 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.uninstall_deployment), '__call__') as call:
        client.uninstall_deployment()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.UninstallDeploymentRequest()

@pytest.mark.asyncio
async def test_uninstall_deployment_async(transport: str='grpc_asyncio', request_type=gsuiteaddons.UninstallDeploymentRequest):
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.uninstall_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.uninstall_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.UninstallDeploymentRequest()
    assert response is None

@pytest.mark.asyncio
async def test_uninstall_deployment_async_from_dict():
    await test_uninstall_deployment_async(request_type=dict)

def test_uninstall_deployment_field_headers():
    if False:
        return 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    request = gsuiteaddons.UninstallDeploymentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.uninstall_deployment), '__call__') as call:
        call.return_value = None
        client.uninstall_deployment(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_uninstall_deployment_field_headers_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gsuiteaddons.UninstallDeploymentRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.uninstall_deployment), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.uninstall_deployment(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_uninstall_deployment_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.uninstall_deployment), '__call__') as call:
        call.return_value = None
        client.uninstall_deployment(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_uninstall_deployment_flattened_error():
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.uninstall_deployment(gsuiteaddons.UninstallDeploymentRequest(), name='name_value')

@pytest.mark.asyncio
async def test_uninstall_deployment_flattened_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.uninstall_deployment), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.uninstall_deployment(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_uninstall_deployment_flattened_error_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.uninstall_deployment(gsuiteaddons.UninstallDeploymentRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gsuiteaddons.GetInstallStatusRequest, dict])
def test_get_install_status(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_install_status), '__call__') as call:
        call.return_value = gsuiteaddons.InstallStatus(name='name_value')
        response = client.get_install_status(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.GetInstallStatusRequest()
    assert isinstance(response, gsuiteaddons.InstallStatus)
    assert response.name == 'name_value'

def test_get_install_status_empty_call():
    if False:
        return 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_install_status), '__call__') as call:
        client.get_install_status()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.GetInstallStatusRequest()

@pytest.mark.asyncio
async def test_get_install_status_async(transport: str='grpc_asyncio', request_type=gsuiteaddons.GetInstallStatusRequest):
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_install_status), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gsuiteaddons.InstallStatus(name='name_value'))
        response = await client.get_install_status(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gsuiteaddons.GetInstallStatusRequest()
    assert isinstance(response, gsuiteaddons.InstallStatus)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_get_install_status_async_from_dict():
    await test_get_install_status_async(request_type=dict)

def test_get_install_status_field_headers():
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    request = gsuiteaddons.GetInstallStatusRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_install_status), '__call__') as call:
        call.return_value = gsuiteaddons.InstallStatus()
        client.get_install_status(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_install_status_field_headers_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gsuiteaddons.GetInstallStatusRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_install_status), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gsuiteaddons.InstallStatus())
        await client.get_install_status(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_install_status_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_install_status), '__call__') as call:
        call.return_value = gsuiteaddons.InstallStatus()
        client.get_install_status(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_install_status_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_install_status(gsuiteaddons.GetInstallStatusRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_install_status_flattened_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_install_status), '__call__') as call:
        call.return_value = gsuiteaddons.InstallStatus()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gsuiteaddons.InstallStatus())
        response = await client.get_install_status(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_install_status_flattened_error_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_install_status(gsuiteaddons.GetInstallStatusRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [gsuiteaddons.GetAuthorizationRequest, dict])
def test_get_authorization_rest(request_type):
    if False:
        return 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/authorization'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gsuiteaddons.Authorization(name='name_value', service_account_email='service_account_email_value', oauth_client_id='oauth_client_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gsuiteaddons.Authorization.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_authorization(request)
    assert isinstance(response, gsuiteaddons.Authorization)
    assert response.name == 'name_value'
    assert response.service_account_email == 'service_account_email_value'
    assert response.oauth_client_id == 'oauth_client_id_value'

def test_get_authorization_rest_required_fields(request_type=gsuiteaddons.GetAuthorizationRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.GSuiteAddOnsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_authorization._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_authorization._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gsuiteaddons.Authorization()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gsuiteaddons.Authorization.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_authorization(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_authorization_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.GSuiteAddOnsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_authorization._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_authorization_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.GSuiteAddOnsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GSuiteAddOnsRestInterceptor())
    client = GSuiteAddOnsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GSuiteAddOnsRestInterceptor, 'post_get_authorization') as post, mock.patch.object(transports.GSuiteAddOnsRestInterceptor, 'pre_get_authorization') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gsuiteaddons.GetAuthorizationRequest.pb(gsuiteaddons.GetAuthorizationRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gsuiteaddons.Authorization.to_json(gsuiteaddons.Authorization())
        request = gsuiteaddons.GetAuthorizationRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gsuiteaddons.Authorization()
        client.get_authorization(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_authorization_rest_bad_request(transport: str='rest', request_type=gsuiteaddons.GetAuthorizationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/authorization'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_authorization(request)

def test_get_authorization_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gsuiteaddons.Authorization()
        sample_request = {'name': 'projects/sample1/authorization'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gsuiteaddons.Authorization.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_authorization(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/authorization}' % client.transport._host, args[1])

def test_get_authorization_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_authorization(gsuiteaddons.GetAuthorizationRequest(), name='name_value')

def test_get_authorization_rest_error():
    if False:
        while True:
            i = 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gsuiteaddons.CreateDeploymentRequest, dict])
def test_create_deployment_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request_init['deployment'] = {'name': 'name_value', 'oauth_scopes': ['oauth_scopes_value1', 'oauth_scopes_value2'], 'add_ons': {'common': {'name': 'name_value', 'logo_url': 'logo_url_value', 'layout_properties': {'primary_color': 'primary_color_value', 'secondary_color': 'secondary_color_value'}, 'add_on_widget_set': {'used_widgets': [1]}, 'use_locale_from_app': True, 'homepage_trigger': {'run_function': 'run_function_value', 'enabled': {'value': True}}, 'universal_actions': [{'label': 'label_value', 'open_link': 'open_link_value', 'run_function': 'run_function_value'}], 'open_link_url_prefixes': {'values': [{'null_value': 0, 'number_value': 0.1285, 'string_value': 'string_value_value', 'bool_value': True, 'struct_value': {'fields': {}}, 'list_value': {}}]}}, 'gmail': {'homepage_trigger': {}, 'contextual_triggers': [{'unconditional': {}, 'on_trigger_function': 'on_trigger_function_value'}], 'universal_actions': [{'text': 'text_value', 'open_link': 'open_link_value', 'run_function': 'run_function_value'}], 'compose_trigger': {'actions': [{'run_function': 'run_function_value', 'label': 'label_value', 'logo_url': 'logo_url_value'}], 'draft_access': 1}, 'authorization_check_function': 'authorization_check_function_value'}, 'drive': {'homepage_trigger': {}, 'on_items_selected_trigger': {'run_function': 'run_function_value'}}, 'calendar': {'homepage_trigger': {}, 'conference_solution': [{'on_create_function': 'on_create_function_value', 'id': 'id_value', 'name': 'name_value', 'logo_url': 'logo_url_value'}], 'create_settings_url_function': 'create_settings_url_function_value', 'event_open_trigger': {'run_function': 'run_function_value'}, 'event_update_trigger': {}, 'current_event_access': 1}, 'docs': {'homepage_trigger': {}, 'on_file_scope_granted_trigger': {'run_function': 'run_function_value'}}, 'sheets': {'homepage_trigger': {}, 'on_file_scope_granted_trigger': {'run_function': 'run_function_value'}}, 'slides': {'homepage_trigger': {}, 'on_file_scope_granted_trigger': {'run_function': 'run_function_value'}}, 'http_options': {'authorization_header': 1}}, 'etag': 'etag_value'}
    test_field = gsuiteaddons.CreateDeploymentRequest.meta.fields['deployment']

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
    for (field, value) in request_init['deployment'].items():
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
                for i in range(0, len(request_init['deployment'][field])):
                    del request_init['deployment'][field][i][subfield]
            else:
                del request_init['deployment'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gsuiteaddons.Deployment(name='name_value', oauth_scopes=['oauth_scopes_value'], etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gsuiteaddons.Deployment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_deployment(request)
    assert isinstance(response, gsuiteaddons.Deployment)
    assert response.name == 'name_value'
    assert response.oauth_scopes == ['oauth_scopes_value']
    assert response.etag == 'etag_value'

def test_create_deployment_rest_required_fields(request_type=gsuiteaddons.CreateDeploymentRequest):
    if False:
        print('Hello World!')
    transport_class = transports.GSuiteAddOnsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['deployment_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'deploymentId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'deploymentId' in jsonified_request
    assert jsonified_request['deploymentId'] == request_init['deployment_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['deploymentId'] = 'deployment_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_deployment._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('deployment_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'deploymentId' in jsonified_request
    assert jsonified_request['deploymentId'] == 'deployment_id_value'
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gsuiteaddons.Deployment()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gsuiteaddons.Deployment.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_deployment(request)
            expected_params = [('deploymentId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_deployment_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.GSuiteAddOnsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_deployment._get_unset_required_fields({})
    assert set(unset_fields) == set(('deploymentId',)) & set(('parent', 'deploymentId', 'deployment'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_deployment_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.GSuiteAddOnsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GSuiteAddOnsRestInterceptor())
    client = GSuiteAddOnsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GSuiteAddOnsRestInterceptor, 'post_create_deployment') as post, mock.patch.object(transports.GSuiteAddOnsRestInterceptor, 'pre_create_deployment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gsuiteaddons.CreateDeploymentRequest.pb(gsuiteaddons.CreateDeploymentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gsuiteaddons.Deployment.to_json(gsuiteaddons.Deployment())
        request = gsuiteaddons.CreateDeploymentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gsuiteaddons.Deployment()
        client.create_deployment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_deployment_rest_bad_request(transport: str='rest', request_type=gsuiteaddons.CreateDeploymentRequest):
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_deployment(request)

def test_create_deployment_rest_flattened():
    if False:
        while True:
            i = 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gsuiteaddons.Deployment()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value', deployment=gsuiteaddons.Deployment(name='name_value'), deployment_id='deployment_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gsuiteaddons.Deployment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_deployment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*}/deployments' % client.transport._host, args[1])

def test_create_deployment_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_deployment(gsuiteaddons.CreateDeploymentRequest(), parent='parent_value', deployment=gsuiteaddons.Deployment(name='name_value'), deployment_id='deployment_id_value')

def test_create_deployment_rest_error():
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gsuiteaddons.ReplaceDeploymentRequest, dict])
def test_replace_deployment_rest(request_type):
    if False:
        return 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'deployment': {'name': 'projects/sample1/deployments/sample2'}}
    request_init['deployment'] = {'name': 'projects/sample1/deployments/sample2', 'oauth_scopes': ['oauth_scopes_value1', 'oauth_scopes_value2'], 'add_ons': {'common': {'name': 'name_value', 'logo_url': 'logo_url_value', 'layout_properties': {'primary_color': 'primary_color_value', 'secondary_color': 'secondary_color_value'}, 'add_on_widget_set': {'used_widgets': [1]}, 'use_locale_from_app': True, 'homepage_trigger': {'run_function': 'run_function_value', 'enabled': {'value': True}}, 'universal_actions': [{'label': 'label_value', 'open_link': 'open_link_value', 'run_function': 'run_function_value'}], 'open_link_url_prefixes': {'values': [{'null_value': 0, 'number_value': 0.1285, 'string_value': 'string_value_value', 'bool_value': True, 'struct_value': {'fields': {}}, 'list_value': {}}]}}, 'gmail': {'homepage_trigger': {}, 'contextual_triggers': [{'unconditional': {}, 'on_trigger_function': 'on_trigger_function_value'}], 'universal_actions': [{'text': 'text_value', 'open_link': 'open_link_value', 'run_function': 'run_function_value'}], 'compose_trigger': {'actions': [{'run_function': 'run_function_value', 'label': 'label_value', 'logo_url': 'logo_url_value'}], 'draft_access': 1}, 'authorization_check_function': 'authorization_check_function_value'}, 'drive': {'homepage_trigger': {}, 'on_items_selected_trigger': {'run_function': 'run_function_value'}}, 'calendar': {'homepage_trigger': {}, 'conference_solution': [{'on_create_function': 'on_create_function_value', 'id': 'id_value', 'name': 'name_value', 'logo_url': 'logo_url_value'}], 'create_settings_url_function': 'create_settings_url_function_value', 'event_open_trigger': {'run_function': 'run_function_value'}, 'event_update_trigger': {}, 'current_event_access': 1}, 'docs': {'homepage_trigger': {}, 'on_file_scope_granted_trigger': {'run_function': 'run_function_value'}}, 'sheets': {'homepage_trigger': {}, 'on_file_scope_granted_trigger': {'run_function': 'run_function_value'}}, 'slides': {'homepage_trigger': {}, 'on_file_scope_granted_trigger': {'run_function': 'run_function_value'}}, 'http_options': {'authorization_header': 1}}, 'etag': 'etag_value'}
    test_field = gsuiteaddons.ReplaceDeploymentRequest.meta.fields['deployment']

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
    for (field, value) in request_init['deployment'].items():
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
                for i in range(0, len(request_init['deployment'][field])):
                    del request_init['deployment'][field][i][subfield]
            else:
                del request_init['deployment'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gsuiteaddons.Deployment(name='name_value', oauth_scopes=['oauth_scopes_value'], etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gsuiteaddons.Deployment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.replace_deployment(request)
    assert isinstance(response, gsuiteaddons.Deployment)
    assert response.name == 'name_value'
    assert response.oauth_scopes == ['oauth_scopes_value']
    assert response.etag == 'etag_value'

def test_replace_deployment_rest_required_fields(request_type=gsuiteaddons.ReplaceDeploymentRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.GSuiteAddOnsRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).replace_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).replace_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gsuiteaddons.Deployment()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'put', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gsuiteaddons.Deployment.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.replace_deployment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_replace_deployment_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.GSuiteAddOnsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.replace_deployment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('deployment',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_replace_deployment_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.GSuiteAddOnsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GSuiteAddOnsRestInterceptor())
    client = GSuiteAddOnsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GSuiteAddOnsRestInterceptor, 'post_replace_deployment') as post, mock.patch.object(transports.GSuiteAddOnsRestInterceptor, 'pre_replace_deployment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gsuiteaddons.ReplaceDeploymentRequest.pb(gsuiteaddons.ReplaceDeploymentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gsuiteaddons.Deployment.to_json(gsuiteaddons.Deployment())
        request = gsuiteaddons.ReplaceDeploymentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gsuiteaddons.Deployment()
        client.replace_deployment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_replace_deployment_rest_bad_request(transport: str='rest', request_type=gsuiteaddons.ReplaceDeploymentRequest):
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'deployment': {'name': 'projects/sample1/deployments/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.replace_deployment(request)

def test_replace_deployment_rest_flattened():
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gsuiteaddons.Deployment()
        sample_request = {'deployment': {'name': 'projects/sample1/deployments/sample2'}}
        mock_args = dict(deployment=gsuiteaddons.Deployment(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gsuiteaddons.Deployment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.replace_deployment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{deployment.name=projects/*/deployments/*}' % client.transport._host, args[1])

def test_replace_deployment_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.replace_deployment(gsuiteaddons.ReplaceDeploymentRequest(), deployment=gsuiteaddons.Deployment(name='name_value'))

def test_replace_deployment_rest_error():
    if False:
        return 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gsuiteaddons.GetDeploymentRequest, dict])
def test_get_deployment_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/deployments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gsuiteaddons.Deployment(name='name_value', oauth_scopes=['oauth_scopes_value'], etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gsuiteaddons.Deployment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_deployment(request)
    assert isinstance(response, gsuiteaddons.Deployment)
    assert response.name == 'name_value'
    assert response.oauth_scopes == ['oauth_scopes_value']
    assert response.etag == 'etag_value'

def test_get_deployment_rest_required_fields(request_type=gsuiteaddons.GetDeploymentRequest):
    if False:
        return 10
    transport_class = transports.GSuiteAddOnsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gsuiteaddons.Deployment()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gsuiteaddons.Deployment.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_deployment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_deployment_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.GSuiteAddOnsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_deployment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_deployment_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.GSuiteAddOnsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GSuiteAddOnsRestInterceptor())
    client = GSuiteAddOnsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GSuiteAddOnsRestInterceptor, 'post_get_deployment') as post, mock.patch.object(transports.GSuiteAddOnsRestInterceptor, 'pre_get_deployment') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gsuiteaddons.GetDeploymentRequest.pb(gsuiteaddons.GetDeploymentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gsuiteaddons.Deployment.to_json(gsuiteaddons.Deployment())
        request = gsuiteaddons.GetDeploymentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gsuiteaddons.Deployment()
        client.get_deployment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_deployment_rest_bad_request(transport: str='rest', request_type=gsuiteaddons.GetDeploymentRequest):
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/deployments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_deployment(request)

def test_get_deployment_rest_flattened():
    if False:
        while True:
            i = 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gsuiteaddons.Deployment()
        sample_request = {'name': 'projects/sample1/deployments/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gsuiteaddons.Deployment.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_deployment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/deployments/*}' % client.transport._host, args[1])

def test_get_deployment_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_deployment(gsuiteaddons.GetDeploymentRequest(), name='name_value')

def test_get_deployment_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gsuiteaddons.ListDeploymentsRequest, dict])
def test_list_deployments_rest(request_type):
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gsuiteaddons.ListDeploymentsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gsuiteaddons.ListDeploymentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_deployments(request)
    assert isinstance(response, pagers.ListDeploymentsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_deployments_rest_required_fields(request_type=gsuiteaddons.ListDeploymentsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.GSuiteAddOnsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_deployments._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_deployments._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gsuiteaddons.ListDeploymentsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gsuiteaddons.ListDeploymentsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_deployments(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_deployments_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.GSuiteAddOnsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_deployments._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_deployments_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.GSuiteAddOnsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GSuiteAddOnsRestInterceptor())
    client = GSuiteAddOnsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GSuiteAddOnsRestInterceptor, 'post_list_deployments') as post, mock.patch.object(transports.GSuiteAddOnsRestInterceptor, 'pre_list_deployments') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gsuiteaddons.ListDeploymentsRequest.pb(gsuiteaddons.ListDeploymentsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gsuiteaddons.ListDeploymentsResponse.to_json(gsuiteaddons.ListDeploymentsResponse())
        request = gsuiteaddons.ListDeploymentsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gsuiteaddons.ListDeploymentsResponse()
        client.list_deployments(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_deployments_rest_bad_request(transport: str='rest', request_type=gsuiteaddons.ListDeploymentsRequest):
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_deployments(request)

def test_list_deployments_rest_flattened():
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gsuiteaddons.ListDeploymentsResponse()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gsuiteaddons.ListDeploymentsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_deployments(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*}/deployments' % client.transport._host, args[1])

def test_list_deployments_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_deployments(gsuiteaddons.ListDeploymentsRequest(), parent='parent_value')

def test_list_deployments_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (gsuiteaddons.ListDeploymentsResponse(deployments=[gsuiteaddons.Deployment(), gsuiteaddons.Deployment(), gsuiteaddons.Deployment()], next_page_token='abc'), gsuiteaddons.ListDeploymentsResponse(deployments=[], next_page_token='def'), gsuiteaddons.ListDeploymentsResponse(deployments=[gsuiteaddons.Deployment()], next_page_token='ghi'), gsuiteaddons.ListDeploymentsResponse(deployments=[gsuiteaddons.Deployment(), gsuiteaddons.Deployment()]))
        response = response + response
        response = tuple((gsuiteaddons.ListDeploymentsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1'}
        pager = client.list_deployments(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, gsuiteaddons.Deployment) for i in results))
        pages = list(client.list_deployments(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [gsuiteaddons.DeleteDeploymentRequest, dict])
def test_delete_deployment_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/deployments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_deployment(request)
    assert response is None

def test_delete_deployment_rest_required_fields(request_type=gsuiteaddons.DeleteDeploymentRequest):
    if False:
        return 10
    transport_class = transports.GSuiteAddOnsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_deployment._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('etag',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_deployment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_deployment_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.GSuiteAddOnsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_deployment._get_unset_required_fields({})
    assert set(unset_fields) == set(('etag',)) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_deployment_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.GSuiteAddOnsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GSuiteAddOnsRestInterceptor())
    client = GSuiteAddOnsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GSuiteAddOnsRestInterceptor, 'pre_delete_deployment') as pre:
        pre.assert_not_called()
        pb_message = gsuiteaddons.DeleteDeploymentRequest.pb(gsuiteaddons.DeleteDeploymentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = gsuiteaddons.DeleteDeploymentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_deployment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_deployment_rest_bad_request(transport: str='rest', request_type=gsuiteaddons.DeleteDeploymentRequest):
    if False:
        return 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/deployments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_deployment(request)

def test_delete_deployment_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/deployments/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_deployment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/deployments/*}' % client.transport._host, args[1])

def test_delete_deployment_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_deployment(gsuiteaddons.DeleteDeploymentRequest(), name='name_value')

def test_delete_deployment_rest_error():
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gsuiteaddons.InstallDeploymentRequest, dict])
def test_install_deployment_rest(request_type):
    if False:
        return 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/deployments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.install_deployment(request)
    assert response is None

def test_install_deployment_rest_required_fields(request_type=gsuiteaddons.InstallDeploymentRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.GSuiteAddOnsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).install_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).install_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = None
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = ''
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.install_deployment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_install_deployment_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.GSuiteAddOnsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.install_deployment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_install_deployment_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.GSuiteAddOnsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GSuiteAddOnsRestInterceptor())
    client = GSuiteAddOnsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GSuiteAddOnsRestInterceptor, 'pre_install_deployment') as pre:
        pre.assert_not_called()
        pb_message = gsuiteaddons.InstallDeploymentRequest.pb(gsuiteaddons.InstallDeploymentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = gsuiteaddons.InstallDeploymentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.install_deployment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_install_deployment_rest_bad_request(transport: str='rest', request_type=gsuiteaddons.InstallDeploymentRequest):
    if False:
        return 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/deployments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.install_deployment(request)

def test_install_deployment_rest_flattened():
    if False:
        return 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/deployments/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.install_deployment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/deployments/*}:install' % client.transport._host, args[1])

def test_install_deployment_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.install_deployment(gsuiteaddons.InstallDeploymentRequest(), name='name_value')

def test_install_deployment_rest_error():
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gsuiteaddons.UninstallDeploymentRequest, dict])
def test_uninstall_deployment_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/deployments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.uninstall_deployment(request)
    assert response is None

def test_uninstall_deployment_rest_required_fields(request_type=gsuiteaddons.UninstallDeploymentRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.GSuiteAddOnsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).uninstall_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).uninstall_deployment._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = None
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = ''
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.uninstall_deployment(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_uninstall_deployment_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.GSuiteAddOnsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.uninstall_deployment._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_uninstall_deployment_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.GSuiteAddOnsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GSuiteAddOnsRestInterceptor())
    client = GSuiteAddOnsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GSuiteAddOnsRestInterceptor, 'pre_uninstall_deployment') as pre:
        pre.assert_not_called()
        pb_message = gsuiteaddons.UninstallDeploymentRequest.pb(gsuiteaddons.UninstallDeploymentRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = gsuiteaddons.UninstallDeploymentRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.uninstall_deployment(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_uninstall_deployment_rest_bad_request(transport: str='rest', request_type=gsuiteaddons.UninstallDeploymentRequest):
    if False:
        return 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/deployments/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.uninstall_deployment(request)

def test_uninstall_deployment_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/deployments/sample2'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.uninstall_deployment(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/deployments/*}:uninstall' % client.transport._host, args[1])

def test_uninstall_deployment_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.uninstall_deployment(gsuiteaddons.UninstallDeploymentRequest(), name='name_value')

def test_uninstall_deployment_rest_error():
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [gsuiteaddons.GetInstallStatusRequest, dict])
def test_get_install_status_rest(request_type):
    if False:
        print('Hello World!')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/deployments/sample2/installStatus'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gsuiteaddons.InstallStatus(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gsuiteaddons.InstallStatus.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_install_status(request)
    assert isinstance(response, gsuiteaddons.InstallStatus)
    assert response.name == 'name_value'

def test_get_install_status_rest_required_fields(request_type=gsuiteaddons.GetInstallStatusRequest):
    if False:
        print('Hello World!')
    transport_class = transports.GSuiteAddOnsRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_install_status._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_install_status._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gsuiteaddons.InstallStatus()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gsuiteaddons.InstallStatus.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_install_status(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_install_status_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.GSuiteAddOnsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_install_status._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_install_status_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.GSuiteAddOnsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.GSuiteAddOnsRestInterceptor())
    client = GSuiteAddOnsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.GSuiteAddOnsRestInterceptor, 'post_get_install_status') as post, mock.patch.object(transports.GSuiteAddOnsRestInterceptor, 'pre_get_install_status') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gsuiteaddons.GetInstallStatusRequest.pb(gsuiteaddons.GetInstallStatusRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gsuiteaddons.InstallStatus.to_json(gsuiteaddons.InstallStatus())
        request = gsuiteaddons.GetInstallStatusRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gsuiteaddons.InstallStatus()
        client.get_install_status(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_install_status_rest_bad_request(transport: str='rest', request_type=gsuiteaddons.GetInstallStatusRequest):
    if False:
        while True:
            i = 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/deployments/sample2/installStatus'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_install_status(request)

def test_get_install_status_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gsuiteaddons.InstallStatus()
        sample_request = {'name': 'projects/sample1/deployments/sample2/installStatus'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gsuiteaddons.InstallStatus.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_install_status(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/deployments/*/installStatus}' % client.transport._host, args[1])

def test_get_install_status_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_install_status(gsuiteaddons.GetInstallStatusRequest(), name='name_value')

def test_get_install_status_rest_error():
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.GSuiteAddOnsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.GSuiteAddOnsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = GSuiteAddOnsClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.GSuiteAddOnsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = GSuiteAddOnsClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = GSuiteAddOnsClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.GSuiteAddOnsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = GSuiteAddOnsClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.GSuiteAddOnsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = GSuiteAddOnsClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.GSuiteAddOnsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.GSuiteAddOnsGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.GSuiteAddOnsGrpcTransport, transports.GSuiteAddOnsGrpcAsyncIOTransport, transports.GSuiteAddOnsRestTransport])
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
        return 10
    transport = GSuiteAddOnsClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        for i in range(10):
            print('nop')
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.GSuiteAddOnsGrpcTransport)

def test_g_suite_add_ons_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.GSuiteAddOnsTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_g_suite_add_ons_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.gsuiteaddons_v1.services.g_suite_add_ons.transports.GSuiteAddOnsTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.GSuiteAddOnsTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('get_authorization', 'create_deployment', 'replace_deployment', 'get_deployment', 'list_deployments', 'delete_deployment', 'install_deployment', 'uninstall_deployment', 'get_install_status')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_g_suite_add_ons_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.gsuiteaddons_v1.services.g_suite_add_ons.transports.GSuiteAddOnsTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.GSuiteAddOnsTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_g_suite_add_ons_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.gsuiteaddons_v1.services.g_suite_add_ons.transports.GSuiteAddOnsTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.GSuiteAddOnsTransport()
        adc.assert_called_once()

def test_g_suite_add_ons_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        GSuiteAddOnsClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.GSuiteAddOnsGrpcTransport, transports.GSuiteAddOnsGrpcAsyncIOTransport])
def test_g_suite_add_ons_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.GSuiteAddOnsGrpcTransport, transports.GSuiteAddOnsGrpcAsyncIOTransport, transports.GSuiteAddOnsRestTransport])
def test_g_suite_add_ons_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.GSuiteAddOnsGrpcTransport, grpc_helpers), (transports.GSuiteAddOnsGrpcAsyncIOTransport, grpc_helpers_async)])
def test_g_suite_add_ons_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('gsuiteaddons.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='gsuiteaddons.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.GSuiteAddOnsGrpcTransport, transports.GSuiteAddOnsGrpcAsyncIOTransport])
def test_g_suite_add_ons_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_g_suite_add_ons_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.GSuiteAddOnsRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_g_suite_add_ons_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='gsuiteaddons.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('gsuiteaddons.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://gsuiteaddons.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_g_suite_add_ons_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='gsuiteaddons.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('gsuiteaddons.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://gsuiteaddons.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_g_suite_add_ons_client_transport_session_collision(transport_name):
    if False:
        while True:
            i = 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = GSuiteAddOnsClient(credentials=creds1, transport=transport_name)
    client2 = GSuiteAddOnsClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.get_authorization._session
    session2 = client2.transport.get_authorization._session
    assert session1 != session2
    session1 = client1.transport.create_deployment._session
    session2 = client2.transport.create_deployment._session
    assert session1 != session2
    session1 = client1.transport.replace_deployment._session
    session2 = client2.transport.replace_deployment._session
    assert session1 != session2
    session1 = client1.transport.get_deployment._session
    session2 = client2.transport.get_deployment._session
    assert session1 != session2
    session1 = client1.transport.list_deployments._session
    session2 = client2.transport.list_deployments._session
    assert session1 != session2
    session1 = client1.transport.delete_deployment._session
    session2 = client2.transport.delete_deployment._session
    assert session1 != session2
    session1 = client1.transport.install_deployment._session
    session2 = client2.transport.install_deployment._session
    assert session1 != session2
    session1 = client1.transport.uninstall_deployment._session
    session2 = client2.transport.uninstall_deployment._session
    assert session1 != session2
    session1 = client1.transport.get_install_status._session
    session2 = client2.transport.get_install_status._session
    assert session1 != session2

def test_g_suite_add_ons_grpc_transport_channel():
    if False:
        print('Hello World!')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.GSuiteAddOnsGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_g_suite_add_ons_grpc_asyncio_transport_channel():
    if False:
        while True:
            i = 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.GSuiteAddOnsGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.GSuiteAddOnsGrpcTransport, transports.GSuiteAddOnsGrpcAsyncIOTransport])
def test_g_suite_add_ons_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.GSuiteAddOnsGrpcTransport, transports.GSuiteAddOnsGrpcAsyncIOTransport])
def test_g_suite_add_ons_transport_channel_mtls_with_adc(transport_class):
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

def test_authorization_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    expected = 'projects/{project}/authorization'.format(project=project)
    actual = GSuiteAddOnsClient.authorization_path(project)
    assert expected == actual

def test_parse_authorization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'clam'}
    path = GSuiteAddOnsClient.authorization_path(**expected)
    actual = GSuiteAddOnsClient.parse_authorization_path(path)
    assert expected == actual

def test_deployment_path():
    if False:
        return 10
    project = 'whelk'
    deployment = 'octopus'
    expected = 'projects/{project}/deployments/{deployment}'.format(project=project, deployment=deployment)
    actual = GSuiteAddOnsClient.deployment_path(project, deployment)
    assert expected == actual

def test_parse_deployment_path():
    if False:
        return 10
    expected = {'project': 'oyster', 'deployment': 'nudibranch'}
    path = GSuiteAddOnsClient.deployment_path(**expected)
    actual = GSuiteAddOnsClient.parse_deployment_path(path)
    assert expected == actual

def test_install_status_path():
    if False:
        i = 10
        return i + 15
    project = 'cuttlefish'
    deployment = 'mussel'
    expected = 'projects/{project}/deployments/{deployment}/installStatus'.format(project=project, deployment=deployment)
    actual = GSuiteAddOnsClient.install_status_path(project, deployment)
    assert expected == actual

def test_parse_install_status_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'winkle', 'deployment': 'nautilus'}
    path = GSuiteAddOnsClient.install_status_path(**expected)
    actual = GSuiteAddOnsClient.parse_install_status_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'scallop'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = GSuiteAddOnsClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'abalone'}
    path = GSuiteAddOnsClient.common_billing_account_path(**expected)
    actual = GSuiteAddOnsClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'squid'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = GSuiteAddOnsClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'clam'}
    path = GSuiteAddOnsClient.common_folder_path(**expected)
    actual = GSuiteAddOnsClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'whelk'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = GSuiteAddOnsClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'octopus'}
    path = GSuiteAddOnsClient.common_organization_path(**expected)
    actual = GSuiteAddOnsClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'oyster'
    expected = 'projects/{project}'.format(project=project)
    actual = GSuiteAddOnsClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'nudibranch'}
    path = GSuiteAddOnsClient.common_project_path(**expected)
    actual = GSuiteAddOnsClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    location = 'mussel'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = GSuiteAddOnsClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'winkle', 'location': 'nautilus'}
    path = GSuiteAddOnsClient.common_location_path(**expected)
    actual = GSuiteAddOnsClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.GSuiteAddOnsTransport, '_prep_wrapped_messages') as prep:
        client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.GSuiteAddOnsTransport, '_prep_wrapped_messages') as prep:
        transport_class = GSuiteAddOnsClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = GSuiteAddOnsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        return 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = GSuiteAddOnsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(GSuiteAddOnsClient, transports.GSuiteAddOnsGrpcTransport), (GSuiteAddOnsAsyncClient, transports.GSuiteAddOnsGrpcAsyncIOTransport)])
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