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
from google.iam.v1 import iam_policy_pb2
from google.iam.v1 import options_pb2
from google.iam.v1 import policy_pb2
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.protobuf import timestamp_pb2
from google.rpc import code_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.eventarc_v1.services.eventarc import EventarcAsyncClient, EventarcClient, pagers, transports
from google.cloud.eventarc_v1.types import channel_connection as gce_channel_connection
from google.cloud.eventarc_v1.types import google_channel_config as gce_google_channel_config
from google.cloud.eventarc_v1.types import channel
from google.cloud.eventarc_v1.types import channel as gce_channel
from google.cloud.eventarc_v1.types import channel_connection
from google.cloud.eventarc_v1.types import discovery, eventarc
from google.cloud.eventarc_v1.types import google_channel_config
from google.cloud.eventarc_v1.types import trigger
from google.cloud.eventarc_v1.types import trigger as gce_trigger

def client_cert_source_callback():
    if False:
        for i in range(10):
            print('nop')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        for i in range(10):
            print('nop')
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
    assert EventarcClient._get_default_mtls_endpoint(None) is None
    assert EventarcClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert EventarcClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert EventarcClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert EventarcClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert EventarcClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(EventarcClient, 'grpc'), (EventarcAsyncClient, 'grpc_asyncio'), (EventarcClient, 'rest')])
def test_eventarc_client_from_service_account_info(client_class, transport_name):
    if False:
        print('Hello World!')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('eventarc.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://eventarc.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.EventarcGrpcTransport, 'grpc'), (transports.EventarcGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.EventarcRestTransport, 'rest')])
def test_eventarc_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(EventarcClient, 'grpc'), (EventarcAsyncClient, 'grpc_asyncio'), (EventarcClient, 'rest')])
def test_eventarc_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('eventarc.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://eventarc.googleapis.com')

def test_eventarc_client_get_transport_class():
    if False:
        return 10
    transport = EventarcClient.get_transport_class()
    available_transports = [transports.EventarcGrpcTransport, transports.EventarcRestTransport]
    assert transport in available_transports
    transport = EventarcClient.get_transport_class('grpc')
    assert transport == transports.EventarcGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(EventarcClient, transports.EventarcGrpcTransport, 'grpc'), (EventarcAsyncClient, transports.EventarcGrpcAsyncIOTransport, 'grpc_asyncio'), (EventarcClient, transports.EventarcRestTransport, 'rest')])
@mock.patch.object(EventarcClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EventarcClient))
@mock.patch.object(EventarcAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EventarcAsyncClient))
def test_eventarc_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(EventarcClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(EventarcClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(EventarcClient, transports.EventarcGrpcTransport, 'grpc', 'true'), (EventarcAsyncClient, transports.EventarcGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (EventarcClient, transports.EventarcGrpcTransport, 'grpc', 'false'), (EventarcAsyncClient, transports.EventarcGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (EventarcClient, transports.EventarcRestTransport, 'rest', 'true'), (EventarcClient, transports.EventarcRestTransport, 'rest', 'false')])
@mock.patch.object(EventarcClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EventarcClient))
@mock.patch.object(EventarcAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EventarcAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_eventarc_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [EventarcClient, EventarcAsyncClient])
@mock.patch.object(EventarcClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EventarcClient))
@mock.patch.object(EventarcAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(EventarcAsyncClient))
def test_eventarc_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(EventarcClient, transports.EventarcGrpcTransport, 'grpc'), (EventarcAsyncClient, transports.EventarcGrpcAsyncIOTransport, 'grpc_asyncio'), (EventarcClient, transports.EventarcRestTransport, 'rest')])
def test_eventarc_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(EventarcClient, transports.EventarcGrpcTransport, 'grpc', grpc_helpers), (EventarcAsyncClient, transports.EventarcGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (EventarcClient, transports.EventarcRestTransport, 'rest', None)])
def test_eventarc_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        return 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_eventarc_client_client_options_from_dict():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.eventarc_v1.services.eventarc.transports.EventarcGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = EventarcClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(EventarcClient, transports.EventarcGrpcTransport, 'grpc', grpc_helpers), (EventarcAsyncClient, transports.EventarcGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_eventarc_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('eventarc.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='eventarc.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [eventarc.GetTriggerRequest, dict])
def test_get_trigger(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_trigger), '__call__') as call:
        call.return_value = trigger.Trigger(name='name_value', uid='uid_value', service_account='service_account_value', channel='channel_value', etag='etag_value')
        response = client.get_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.GetTriggerRequest()
    assert isinstance(response, trigger.Trigger)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.service_account == 'service_account_value'
    assert response.channel == 'channel_value'
    assert response.etag == 'etag_value'

def test_get_trigger_empty_call():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_trigger), '__call__') as call:
        client.get_trigger()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.GetTriggerRequest()

@pytest.mark.asyncio
async def test_get_trigger_async(transport: str='grpc_asyncio', request_type=eventarc.GetTriggerRequest):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(trigger.Trigger(name='name_value', uid='uid_value', service_account='service_account_value', channel='channel_value', etag='etag_value'))
        response = await client.get_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.GetTriggerRequest()
    assert isinstance(response, trigger.Trigger)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.service_account == 'service_account_value'
    assert response.channel == 'channel_value'
    assert response.etag == 'etag_value'

@pytest.mark.asyncio
async def test_get_trigger_async_from_dict():
    await test_get_trigger_async(request_type=dict)

def test_get_trigger_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.GetTriggerRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_trigger), '__call__') as call:
        call.return_value = trigger.Trigger()
        client.get_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_trigger_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.GetTriggerRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(trigger.Trigger())
        await client.get_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_trigger_flattened():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_trigger), '__call__') as call:
        call.return_value = trigger.Trigger()
        client.get_trigger(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_trigger_flattened_error():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_trigger(eventarc.GetTriggerRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_trigger_flattened_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_trigger), '__call__') as call:
        call.return_value = trigger.Trigger()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(trigger.Trigger())
        response = await client.get_trigger(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_trigger_flattened_error_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_trigger(eventarc.GetTriggerRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [eventarc.ListTriggersRequest, dict])
def test_list_triggers(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_triggers), '__call__') as call:
        call.return_value = eventarc.ListTriggersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_triggers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.ListTriggersRequest()
    assert isinstance(response, pagers.ListTriggersPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_triggers_empty_call():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_triggers), '__call__') as call:
        client.list_triggers()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.ListTriggersRequest()

@pytest.mark.asyncio
async def test_list_triggers_async(transport: str='grpc_asyncio', request_type=eventarc.ListTriggersRequest):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_triggers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(eventarc.ListTriggersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_triggers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.ListTriggersRequest()
    assert isinstance(response, pagers.ListTriggersAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_triggers_async_from_dict():
    await test_list_triggers_async(request_type=dict)

def test_list_triggers_field_headers():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.ListTriggersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_triggers), '__call__') as call:
        call.return_value = eventarc.ListTriggersResponse()
        client.list_triggers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_triggers_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.ListTriggersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_triggers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(eventarc.ListTriggersResponse())
        await client.list_triggers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_triggers_flattened():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_triggers), '__call__') as call:
        call.return_value = eventarc.ListTriggersResponse()
        client.list_triggers(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_triggers_flattened_error():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_triggers(eventarc.ListTriggersRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_triggers_flattened_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_triggers), '__call__') as call:
        call.return_value = eventarc.ListTriggersResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(eventarc.ListTriggersResponse())
        response = await client.list_triggers(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_triggers_flattened_error_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_triggers(eventarc.ListTriggersRequest(), parent='parent_value')

def test_list_triggers_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_triggers), '__call__') as call:
        call.side_effect = (eventarc.ListTriggersResponse(triggers=[trigger.Trigger(), trigger.Trigger(), trigger.Trigger()], next_page_token='abc'), eventarc.ListTriggersResponse(triggers=[], next_page_token='def'), eventarc.ListTriggersResponse(triggers=[trigger.Trigger()], next_page_token='ghi'), eventarc.ListTriggersResponse(triggers=[trigger.Trigger(), trigger.Trigger()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_triggers(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, trigger.Trigger) for i in results))

def test_list_triggers_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_triggers), '__call__') as call:
        call.side_effect = (eventarc.ListTriggersResponse(triggers=[trigger.Trigger(), trigger.Trigger(), trigger.Trigger()], next_page_token='abc'), eventarc.ListTriggersResponse(triggers=[], next_page_token='def'), eventarc.ListTriggersResponse(triggers=[trigger.Trigger()], next_page_token='ghi'), eventarc.ListTriggersResponse(triggers=[trigger.Trigger(), trigger.Trigger()]), RuntimeError)
        pages = list(client.list_triggers(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_triggers_async_pager():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_triggers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (eventarc.ListTriggersResponse(triggers=[trigger.Trigger(), trigger.Trigger(), trigger.Trigger()], next_page_token='abc'), eventarc.ListTriggersResponse(triggers=[], next_page_token='def'), eventarc.ListTriggersResponse(triggers=[trigger.Trigger()], next_page_token='ghi'), eventarc.ListTriggersResponse(triggers=[trigger.Trigger(), trigger.Trigger()]), RuntimeError)
        async_pager = await client.list_triggers(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, trigger.Trigger) for i in responses))

@pytest.mark.asyncio
async def test_list_triggers_async_pages():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_triggers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (eventarc.ListTriggersResponse(triggers=[trigger.Trigger(), trigger.Trigger(), trigger.Trigger()], next_page_token='abc'), eventarc.ListTriggersResponse(triggers=[], next_page_token='def'), eventarc.ListTriggersResponse(triggers=[trigger.Trigger()], next_page_token='ghi'), eventarc.ListTriggersResponse(triggers=[trigger.Trigger(), trigger.Trigger()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_triggers(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [eventarc.CreateTriggerRequest, dict])
def test_create_trigger(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_trigger), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.CreateTriggerRequest()
    assert isinstance(response, future.Future)

def test_create_trigger_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_trigger), '__call__') as call:
        client.create_trigger()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.CreateTriggerRequest()

@pytest.mark.asyncio
async def test_create_trigger_async(transport: str='grpc_asyncio', request_type=eventarc.CreateTriggerRequest):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.CreateTriggerRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_trigger_async_from_dict():
    await test_create_trigger_async(request_type=dict)

def test_create_trigger_field_headers():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.CreateTriggerRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_trigger), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_trigger_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.CreateTriggerRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_trigger_flattened():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_trigger), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_trigger(parent='parent_value', trigger=gce_trigger.Trigger(name='name_value'), trigger_id='trigger_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].trigger
        mock_val = gce_trigger.Trigger(name='name_value')
        assert arg == mock_val
        arg = args[0].trigger_id
        mock_val = 'trigger_id_value'
        assert arg == mock_val

def test_create_trigger_flattened_error():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_trigger(eventarc.CreateTriggerRequest(), parent='parent_value', trigger=gce_trigger.Trigger(name='name_value'), trigger_id='trigger_id_value')

@pytest.mark.asyncio
async def test_create_trigger_flattened_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_trigger), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_trigger(parent='parent_value', trigger=gce_trigger.Trigger(name='name_value'), trigger_id='trigger_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].trigger
        mock_val = gce_trigger.Trigger(name='name_value')
        assert arg == mock_val
        arg = args[0].trigger_id
        mock_val = 'trigger_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_trigger_flattened_error_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_trigger(eventarc.CreateTriggerRequest(), parent='parent_value', trigger=gce_trigger.Trigger(name='name_value'), trigger_id='trigger_id_value')

@pytest.mark.parametrize('request_type', [eventarc.UpdateTriggerRequest, dict])
def test_update_trigger(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_trigger), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.UpdateTriggerRequest()
    assert isinstance(response, future.Future)

def test_update_trigger_empty_call():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_trigger), '__call__') as call:
        client.update_trigger()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.UpdateTriggerRequest()

@pytest.mark.asyncio
async def test_update_trigger_async(transport: str='grpc_asyncio', request_type=eventarc.UpdateTriggerRequest):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.UpdateTriggerRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_trigger_async_from_dict():
    await test_update_trigger_async(request_type=dict)

def test_update_trigger_field_headers():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.UpdateTriggerRequest()
    request.trigger.name = 'name_value'
    with mock.patch.object(type(client.transport.update_trigger), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'trigger.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_trigger_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.UpdateTriggerRequest()
    request.trigger.name = 'name_value'
    with mock.patch.object(type(client.transport.update_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'trigger.name=name_value') in kw['metadata']

def test_update_trigger_flattened():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_trigger), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_trigger(trigger=gce_trigger.Trigger(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), allow_missing=True)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].trigger
        mock_val = gce_trigger.Trigger(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val
        arg = args[0].allow_missing
        mock_val = True
        assert arg == mock_val

def test_update_trigger_flattened_error():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_trigger(eventarc.UpdateTriggerRequest(), trigger=gce_trigger.Trigger(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), allow_missing=True)

@pytest.mark.asyncio
async def test_update_trigger_flattened_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_trigger), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_trigger(trigger=gce_trigger.Trigger(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), allow_missing=True)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].trigger
        mock_val = gce_trigger.Trigger(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val
        arg = args[0].allow_missing
        mock_val = True
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_trigger_flattened_error_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_trigger(eventarc.UpdateTriggerRequest(), trigger=gce_trigger.Trigger(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), allow_missing=True)

@pytest.mark.parametrize('request_type', [eventarc.DeleteTriggerRequest, dict])
def test_delete_trigger(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_trigger), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.DeleteTriggerRequest()
    assert isinstance(response, future.Future)

def test_delete_trigger_empty_call():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_trigger), '__call__') as call:
        client.delete_trigger()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.DeleteTriggerRequest()

@pytest.mark.asyncio
async def test_delete_trigger_async(transport: str='grpc_asyncio', request_type=eventarc.DeleteTriggerRequest):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.DeleteTriggerRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_trigger_async_from_dict():
    await test_delete_trigger_async(request_type=dict)

def test_delete_trigger_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.DeleteTriggerRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_trigger), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_trigger(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_trigger_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.DeleteTriggerRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_trigger), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_trigger(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_trigger_flattened():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_trigger), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_trigger(name='name_value', allow_missing=True)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].allow_missing
        mock_val = True
        assert arg == mock_val

def test_delete_trigger_flattened_error():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_trigger(eventarc.DeleteTriggerRequest(), name='name_value', allow_missing=True)

@pytest.mark.asyncio
async def test_delete_trigger_flattened_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_trigger), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_trigger(name='name_value', allow_missing=True)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].allow_missing
        mock_val = True
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_trigger_flattened_error_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_trigger(eventarc.DeleteTriggerRequest(), name='name_value', allow_missing=True)

@pytest.mark.parametrize('request_type', [eventarc.GetChannelRequest, dict])
def test_get_channel(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_channel), '__call__') as call:
        call.return_value = channel.Channel(name='name_value', uid='uid_value', provider='provider_value', state=channel.Channel.State.PENDING, activation_token='activation_token_value', crypto_key_name='crypto_key_name_value', pubsub_topic='pubsub_topic_value')
        response = client.get_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.GetChannelRequest()
    assert isinstance(response, channel.Channel)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.provider == 'provider_value'
    assert response.state == channel.Channel.State.PENDING
    assert response.activation_token == 'activation_token_value'
    assert response.crypto_key_name == 'crypto_key_name_value'

def test_get_channel_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_channel), '__call__') as call:
        client.get_channel()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.GetChannelRequest()

@pytest.mark.asyncio
async def test_get_channel_async(transport: str='grpc_asyncio', request_type=eventarc.GetChannelRequest):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_channel), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(channel.Channel(name='name_value', uid='uid_value', provider='provider_value', state=channel.Channel.State.PENDING, activation_token='activation_token_value', crypto_key_name='crypto_key_name_value'))
        response = await client.get_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.GetChannelRequest()
    assert isinstance(response, channel.Channel)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.provider == 'provider_value'
    assert response.state == channel.Channel.State.PENDING
    assert response.activation_token == 'activation_token_value'
    assert response.crypto_key_name == 'crypto_key_name_value'

@pytest.mark.asyncio
async def test_get_channel_async_from_dict():
    await test_get_channel_async(request_type=dict)

def test_get_channel_field_headers():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.GetChannelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_channel), '__call__') as call:
        call.return_value = channel.Channel()
        client.get_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_channel_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.GetChannelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_channel), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(channel.Channel())
        await client.get_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_channel_flattened():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_channel), '__call__') as call:
        call.return_value = channel.Channel()
        client.get_channel(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_channel_flattened_error():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_channel(eventarc.GetChannelRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_channel_flattened_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_channel), '__call__') as call:
        call.return_value = channel.Channel()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(channel.Channel())
        response = await client.get_channel(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_channel_flattened_error_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_channel(eventarc.GetChannelRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [eventarc.ListChannelsRequest, dict])
def test_list_channels(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_channels), '__call__') as call:
        call.return_value = eventarc.ListChannelsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_channels(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.ListChannelsRequest()
    assert isinstance(response, pagers.ListChannelsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_channels_empty_call():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_channels), '__call__') as call:
        client.list_channels()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.ListChannelsRequest()

@pytest.mark.asyncio
async def test_list_channels_async(transport: str='grpc_asyncio', request_type=eventarc.ListChannelsRequest):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_channels), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(eventarc.ListChannelsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_channels(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.ListChannelsRequest()
    assert isinstance(response, pagers.ListChannelsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_channels_async_from_dict():
    await test_list_channels_async(request_type=dict)

def test_list_channels_field_headers():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.ListChannelsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_channels), '__call__') as call:
        call.return_value = eventarc.ListChannelsResponse()
        client.list_channels(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_channels_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.ListChannelsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_channels), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(eventarc.ListChannelsResponse())
        await client.list_channels(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_channels_flattened():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_channels), '__call__') as call:
        call.return_value = eventarc.ListChannelsResponse()
        client.list_channels(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_channels_flattened_error():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_channels(eventarc.ListChannelsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_channels_flattened_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_channels), '__call__') as call:
        call.return_value = eventarc.ListChannelsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(eventarc.ListChannelsResponse())
        response = await client.list_channels(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_channels_flattened_error_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_channels(eventarc.ListChannelsRequest(), parent='parent_value')

def test_list_channels_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_channels), '__call__') as call:
        call.side_effect = (eventarc.ListChannelsResponse(channels=[channel.Channel(), channel.Channel(), channel.Channel()], next_page_token='abc'), eventarc.ListChannelsResponse(channels=[], next_page_token='def'), eventarc.ListChannelsResponse(channels=[channel.Channel()], next_page_token='ghi'), eventarc.ListChannelsResponse(channels=[channel.Channel(), channel.Channel()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_channels(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, channel.Channel) for i in results))

def test_list_channels_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_channels), '__call__') as call:
        call.side_effect = (eventarc.ListChannelsResponse(channels=[channel.Channel(), channel.Channel(), channel.Channel()], next_page_token='abc'), eventarc.ListChannelsResponse(channels=[], next_page_token='def'), eventarc.ListChannelsResponse(channels=[channel.Channel()], next_page_token='ghi'), eventarc.ListChannelsResponse(channels=[channel.Channel(), channel.Channel()]), RuntimeError)
        pages = list(client.list_channels(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_channels_async_pager():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_channels), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (eventarc.ListChannelsResponse(channels=[channel.Channel(), channel.Channel(), channel.Channel()], next_page_token='abc'), eventarc.ListChannelsResponse(channels=[], next_page_token='def'), eventarc.ListChannelsResponse(channels=[channel.Channel()], next_page_token='ghi'), eventarc.ListChannelsResponse(channels=[channel.Channel(), channel.Channel()]), RuntimeError)
        async_pager = await client.list_channels(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, channel.Channel) for i in responses))

@pytest.mark.asyncio
async def test_list_channels_async_pages():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_channels), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (eventarc.ListChannelsResponse(channels=[channel.Channel(), channel.Channel(), channel.Channel()], next_page_token='abc'), eventarc.ListChannelsResponse(channels=[], next_page_token='def'), eventarc.ListChannelsResponse(channels=[channel.Channel()], next_page_token='ghi'), eventarc.ListChannelsResponse(channels=[channel.Channel(), channel.Channel()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_channels(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [eventarc.CreateChannelRequest, dict])
def test_create_channel(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_channel_), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.CreateChannelRequest()
    assert isinstance(response, future.Future)

def test_create_channel_empty_call():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_channel_), '__call__') as call:
        client.create_channel()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.CreateChannelRequest()

@pytest.mark.asyncio
async def test_create_channel_async(transport: str='grpc_asyncio', request_type=eventarc.CreateChannelRequest):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_channel_), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.CreateChannelRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_channel_async_from_dict():
    await test_create_channel_async(request_type=dict)

def test_create_channel_field_headers():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.CreateChannelRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_channel_), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_channel_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.CreateChannelRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_channel_), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_channel_flattened():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_channel_), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_channel(parent='parent_value', channel=gce_channel.Channel(name='name_value'), channel_id='channel_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].channel
        mock_val = gce_channel.Channel(name='name_value')
        assert arg == mock_val
        arg = args[0].channel_id
        mock_val = 'channel_id_value'
        assert arg == mock_val

def test_create_channel_flattened_error():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_channel(eventarc.CreateChannelRequest(), parent='parent_value', channel=gce_channel.Channel(name='name_value'), channel_id='channel_id_value')

@pytest.mark.asyncio
async def test_create_channel_flattened_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_channel_), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_channel(parent='parent_value', channel=gce_channel.Channel(name='name_value'), channel_id='channel_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].channel
        mock_val = gce_channel.Channel(name='name_value')
        assert arg == mock_val
        arg = args[0].channel_id
        mock_val = 'channel_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_channel_flattened_error_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_channel(eventarc.CreateChannelRequest(), parent='parent_value', channel=gce_channel.Channel(name='name_value'), channel_id='channel_id_value')

@pytest.mark.parametrize('request_type', [eventarc.UpdateChannelRequest, dict])
def test_update_channel(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.update_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.UpdateChannelRequest()
    assert isinstance(response, future.Future)

def test_update_channel_empty_call():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_channel), '__call__') as call:
        client.update_channel()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.UpdateChannelRequest()

@pytest.mark.asyncio
async def test_update_channel_async(transport: str='grpc_asyncio', request_type=eventarc.UpdateChannelRequest):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_channel), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.UpdateChannelRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_update_channel_async_from_dict():
    await test_update_channel_async(request_type=dict)

def test_update_channel_field_headers():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.UpdateChannelRequest()
    request.channel.name = 'name_value'
    with mock.patch.object(type(client.transport.update_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'channel.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_channel_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.UpdateChannelRequest()
    request.channel.name = 'name_value'
    with mock.patch.object(type(client.transport.update_channel), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.update_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'channel.name=name_value') in kw['metadata']

def test_update_channel_flattened():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.update_channel(channel=gce_channel.Channel(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].channel
        mock_val = gce_channel.Channel(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_channel_flattened_error():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_channel(eventarc.UpdateChannelRequest(), channel=gce_channel.Channel(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_channel_flattened_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.update_channel(channel=gce_channel.Channel(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].channel
        mock_val = gce_channel.Channel(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_channel_flattened_error_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_channel(eventarc.UpdateChannelRequest(), channel=gce_channel.Channel(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [eventarc.DeleteChannelRequest, dict])
def test_delete_channel(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.DeleteChannelRequest()
    assert isinstance(response, future.Future)

def test_delete_channel_empty_call():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_channel), '__call__') as call:
        client.delete_channel()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.DeleteChannelRequest()

@pytest.mark.asyncio
async def test_delete_channel_async(transport: str='grpc_asyncio', request_type=eventarc.DeleteChannelRequest):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_channel), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.DeleteChannelRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_channel_async_from_dict():
    await test_delete_channel_async(request_type=dict)

def test_delete_channel_field_headers():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.DeleteChannelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_channel(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_channel_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.DeleteChannelRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_channel), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_channel(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_channel_flattened():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_channel(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_channel_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_channel(eventarc.DeleteChannelRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_channel_flattened_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_channel), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_channel(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_channel_flattened_error_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_channel(eventarc.DeleteChannelRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [eventarc.GetProviderRequest, dict])
def test_get_provider(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_provider), '__call__') as call:
        call.return_value = discovery.Provider(name='name_value', display_name='display_name_value')
        response = client.get_provider(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.GetProviderRequest()
    assert isinstance(response, discovery.Provider)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_get_provider_empty_call():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_provider), '__call__') as call:
        client.get_provider()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.GetProviderRequest()

@pytest.mark.asyncio
async def test_get_provider_async(transport: str='grpc_asyncio', request_type=eventarc.GetProviderRequest):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_provider), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(discovery.Provider(name='name_value', display_name='display_name_value'))
        response = await client.get_provider(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.GetProviderRequest()
    assert isinstance(response, discovery.Provider)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

@pytest.mark.asyncio
async def test_get_provider_async_from_dict():
    await test_get_provider_async(request_type=dict)

def test_get_provider_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.GetProviderRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_provider), '__call__') as call:
        call.return_value = discovery.Provider()
        client.get_provider(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_provider_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.GetProviderRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_provider), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(discovery.Provider())
        await client.get_provider(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_provider_flattened():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_provider), '__call__') as call:
        call.return_value = discovery.Provider()
        client.get_provider(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_provider_flattened_error():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_provider(eventarc.GetProviderRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_provider_flattened_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_provider), '__call__') as call:
        call.return_value = discovery.Provider()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(discovery.Provider())
        response = await client.get_provider(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_provider_flattened_error_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_provider(eventarc.GetProviderRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [eventarc.ListProvidersRequest, dict])
def test_list_providers(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_providers), '__call__') as call:
        call.return_value = eventarc.ListProvidersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_providers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.ListProvidersRequest()
    assert isinstance(response, pagers.ListProvidersPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_providers_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_providers), '__call__') as call:
        client.list_providers()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.ListProvidersRequest()

@pytest.mark.asyncio
async def test_list_providers_async(transport: str='grpc_asyncio', request_type=eventarc.ListProvidersRequest):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_providers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(eventarc.ListProvidersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_providers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.ListProvidersRequest()
    assert isinstance(response, pagers.ListProvidersAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_providers_async_from_dict():
    await test_list_providers_async(request_type=dict)

def test_list_providers_field_headers():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.ListProvidersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_providers), '__call__') as call:
        call.return_value = eventarc.ListProvidersResponse()
        client.list_providers(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_providers_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.ListProvidersRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_providers), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(eventarc.ListProvidersResponse())
        await client.list_providers(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_providers_flattened():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_providers), '__call__') as call:
        call.return_value = eventarc.ListProvidersResponse()
        client.list_providers(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_providers_flattened_error():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_providers(eventarc.ListProvidersRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_providers_flattened_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_providers), '__call__') as call:
        call.return_value = eventarc.ListProvidersResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(eventarc.ListProvidersResponse())
        response = await client.list_providers(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_providers_flattened_error_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_providers(eventarc.ListProvidersRequest(), parent='parent_value')

def test_list_providers_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_providers), '__call__') as call:
        call.side_effect = (eventarc.ListProvidersResponse(providers=[discovery.Provider(), discovery.Provider(), discovery.Provider()], next_page_token='abc'), eventarc.ListProvidersResponse(providers=[], next_page_token='def'), eventarc.ListProvidersResponse(providers=[discovery.Provider()], next_page_token='ghi'), eventarc.ListProvidersResponse(providers=[discovery.Provider(), discovery.Provider()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_providers(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, discovery.Provider) for i in results))

def test_list_providers_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_providers), '__call__') as call:
        call.side_effect = (eventarc.ListProvidersResponse(providers=[discovery.Provider(), discovery.Provider(), discovery.Provider()], next_page_token='abc'), eventarc.ListProvidersResponse(providers=[], next_page_token='def'), eventarc.ListProvidersResponse(providers=[discovery.Provider()], next_page_token='ghi'), eventarc.ListProvidersResponse(providers=[discovery.Provider(), discovery.Provider()]), RuntimeError)
        pages = list(client.list_providers(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_providers_async_pager():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_providers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (eventarc.ListProvidersResponse(providers=[discovery.Provider(), discovery.Provider(), discovery.Provider()], next_page_token='abc'), eventarc.ListProvidersResponse(providers=[], next_page_token='def'), eventarc.ListProvidersResponse(providers=[discovery.Provider()], next_page_token='ghi'), eventarc.ListProvidersResponse(providers=[discovery.Provider(), discovery.Provider()]), RuntimeError)
        async_pager = await client.list_providers(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, discovery.Provider) for i in responses))

@pytest.mark.asyncio
async def test_list_providers_async_pages():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_providers), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (eventarc.ListProvidersResponse(providers=[discovery.Provider(), discovery.Provider(), discovery.Provider()], next_page_token='abc'), eventarc.ListProvidersResponse(providers=[], next_page_token='def'), eventarc.ListProvidersResponse(providers=[discovery.Provider()], next_page_token='ghi'), eventarc.ListProvidersResponse(providers=[discovery.Provider(), discovery.Provider()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_providers(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [eventarc.GetChannelConnectionRequest, dict])
def test_get_channel_connection(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_channel_connection), '__call__') as call:
        call.return_value = channel_connection.ChannelConnection(name='name_value', uid='uid_value', channel='channel_value', activation_token='activation_token_value')
        response = client.get_channel_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.GetChannelConnectionRequest()
    assert isinstance(response, channel_connection.ChannelConnection)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.channel == 'channel_value'
    assert response.activation_token == 'activation_token_value'

def test_get_channel_connection_empty_call():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_channel_connection), '__call__') as call:
        client.get_channel_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.GetChannelConnectionRequest()

@pytest.mark.asyncio
async def test_get_channel_connection_async(transport: str='grpc_asyncio', request_type=eventarc.GetChannelConnectionRequest):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_channel_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(channel_connection.ChannelConnection(name='name_value', uid='uid_value', channel='channel_value', activation_token='activation_token_value'))
        response = await client.get_channel_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.GetChannelConnectionRequest()
    assert isinstance(response, channel_connection.ChannelConnection)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.channel == 'channel_value'
    assert response.activation_token == 'activation_token_value'

@pytest.mark.asyncio
async def test_get_channel_connection_async_from_dict():
    await test_get_channel_connection_async(request_type=dict)

def test_get_channel_connection_field_headers():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.GetChannelConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_channel_connection), '__call__') as call:
        call.return_value = channel_connection.ChannelConnection()
        client.get_channel_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_channel_connection_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.GetChannelConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_channel_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(channel_connection.ChannelConnection())
        await client.get_channel_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_channel_connection_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_channel_connection), '__call__') as call:
        call.return_value = channel_connection.ChannelConnection()
        client.get_channel_connection(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_channel_connection_flattened_error():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_channel_connection(eventarc.GetChannelConnectionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_channel_connection_flattened_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_channel_connection), '__call__') as call:
        call.return_value = channel_connection.ChannelConnection()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(channel_connection.ChannelConnection())
        response = await client.get_channel_connection(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_channel_connection_flattened_error_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_channel_connection(eventarc.GetChannelConnectionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [eventarc.ListChannelConnectionsRequest, dict])
def test_list_channel_connections(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_channel_connections), '__call__') as call:
        call.return_value = eventarc.ListChannelConnectionsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response = client.list_channel_connections(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.ListChannelConnectionsRequest()
    assert isinstance(response, pagers.ListChannelConnectionsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_channel_connections_empty_call():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_channel_connections), '__call__') as call:
        client.list_channel_connections()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.ListChannelConnectionsRequest()

@pytest.mark.asyncio
async def test_list_channel_connections_async(transport: str='grpc_asyncio', request_type=eventarc.ListChannelConnectionsRequest):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_channel_connections), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(eventarc.ListChannelConnectionsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value']))
        response = await client.list_channel_connections(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.ListChannelConnectionsRequest()
    assert isinstance(response, pagers.ListChannelConnectionsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

@pytest.mark.asyncio
async def test_list_channel_connections_async_from_dict():
    await test_list_channel_connections_async(request_type=dict)

def test_list_channel_connections_field_headers():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.ListChannelConnectionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_channel_connections), '__call__') as call:
        call.return_value = eventarc.ListChannelConnectionsResponse()
        client.list_channel_connections(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_channel_connections_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.ListChannelConnectionsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_channel_connections), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(eventarc.ListChannelConnectionsResponse())
        await client.list_channel_connections(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_channel_connections_flattened():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_channel_connections), '__call__') as call:
        call.return_value = eventarc.ListChannelConnectionsResponse()
        client.list_channel_connections(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_channel_connections_flattened_error():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_channel_connections(eventarc.ListChannelConnectionsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_channel_connections_flattened_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_channel_connections), '__call__') as call:
        call.return_value = eventarc.ListChannelConnectionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(eventarc.ListChannelConnectionsResponse())
        response = await client.list_channel_connections(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_channel_connections_flattened_error_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_channel_connections(eventarc.ListChannelConnectionsRequest(), parent='parent_value')

def test_list_channel_connections_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_channel_connections), '__call__') as call:
        call.side_effect = (eventarc.ListChannelConnectionsResponse(channel_connections=[channel_connection.ChannelConnection(), channel_connection.ChannelConnection(), channel_connection.ChannelConnection()], next_page_token='abc'), eventarc.ListChannelConnectionsResponse(channel_connections=[], next_page_token='def'), eventarc.ListChannelConnectionsResponse(channel_connections=[channel_connection.ChannelConnection()], next_page_token='ghi'), eventarc.ListChannelConnectionsResponse(channel_connections=[channel_connection.ChannelConnection(), channel_connection.ChannelConnection()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_channel_connections(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, channel_connection.ChannelConnection) for i in results))

def test_list_channel_connections_pages(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_channel_connections), '__call__') as call:
        call.side_effect = (eventarc.ListChannelConnectionsResponse(channel_connections=[channel_connection.ChannelConnection(), channel_connection.ChannelConnection(), channel_connection.ChannelConnection()], next_page_token='abc'), eventarc.ListChannelConnectionsResponse(channel_connections=[], next_page_token='def'), eventarc.ListChannelConnectionsResponse(channel_connections=[channel_connection.ChannelConnection()], next_page_token='ghi'), eventarc.ListChannelConnectionsResponse(channel_connections=[channel_connection.ChannelConnection(), channel_connection.ChannelConnection()]), RuntimeError)
        pages = list(client.list_channel_connections(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_channel_connections_async_pager():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_channel_connections), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (eventarc.ListChannelConnectionsResponse(channel_connections=[channel_connection.ChannelConnection(), channel_connection.ChannelConnection(), channel_connection.ChannelConnection()], next_page_token='abc'), eventarc.ListChannelConnectionsResponse(channel_connections=[], next_page_token='def'), eventarc.ListChannelConnectionsResponse(channel_connections=[channel_connection.ChannelConnection()], next_page_token='ghi'), eventarc.ListChannelConnectionsResponse(channel_connections=[channel_connection.ChannelConnection(), channel_connection.ChannelConnection()]), RuntimeError)
        async_pager = await client.list_channel_connections(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, channel_connection.ChannelConnection) for i in responses))

@pytest.mark.asyncio
async def test_list_channel_connections_async_pages():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_channel_connections), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (eventarc.ListChannelConnectionsResponse(channel_connections=[channel_connection.ChannelConnection(), channel_connection.ChannelConnection(), channel_connection.ChannelConnection()], next_page_token='abc'), eventarc.ListChannelConnectionsResponse(channel_connections=[], next_page_token='def'), eventarc.ListChannelConnectionsResponse(channel_connections=[channel_connection.ChannelConnection()], next_page_token='ghi'), eventarc.ListChannelConnectionsResponse(channel_connections=[channel_connection.ChannelConnection(), channel_connection.ChannelConnection()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_channel_connections(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [eventarc.CreateChannelConnectionRequest, dict])
def test_create_channel_connection(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_channel_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_channel_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.CreateChannelConnectionRequest()
    assert isinstance(response, future.Future)

def test_create_channel_connection_empty_call():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_channel_connection), '__call__') as call:
        client.create_channel_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.CreateChannelConnectionRequest()

@pytest.mark.asyncio
async def test_create_channel_connection_async(transport: str='grpc_asyncio', request_type=eventarc.CreateChannelConnectionRequest):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_channel_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_channel_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.CreateChannelConnectionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_channel_connection_async_from_dict():
    await test_create_channel_connection_async(request_type=dict)

def test_create_channel_connection_field_headers():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.CreateChannelConnectionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_channel_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_channel_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_channel_connection_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.CreateChannelConnectionRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_channel_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_channel_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_channel_connection_flattened():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_channel_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_channel_connection(parent='parent_value', channel_connection=gce_channel_connection.ChannelConnection(name='name_value'), channel_connection_id='channel_connection_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].channel_connection
        mock_val = gce_channel_connection.ChannelConnection(name='name_value')
        assert arg == mock_val
        arg = args[0].channel_connection_id
        mock_val = 'channel_connection_id_value'
        assert arg == mock_val

def test_create_channel_connection_flattened_error():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_channel_connection(eventarc.CreateChannelConnectionRequest(), parent='parent_value', channel_connection=gce_channel_connection.ChannelConnection(name='name_value'), channel_connection_id='channel_connection_id_value')

@pytest.mark.asyncio
async def test_create_channel_connection_flattened_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_channel_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_channel_connection(parent='parent_value', channel_connection=gce_channel_connection.ChannelConnection(name='name_value'), channel_connection_id='channel_connection_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].channel_connection
        mock_val = gce_channel_connection.ChannelConnection(name='name_value')
        assert arg == mock_val
        arg = args[0].channel_connection_id
        mock_val = 'channel_connection_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_channel_connection_flattened_error_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_channel_connection(eventarc.CreateChannelConnectionRequest(), parent='parent_value', channel_connection=gce_channel_connection.ChannelConnection(name='name_value'), channel_connection_id='channel_connection_id_value')

@pytest.mark.parametrize('request_type', [eventarc.DeleteChannelConnectionRequest, dict])
def test_delete_channel_connection(request_type, transport: str='grpc'):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_channel_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_channel_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.DeleteChannelConnectionRequest()
    assert isinstance(response, future.Future)

def test_delete_channel_connection_empty_call():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_channel_connection), '__call__') as call:
        client.delete_channel_connection()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.DeleteChannelConnectionRequest()

@pytest.mark.asyncio
async def test_delete_channel_connection_async(transport: str='grpc_asyncio', request_type=eventarc.DeleteChannelConnectionRequest):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_channel_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_channel_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.DeleteChannelConnectionRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_channel_connection_async_from_dict():
    await test_delete_channel_connection_async(request_type=dict)

def test_delete_channel_connection_field_headers():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.DeleteChannelConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_channel_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_channel_connection(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_channel_connection_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.DeleteChannelConnectionRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_channel_connection), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_channel_connection(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_channel_connection_flattened():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_channel_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_channel_connection(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_channel_connection_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_channel_connection(eventarc.DeleteChannelConnectionRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_channel_connection_flattened_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_channel_connection), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_channel_connection(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_channel_connection_flattened_error_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_channel_connection(eventarc.DeleteChannelConnectionRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [eventarc.GetGoogleChannelConfigRequest, dict])
def test_get_google_channel_config(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_google_channel_config), '__call__') as call:
        call.return_value = google_channel_config.GoogleChannelConfig(name='name_value', crypto_key_name='crypto_key_name_value')
        response = client.get_google_channel_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.GetGoogleChannelConfigRequest()
    assert isinstance(response, google_channel_config.GoogleChannelConfig)
    assert response.name == 'name_value'
    assert response.crypto_key_name == 'crypto_key_name_value'

def test_get_google_channel_config_empty_call():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_google_channel_config), '__call__') as call:
        client.get_google_channel_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.GetGoogleChannelConfigRequest()

@pytest.mark.asyncio
async def test_get_google_channel_config_async(transport: str='grpc_asyncio', request_type=eventarc.GetGoogleChannelConfigRequest):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_google_channel_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(google_channel_config.GoogleChannelConfig(name='name_value', crypto_key_name='crypto_key_name_value'))
        response = await client.get_google_channel_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.GetGoogleChannelConfigRequest()
    assert isinstance(response, google_channel_config.GoogleChannelConfig)
    assert response.name == 'name_value'
    assert response.crypto_key_name == 'crypto_key_name_value'

@pytest.mark.asyncio
async def test_get_google_channel_config_async_from_dict():
    await test_get_google_channel_config_async(request_type=dict)

def test_get_google_channel_config_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.GetGoogleChannelConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_google_channel_config), '__call__') as call:
        call.return_value = google_channel_config.GoogleChannelConfig()
        client.get_google_channel_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_google_channel_config_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.GetGoogleChannelConfigRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_google_channel_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(google_channel_config.GoogleChannelConfig())
        await client.get_google_channel_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_google_channel_config_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_google_channel_config), '__call__') as call:
        call.return_value = google_channel_config.GoogleChannelConfig()
        client.get_google_channel_config(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_google_channel_config_flattened_error():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_google_channel_config(eventarc.GetGoogleChannelConfigRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_google_channel_config_flattened_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_google_channel_config), '__call__') as call:
        call.return_value = google_channel_config.GoogleChannelConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(google_channel_config.GoogleChannelConfig())
        response = await client.get_google_channel_config(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_google_channel_config_flattened_error_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_google_channel_config(eventarc.GetGoogleChannelConfigRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [eventarc.UpdateGoogleChannelConfigRequest, dict])
def test_update_google_channel_config(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_google_channel_config), '__call__') as call:
        call.return_value = gce_google_channel_config.GoogleChannelConfig(name='name_value', crypto_key_name='crypto_key_name_value')
        response = client.update_google_channel_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.UpdateGoogleChannelConfigRequest()
    assert isinstance(response, gce_google_channel_config.GoogleChannelConfig)
    assert response.name == 'name_value'
    assert response.crypto_key_name == 'crypto_key_name_value'

def test_update_google_channel_config_empty_call():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_google_channel_config), '__call__') as call:
        client.update_google_channel_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.UpdateGoogleChannelConfigRequest()

@pytest.mark.asyncio
async def test_update_google_channel_config_async(transport: str='grpc_asyncio', request_type=eventarc.UpdateGoogleChannelConfigRequest):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_google_channel_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gce_google_channel_config.GoogleChannelConfig(name='name_value', crypto_key_name='crypto_key_name_value'))
        response = await client.update_google_channel_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == eventarc.UpdateGoogleChannelConfigRequest()
    assert isinstance(response, gce_google_channel_config.GoogleChannelConfig)
    assert response.name == 'name_value'
    assert response.crypto_key_name == 'crypto_key_name_value'

@pytest.mark.asyncio
async def test_update_google_channel_config_async_from_dict():
    await test_update_google_channel_config_async(request_type=dict)

def test_update_google_channel_config_field_headers():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.UpdateGoogleChannelConfigRequest()
    request.google_channel_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_google_channel_config), '__call__') as call:
        call.return_value = gce_google_channel_config.GoogleChannelConfig()
        client.update_google_channel_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'google_channel_config.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_google_channel_config_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = eventarc.UpdateGoogleChannelConfigRequest()
    request.google_channel_config.name = 'name_value'
    with mock.patch.object(type(client.transport.update_google_channel_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gce_google_channel_config.GoogleChannelConfig())
        await client.update_google_channel_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'google_channel_config.name=name_value') in kw['metadata']

def test_update_google_channel_config_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_google_channel_config), '__call__') as call:
        call.return_value = gce_google_channel_config.GoogleChannelConfig()
        client.update_google_channel_config(google_channel_config=gce_google_channel_config.GoogleChannelConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].google_channel_config
        mock_val = gce_google_channel_config.GoogleChannelConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_google_channel_config_flattened_error():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_google_channel_config(eventarc.UpdateGoogleChannelConfigRequest(), google_channel_config=gce_google_channel_config.GoogleChannelConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_google_channel_config_flattened_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_google_channel_config), '__call__') as call:
        call.return_value = gce_google_channel_config.GoogleChannelConfig()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gce_google_channel_config.GoogleChannelConfig())
        response = await client.update_google_channel_config(google_channel_config=gce_google_channel_config.GoogleChannelConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].google_channel_config
        mock_val = gce_google_channel_config.GoogleChannelConfig(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_google_channel_config_flattened_error_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_google_channel_config(eventarc.UpdateGoogleChannelConfigRequest(), google_channel_config=gce_google_channel_config.GoogleChannelConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [eventarc.GetTriggerRequest, dict])
def test_get_trigger_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/triggers/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = trigger.Trigger(name='name_value', uid='uid_value', service_account='service_account_value', channel='channel_value', etag='etag_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = trigger.Trigger.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_trigger(request)
    assert isinstance(response, trigger.Trigger)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.service_account == 'service_account_value'
    assert response.channel == 'channel_value'
    assert response.etag == 'etag_value'

def test_get_trigger_rest_required_fields(request_type=eventarc.GetTriggerRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.EventarcRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_trigger._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_trigger._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = trigger.Trigger()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = trigger.Trigger.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_trigger(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_trigger_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_trigger._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_trigger_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EventarcRestInterceptor())
    client = EventarcClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EventarcRestInterceptor, 'post_get_trigger') as post, mock.patch.object(transports.EventarcRestInterceptor, 'pre_get_trigger') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = eventarc.GetTriggerRequest.pb(eventarc.GetTriggerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = trigger.Trigger.to_json(trigger.Trigger())
        request = eventarc.GetTriggerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = trigger.Trigger()
        client.get_trigger(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_trigger_rest_bad_request(transport: str='rest', request_type=eventarc.GetTriggerRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/triggers/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_trigger(request)

def test_get_trigger_rest_flattened():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = trigger.Trigger()
        sample_request = {'name': 'projects/sample1/locations/sample2/triggers/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = trigger.Trigger.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_trigger(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/triggers/*}' % client.transport._host, args[1])

def test_get_trigger_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_trigger(eventarc.GetTriggerRequest(), name='name_value')

def test_get_trigger_rest_error():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [eventarc.ListTriggersRequest, dict])
def test_list_triggers_rest(request_type):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = eventarc.ListTriggersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = eventarc.ListTriggersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_triggers(request)
    assert isinstance(response, pagers.ListTriggersPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_triggers_rest_required_fields(request_type=eventarc.ListTriggersRequest):
    if False:
        return 10
    transport_class = transports.EventarcRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_triggers._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_triggers._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = eventarc.ListTriggersResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = eventarc.ListTriggersResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_triggers(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_triggers_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_triggers._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_triggers_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EventarcRestInterceptor())
    client = EventarcClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EventarcRestInterceptor, 'post_list_triggers') as post, mock.patch.object(transports.EventarcRestInterceptor, 'pre_list_triggers') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = eventarc.ListTriggersRequest.pb(eventarc.ListTriggersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = eventarc.ListTriggersResponse.to_json(eventarc.ListTriggersResponse())
        request = eventarc.ListTriggersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = eventarc.ListTriggersResponse()
        client.list_triggers(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_triggers_rest_bad_request(transport: str='rest', request_type=eventarc.ListTriggersRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_triggers(request)

def test_list_triggers_rest_flattened():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = eventarc.ListTriggersResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = eventarc.ListTriggersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_triggers(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/triggers' % client.transport._host, args[1])

def test_list_triggers_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_triggers(eventarc.ListTriggersRequest(), parent='parent_value')

def test_list_triggers_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (eventarc.ListTriggersResponse(triggers=[trigger.Trigger(), trigger.Trigger(), trigger.Trigger()], next_page_token='abc'), eventarc.ListTriggersResponse(triggers=[], next_page_token='def'), eventarc.ListTriggersResponse(triggers=[trigger.Trigger()], next_page_token='ghi'), eventarc.ListTriggersResponse(triggers=[trigger.Trigger(), trigger.Trigger()]))
        response = response + response
        response = tuple((eventarc.ListTriggersResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_triggers(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, trigger.Trigger) for i in results))
        pages = list(client.list_triggers(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [eventarc.CreateTriggerRequest, dict])
def test_create_trigger_rest(request_type):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['trigger'] = {'name': 'name_value', 'uid': 'uid_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'event_filters': [{'attribute': 'attribute_value', 'value': 'value_value', 'operator': 'operator_value'}], 'service_account': 'service_account_value', 'destination': {'cloud_run': {'service': 'service_value', 'path': 'path_value', 'region': 'region_value'}, 'cloud_function': 'cloud_function_value', 'gke': {'cluster': 'cluster_value', 'location': 'location_value', 'namespace': 'namespace_value', 'service': 'service_value', 'path': 'path_value'}, 'workflow': 'workflow_value'}, 'transport': {'pubsub': {'topic': 'topic_value', 'subscription': 'subscription_value'}}, 'labels': {}, 'channel': 'channel_value', 'conditions': {}, 'etag': 'etag_value'}
    test_field = eventarc.CreateTriggerRequest.meta.fields['trigger']

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
    for (field, value) in request_init['trigger'].items():
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
                for i in range(0, len(request_init['trigger'][field])):
                    del request_init['trigger'][field][i][subfield]
            else:
                del request_init['trigger'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_trigger(request)
    assert response.operation.name == 'operations/spam'

def test_create_trigger_rest_required_fields(request_type=eventarc.CreateTriggerRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EventarcRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['trigger_id'] = ''
    request_init['validate_only'] = False
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'triggerId' not in jsonified_request
    assert 'validateOnly' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_trigger._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'triggerId' in jsonified_request
    assert jsonified_request['triggerId'] == request_init['trigger_id']
    assert 'validateOnly' in jsonified_request
    assert jsonified_request['validateOnly'] == request_init['validate_only']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['triggerId'] = 'trigger_id_value'
    jsonified_request['validateOnly'] = True
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_trigger._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('trigger_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'triggerId' in jsonified_request
    assert jsonified_request['triggerId'] == 'trigger_id_value'
    assert 'validateOnly' in jsonified_request
    assert jsonified_request['validateOnly'] == True
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_trigger(request)
            expected_params = [('triggerId', ''), ('validateOnly', str(False).lower()), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_trigger_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_trigger._get_unset_required_fields({})
    assert set(unset_fields) == set(('triggerId', 'validateOnly')) & set(('parent', 'trigger', 'triggerId', 'validateOnly'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_trigger_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EventarcRestInterceptor())
    client = EventarcClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EventarcRestInterceptor, 'post_create_trigger') as post, mock.patch.object(transports.EventarcRestInterceptor, 'pre_create_trigger') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = eventarc.CreateTriggerRequest.pb(eventarc.CreateTriggerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = eventarc.CreateTriggerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_trigger(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_trigger_rest_bad_request(transport: str='rest', request_type=eventarc.CreateTriggerRequest):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_trigger(request)

def test_create_trigger_rest_flattened():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', trigger=gce_trigger.Trigger(name='name_value'), trigger_id='trigger_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_trigger(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/triggers' % client.transport._host, args[1])

def test_create_trigger_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_trigger(eventarc.CreateTriggerRequest(), parent='parent_value', trigger=gce_trigger.Trigger(name='name_value'), trigger_id='trigger_id_value')

def test_create_trigger_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [eventarc.UpdateTriggerRequest, dict])
def test_update_trigger_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'trigger': {'name': 'projects/sample1/locations/sample2/triggers/sample3'}}
    request_init['trigger'] = {'name': 'projects/sample1/locations/sample2/triggers/sample3', 'uid': 'uid_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'event_filters': [{'attribute': 'attribute_value', 'value': 'value_value', 'operator': 'operator_value'}], 'service_account': 'service_account_value', 'destination': {'cloud_run': {'service': 'service_value', 'path': 'path_value', 'region': 'region_value'}, 'cloud_function': 'cloud_function_value', 'gke': {'cluster': 'cluster_value', 'location': 'location_value', 'namespace': 'namespace_value', 'service': 'service_value', 'path': 'path_value'}, 'workflow': 'workflow_value'}, 'transport': {'pubsub': {'topic': 'topic_value', 'subscription': 'subscription_value'}}, 'labels': {}, 'channel': 'channel_value', 'conditions': {}, 'etag': 'etag_value'}
    test_field = eventarc.UpdateTriggerRequest.meta.fields['trigger']

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
    for (field, value) in request_init['trigger'].items():
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
                for i in range(0, len(request_init['trigger'][field])):
                    del request_init['trigger'][field][i][subfield]
            else:
                del request_init['trigger'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_trigger(request)
    assert response.operation.name == 'operations/spam'

def test_update_trigger_rest_required_fields(request_type=eventarc.UpdateTriggerRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.EventarcRestTransport
    request_init = {}
    request_init['validate_only'] = False
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'validateOnly' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_trigger._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'validateOnly' in jsonified_request
    assert jsonified_request['validateOnly'] == request_init['validate_only']
    jsonified_request['validateOnly'] = True
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_trigger._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'validateOnly' in jsonified_request
    assert jsonified_request['validateOnly'] == True
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_trigger(request)
            expected_params = [('validateOnly', str(False).lower()), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_trigger_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_trigger._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'updateMask', 'validateOnly')) & set(('validateOnly',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_trigger_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EventarcRestInterceptor())
    client = EventarcClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EventarcRestInterceptor, 'post_update_trigger') as post, mock.patch.object(transports.EventarcRestInterceptor, 'pre_update_trigger') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = eventarc.UpdateTriggerRequest.pb(eventarc.UpdateTriggerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = eventarc.UpdateTriggerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_trigger(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_trigger_rest_bad_request(transport: str='rest', request_type=eventarc.UpdateTriggerRequest):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'trigger': {'name': 'projects/sample1/locations/sample2/triggers/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_trigger(request)

def test_update_trigger_rest_flattened():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'trigger': {'name': 'projects/sample1/locations/sample2/triggers/sample3'}}
        mock_args = dict(trigger=gce_trigger.Trigger(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), allow_missing=True)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_trigger(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{trigger.name=projects/*/locations/*/triggers/*}' % client.transport._host, args[1])

def test_update_trigger_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_trigger(eventarc.UpdateTriggerRequest(), trigger=gce_trigger.Trigger(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']), allow_missing=True)

def test_update_trigger_rest_error():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [eventarc.DeleteTriggerRequest, dict])
def test_delete_trigger_rest(request_type):
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/triggers/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_trigger(request)
    assert response.operation.name == 'operations/spam'

def test_delete_trigger_rest_required_fields(request_type=eventarc.DeleteTriggerRequest):
    if False:
        return 10
    transport_class = transports.EventarcRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['validate_only'] = False
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'validateOnly' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_trigger._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'validateOnly' in jsonified_request
    assert jsonified_request['validateOnly'] == request_init['validate_only']
    jsonified_request['name'] = 'name_value'
    jsonified_request['validateOnly'] = True
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_trigger._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('allow_missing', 'etag', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'validateOnly' in jsonified_request
    assert jsonified_request['validateOnly'] == True
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_trigger(request)
            expected_params = [('validateOnly', str(False).lower()), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_trigger_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_trigger._get_unset_required_fields({})
    assert set(unset_fields) == set(('allowMissing', 'etag', 'validateOnly')) & set(('name', 'validateOnly'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_trigger_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EventarcRestInterceptor())
    client = EventarcClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EventarcRestInterceptor, 'post_delete_trigger') as post, mock.patch.object(transports.EventarcRestInterceptor, 'pre_delete_trigger') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = eventarc.DeleteTriggerRequest.pb(eventarc.DeleteTriggerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = eventarc.DeleteTriggerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_trigger(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_trigger_rest_bad_request(transport: str='rest', request_type=eventarc.DeleteTriggerRequest):
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/triggers/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_trigger(request)

def test_delete_trigger_rest_flattened():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/triggers/sample3'}
        mock_args = dict(name='name_value', allow_missing=True)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_trigger(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/triggers/*}' % client.transport._host, args[1])

def test_delete_trigger_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_trigger(eventarc.DeleteTriggerRequest(), name='name_value', allow_missing=True)

def test_delete_trigger_rest_error():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [eventarc.GetChannelRequest, dict])
def test_get_channel_rest(request_type):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/channels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = channel.Channel(name='name_value', uid='uid_value', provider='provider_value', state=channel.Channel.State.PENDING, activation_token='activation_token_value', crypto_key_name='crypto_key_name_value', pubsub_topic='pubsub_topic_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = channel.Channel.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_channel(request)
    assert isinstance(response, channel.Channel)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.provider == 'provider_value'
    assert response.state == channel.Channel.State.PENDING
    assert response.activation_token == 'activation_token_value'
    assert response.crypto_key_name == 'crypto_key_name_value'

def test_get_channel_rest_required_fields(request_type=eventarc.GetChannelRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EventarcRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_channel._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_channel._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = channel.Channel()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = channel.Channel.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_channel(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_channel_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_channel._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_channel_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EventarcRestInterceptor())
    client = EventarcClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EventarcRestInterceptor, 'post_get_channel') as post, mock.patch.object(transports.EventarcRestInterceptor, 'pre_get_channel') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = eventarc.GetChannelRequest.pb(eventarc.GetChannelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = channel.Channel.to_json(channel.Channel())
        request = eventarc.GetChannelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = channel.Channel()
        client.get_channel(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_channel_rest_bad_request(transport: str='rest', request_type=eventarc.GetChannelRequest):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/channels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_channel(request)

def test_get_channel_rest_flattened():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = channel.Channel()
        sample_request = {'name': 'projects/sample1/locations/sample2/channels/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = channel.Channel.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_channel(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/channels/*}' % client.transport._host, args[1])

def test_get_channel_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_channel(eventarc.GetChannelRequest(), name='name_value')

def test_get_channel_rest_error():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [eventarc.ListChannelsRequest, dict])
def test_list_channels_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = eventarc.ListChannelsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = eventarc.ListChannelsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_channels(request)
    assert isinstance(response, pagers.ListChannelsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_channels_rest_required_fields(request_type=eventarc.ListChannelsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EventarcRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_channels._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_channels._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = eventarc.ListChannelsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = eventarc.ListChannelsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_channels(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_channels_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_channels._get_unset_required_fields({})
    assert set(unset_fields) == set(('orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_channels_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EventarcRestInterceptor())
    client = EventarcClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EventarcRestInterceptor, 'post_list_channels') as post, mock.patch.object(transports.EventarcRestInterceptor, 'pre_list_channels') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = eventarc.ListChannelsRequest.pb(eventarc.ListChannelsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = eventarc.ListChannelsResponse.to_json(eventarc.ListChannelsResponse())
        request = eventarc.ListChannelsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = eventarc.ListChannelsResponse()
        client.list_channels(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_channels_rest_bad_request(transport: str='rest', request_type=eventarc.ListChannelsRequest):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_channels(request)

def test_list_channels_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = eventarc.ListChannelsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = eventarc.ListChannelsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_channels(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/channels' % client.transport._host, args[1])

def test_list_channels_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_channels(eventarc.ListChannelsRequest(), parent='parent_value')

def test_list_channels_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (eventarc.ListChannelsResponse(channels=[channel.Channel(), channel.Channel(), channel.Channel()], next_page_token='abc'), eventarc.ListChannelsResponse(channels=[], next_page_token='def'), eventarc.ListChannelsResponse(channels=[channel.Channel()], next_page_token='ghi'), eventarc.ListChannelsResponse(channels=[channel.Channel(), channel.Channel()]))
        response = response + response
        response = tuple((eventarc.ListChannelsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_channels(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, channel.Channel) for i in results))
        pages = list(client.list_channels(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [eventarc.CreateChannelRequest, dict])
def test_create_channel_rest(request_type):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['channel'] = {'name': 'name_value', 'uid': 'uid_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'provider': 'provider_value', 'pubsub_topic': 'pubsub_topic_value', 'state': 1, 'activation_token': 'activation_token_value', 'crypto_key_name': 'crypto_key_name_value'}
    test_field = eventarc.CreateChannelRequest.meta.fields['channel']

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
    for (field, value) in request_init['channel'].items():
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
                for i in range(0, len(request_init['channel'][field])):
                    del request_init['channel'][field][i][subfield]
            else:
                del request_init['channel'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_channel(request)
    assert response.operation.name == 'operations/spam'

def test_create_channel_rest_required_fields(request_type=eventarc.CreateChannelRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EventarcRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['channel_id'] = ''
    request_init['validate_only'] = False
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'channelId' not in jsonified_request
    assert 'validateOnly' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_channel_._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'channelId' in jsonified_request
    assert jsonified_request['channelId'] == request_init['channel_id']
    assert 'validateOnly' in jsonified_request
    assert jsonified_request['validateOnly'] == request_init['validate_only']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['channelId'] = 'channel_id_value'
    jsonified_request['validateOnly'] = True
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_channel_._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('channel_id', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'channelId' in jsonified_request
    assert jsonified_request['channelId'] == 'channel_id_value'
    assert 'validateOnly' in jsonified_request
    assert jsonified_request['validateOnly'] == True
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_channel(request)
            expected_params = [('channelId', ''), ('validateOnly', str(False).lower()), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_channel_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_channel_._get_unset_required_fields({})
    assert set(unset_fields) == set(('channelId', 'validateOnly')) & set(('parent', 'channel', 'channelId', 'validateOnly'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_channel_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EventarcRestInterceptor())
    client = EventarcClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EventarcRestInterceptor, 'post_create_channel') as post, mock.patch.object(transports.EventarcRestInterceptor, 'pre_create_channel') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = eventarc.CreateChannelRequest.pb(eventarc.CreateChannelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = eventarc.CreateChannelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_channel(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_channel_rest_bad_request(transport: str='rest', request_type=eventarc.CreateChannelRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_channel(request)

def test_create_channel_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', channel=gce_channel.Channel(name='name_value'), channel_id='channel_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_channel(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/channels' % client.transport._host, args[1])

def test_create_channel_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_channel(eventarc.CreateChannelRequest(), parent='parent_value', channel=gce_channel.Channel(name='name_value'), channel_id='channel_id_value')

def test_create_channel_rest_error():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [eventarc.UpdateChannelRequest, dict])
def test_update_channel_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'channel': {'name': 'projects/sample1/locations/sample2/channels/sample3'}}
    request_init['channel'] = {'name': 'projects/sample1/locations/sample2/channels/sample3', 'uid': 'uid_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'provider': 'provider_value', 'pubsub_topic': 'pubsub_topic_value', 'state': 1, 'activation_token': 'activation_token_value', 'crypto_key_name': 'crypto_key_name_value'}
    test_field = eventarc.UpdateChannelRequest.meta.fields['channel']

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
    for (field, value) in request_init['channel'].items():
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
                for i in range(0, len(request_init['channel'][field])):
                    del request_init['channel'][field][i][subfield]
            else:
                del request_init['channel'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_channel(request)
    assert response.operation.name == 'operations/spam'

def test_update_channel_rest_required_fields(request_type=eventarc.UpdateChannelRequest):
    if False:
        return 10
    transport_class = transports.EventarcRestTransport
    request_init = {}
    request_init['validate_only'] = False
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'validateOnly' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_channel._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'validateOnly' in jsonified_request
    assert jsonified_request['validateOnly'] == request_init['validate_only']
    jsonified_request['validateOnly'] = True
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_channel._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask', 'validate_only'))
    jsonified_request.update(unset_fields)
    assert 'validateOnly' in jsonified_request
    assert jsonified_request['validateOnly'] == True
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_channel(request)
            expected_params = [('validateOnly', str(False).lower()), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_channel_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_channel._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask', 'validateOnly')) & set(('validateOnly',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_channel_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EventarcRestInterceptor())
    client = EventarcClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EventarcRestInterceptor, 'post_update_channel') as post, mock.patch.object(transports.EventarcRestInterceptor, 'pre_update_channel') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = eventarc.UpdateChannelRequest.pb(eventarc.UpdateChannelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = eventarc.UpdateChannelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.update_channel(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_channel_rest_bad_request(transport: str='rest', request_type=eventarc.UpdateChannelRequest):
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'channel': {'name': 'projects/sample1/locations/sample2/channels/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_channel(request)

def test_update_channel_rest_flattened():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'channel': {'name': 'projects/sample1/locations/sample2/channels/sample3'}}
        mock_args = dict(channel=gce_channel.Channel(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_channel(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{channel.name=projects/*/locations/*/channels/*}' % client.transport._host, args[1])

def test_update_channel_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_channel(eventarc.UpdateChannelRequest(), channel=gce_channel.Channel(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_channel_rest_error():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [eventarc.DeleteChannelRequest, dict])
def test_delete_channel_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/channels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_channel(request)
    assert response.operation.name == 'operations/spam'

def test_delete_channel_rest_required_fields(request_type=eventarc.DeleteChannelRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EventarcRestTransport
    request_init = {}
    request_init['name'] = ''
    request_init['validate_only'] = False
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'validateOnly' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_channel._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'validateOnly' in jsonified_request
    assert jsonified_request['validateOnly'] == request_init['validate_only']
    jsonified_request['name'] = 'name_value'
    jsonified_request['validateOnly'] = True
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_channel._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('validate_only',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    assert 'validateOnly' in jsonified_request
    assert jsonified_request['validateOnly'] == True
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_channel(request)
            expected_params = [('validateOnly', str(False).lower()), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_channel_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_channel._get_unset_required_fields({})
    assert set(unset_fields) == set(('validateOnly',)) & set(('name', 'validateOnly'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_channel_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EventarcRestInterceptor())
    client = EventarcClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EventarcRestInterceptor, 'post_delete_channel') as post, mock.patch.object(transports.EventarcRestInterceptor, 'pre_delete_channel') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = eventarc.DeleteChannelRequest.pb(eventarc.DeleteChannelRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = eventarc.DeleteChannelRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_channel(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_channel_rest_bad_request(transport: str='rest', request_type=eventarc.DeleteChannelRequest):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/channels/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_channel(request)

def test_delete_channel_rest_flattened():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/channels/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_channel(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/channels/*}' % client.transport._host, args[1])

def test_delete_channel_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_channel(eventarc.DeleteChannelRequest(), name='name_value')

def test_delete_channel_rest_error():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [eventarc.GetProviderRequest, dict])
def test_get_provider_rest(request_type):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/providers/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = discovery.Provider(name='name_value', display_name='display_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = discovery.Provider.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_provider(request)
    assert isinstance(response, discovery.Provider)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'

def test_get_provider_rest_required_fields(request_type=eventarc.GetProviderRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EventarcRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_provider._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_provider._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = discovery.Provider()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = discovery.Provider.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_provider(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_provider_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_provider._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_provider_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EventarcRestInterceptor())
    client = EventarcClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EventarcRestInterceptor, 'post_get_provider') as post, mock.patch.object(transports.EventarcRestInterceptor, 'pre_get_provider') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = eventarc.GetProviderRequest.pb(eventarc.GetProviderRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = discovery.Provider.to_json(discovery.Provider())
        request = eventarc.GetProviderRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = discovery.Provider()
        client.get_provider(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_provider_rest_bad_request(transport: str='rest', request_type=eventarc.GetProviderRequest):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/providers/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_provider(request)

def test_get_provider_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = discovery.Provider()
        sample_request = {'name': 'projects/sample1/locations/sample2/providers/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = discovery.Provider.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_provider(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/providers/*}' % client.transport._host, args[1])

def test_get_provider_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_provider(eventarc.GetProviderRequest(), name='name_value')

def test_get_provider_rest_error():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [eventarc.ListProvidersRequest, dict])
def test_list_providers_rest(request_type):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = eventarc.ListProvidersResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = eventarc.ListProvidersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_providers(request)
    assert isinstance(response, pagers.ListProvidersPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_providers_rest_required_fields(request_type=eventarc.ListProvidersRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EventarcRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_providers._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_providers._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'order_by', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = eventarc.ListProvidersResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = eventarc.ListProvidersResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_providers(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_providers_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_providers._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'orderBy', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_providers_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EventarcRestInterceptor())
    client = EventarcClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EventarcRestInterceptor, 'post_list_providers') as post, mock.patch.object(transports.EventarcRestInterceptor, 'pre_list_providers') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = eventarc.ListProvidersRequest.pb(eventarc.ListProvidersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = eventarc.ListProvidersResponse.to_json(eventarc.ListProvidersResponse())
        request = eventarc.ListProvidersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = eventarc.ListProvidersResponse()
        client.list_providers(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_providers_rest_bad_request(transport: str='rest', request_type=eventarc.ListProvidersRequest):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_providers(request)

def test_list_providers_rest_flattened():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = eventarc.ListProvidersResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = eventarc.ListProvidersResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_providers(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/providers' % client.transport._host, args[1])

def test_list_providers_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_providers(eventarc.ListProvidersRequest(), parent='parent_value')

def test_list_providers_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (eventarc.ListProvidersResponse(providers=[discovery.Provider(), discovery.Provider(), discovery.Provider()], next_page_token='abc'), eventarc.ListProvidersResponse(providers=[], next_page_token='def'), eventarc.ListProvidersResponse(providers=[discovery.Provider()], next_page_token='ghi'), eventarc.ListProvidersResponse(providers=[discovery.Provider(), discovery.Provider()]))
        response = response + response
        response = tuple((eventarc.ListProvidersResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_providers(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, discovery.Provider) for i in results))
        pages = list(client.list_providers(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [eventarc.GetChannelConnectionRequest, dict])
def test_get_channel_connection_rest(request_type):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/channelConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = channel_connection.ChannelConnection(name='name_value', uid='uid_value', channel='channel_value', activation_token='activation_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = channel_connection.ChannelConnection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_channel_connection(request)
    assert isinstance(response, channel_connection.ChannelConnection)
    assert response.name == 'name_value'
    assert response.uid == 'uid_value'
    assert response.channel == 'channel_value'
    assert response.activation_token == 'activation_token_value'

def test_get_channel_connection_rest_required_fields(request_type=eventarc.GetChannelConnectionRequest):
    if False:
        print('Hello World!')
    transport_class = transports.EventarcRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_channel_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_channel_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = channel_connection.ChannelConnection()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = channel_connection.ChannelConnection.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_channel_connection(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_channel_connection_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_channel_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_channel_connection_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EventarcRestInterceptor())
    client = EventarcClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EventarcRestInterceptor, 'post_get_channel_connection') as post, mock.patch.object(transports.EventarcRestInterceptor, 'pre_get_channel_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = eventarc.GetChannelConnectionRequest.pb(eventarc.GetChannelConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = channel_connection.ChannelConnection.to_json(channel_connection.ChannelConnection())
        request = eventarc.GetChannelConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = channel_connection.ChannelConnection()
        client.get_channel_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_channel_connection_rest_bad_request(transport: str='rest', request_type=eventarc.GetChannelConnectionRequest):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/channelConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_channel_connection(request)

def test_get_channel_connection_rest_flattened():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = channel_connection.ChannelConnection()
        sample_request = {'name': 'projects/sample1/locations/sample2/channelConnections/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = channel_connection.ChannelConnection.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_channel_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/channelConnections/*}' % client.transport._host, args[1])

def test_get_channel_connection_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_channel_connection(eventarc.GetChannelConnectionRequest(), name='name_value')

def test_get_channel_connection_rest_error():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [eventarc.ListChannelConnectionsRequest, dict])
def test_list_channel_connections_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = eventarc.ListChannelConnectionsResponse(next_page_token='next_page_token_value', unreachable=['unreachable_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = eventarc.ListChannelConnectionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_channel_connections(request)
    assert isinstance(response, pagers.ListChannelConnectionsPager)
    assert response.next_page_token == 'next_page_token_value'
    assert response.unreachable == ['unreachable_value']

def test_list_channel_connections_rest_required_fields(request_type=eventarc.ListChannelConnectionsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EventarcRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_channel_connections._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_channel_connections._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = eventarc.ListChannelConnectionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = eventarc.ListChannelConnectionsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_channel_connections(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_channel_connections_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_channel_connections._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_channel_connections_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EventarcRestInterceptor())
    client = EventarcClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EventarcRestInterceptor, 'post_list_channel_connections') as post, mock.patch.object(transports.EventarcRestInterceptor, 'pre_list_channel_connections') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = eventarc.ListChannelConnectionsRequest.pb(eventarc.ListChannelConnectionsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = eventarc.ListChannelConnectionsResponse.to_json(eventarc.ListChannelConnectionsResponse())
        request = eventarc.ListChannelConnectionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = eventarc.ListChannelConnectionsResponse()
        client.list_channel_connections(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_channel_connections_rest_bad_request(transport: str='rest', request_type=eventarc.ListChannelConnectionsRequest):
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_channel_connections(request)

def test_list_channel_connections_rest_flattened():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = eventarc.ListChannelConnectionsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = eventarc.ListChannelConnectionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_channel_connections(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/channelConnections' % client.transport._host, args[1])

def test_list_channel_connections_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_channel_connections(eventarc.ListChannelConnectionsRequest(), parent='parent_value')

def test_list_channel_connections_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (eventarc.ListChannelConnectionsResponse(channel_connections=[channel_connection.ChannelConnection(), channel_connection.ChannelConnection(), channel_connection.ChannelConnection()], next_page_token='abc'), eventarc.ListChannelConnectionsResponse(channel_connections=[], next_page_token='def'), eventarc.ListChannelConnectionsResponse(channel_connections=[channel_connection.ChannelConnection()], next_page_token='ghi'), eventarc.ListChannelConnectionsResponse(channel_connections=[channel_connection.ChannelConnection(), channel_connection.ChannelConnection()]))
        response = response + response
        response = tuple((eventarc.ListChannelConnectionsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        pager = client.list_channel_connections(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, channel_connection.ChannelConnection) for i in results))
        pages = list(client.list_channel_connections(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [eventarc.CreateChannelConnectionRequest, dict])
def test_create_channel_connection_rest(request_type):
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['channel_connection'] = {'name': 'name_value', 'uid': 'uid_value', 'channel': 'channel_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'update_time': {}, 'activation_token': 'activation_token_value'}
    test_field = eventarc.CreateChannelConnectionRequest.meta.fields['channel_connection']

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
    for (field, value) in request_init['channel_connection'].items():
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
                for i in range(0, len(request_init['channel_connection'][field])):
                    del request_init['channel_connection'][field][i][subfield]
            else:
                del request_init['channel_connection'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_channel_connection(request)
    assert response.operation.name == 'operations/spam'

def test_create_channel_connection_rest_required_fields(request_type=eventarc.CreateChannelConnectionRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.EventarcRestTransport
    request_init = {}
    request_init['parent'] = ''
    request_init['channel_connection_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'channelConnectionId' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_channel_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'channelConnectionId' in jsonified_request
    assert jsonified_request['channelConnectionId'] == request_init['channel_connection_id']
    jsonified_request['parent'] = 'parent_value'
    jsonified_request['channelConnectionId'] = 'channel_connection_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_channel_connection._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('channel_connection_id',))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    assert 'channelConnectionId' in jsonified_request
    assert jsonified_request['channelConnectionId'] == 'channel_connection_id_value'
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_channel_connection(request)
            expected_params = [('channelConnectionId', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_channel_connection_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_channel_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(('channelConnectionId',)) & set(('parent', 'channelConnection', 'channelConnectionId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_channel_connection_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EventarcRestInterceptor())
    client = EventarcClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EventarcRestInterceptor, 'post_create_channel_connection') as post, mock.patch.object(transports.EventarcRestInterceptor, 'pre_create_channel_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = eventarc.CreateChannelConnectionRequest.pb(eventarc.CreateChannelConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = eventarc.CreateChannelConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_channel_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_channel_connection_rest_bad_request(transport: str='rest', request_type=eventarc.CreateChannelConnectionRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_channel_connection(request)

def test_create_channel_connection_rest_flattened():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', channel_connection=gce_channel_connection.ChannelConnection(name='name_value'), channel_connection_id='channel_connection_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_channel_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/channelConnections' % client.transport._host, args[1])

def test_create_channel_connection_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_channel_connection(eventarc.CreateChannelConnectionRequest(), parent='parent_value', channel_connection=gce_channel_connection.ChannelConnection(name='name_value'), channel_connection_id='channel_connection_id_value')

def test_create_channel_connection_rest_error():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [eventarc.DeleteChannelConnectionRequest, dict])
def test_delete_channel_connection_rest(request_type):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/channelConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_channel_connection(request)
    assert response.operation.name == 'operations/spam'

def test_delete_channel_connection_rest_required_fields(request_type=eventarc.DeleteChannelConnectionRequest):
    if False:
        return 10
    transport_class = transports.EventarcRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_channel_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_channel_connection._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_channel_connection(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_channel_connection_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_channel_connection._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_channel_connection_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EventarcRestInterceptor())
    client = EventarcClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.EventarcRestInterceptor, 'post_delete_channel_connection') as post, mock.patch.object(transports.EventarcRestInterceptor, 'pre_delete_channel_connection') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = eventarc.DeleteChannelConnectionRequest.pb(eventarc.DeleteChannelConnectionRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = eventarc.DeleteChannelConnectionRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_channel_connection(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_channel_connection_rest_bad_request(transport: str='rest', request_type=eventarc.DeleteChannelConnectionRequest):
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/channelConnections/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_channel_connection(request)

def test_delete_channel_connection_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'name': 'projects/sample1/locations/sample2/channelConnections/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_channel_connection(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/channelConnections/*}' % client.transport._host, args[1])

def test_delete_channel_connection_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_channel_connection(eventarc.DeleteChannelConnectionRequest(), name='name_value')

def test_delete_channel_connection_rest_error():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [eventarc.GetGoogleChannelConfigRequest, dict])
def test_get_google_channel_config_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/googleChannelConfig'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = google_channel_config.GoogleChannelConfig(name='name_value', crypto_key_name='crypto_key_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = google_channel_config.GoogleChannelConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_google_channel_config(request)
    assert isinstance(response, google_channel_config.GoogleChannelConfig)
    assert response.name == 'name_value'
    assert response.crypto_key_name == 'crypto_key_name_value'

def test_get_google_channel_config_rest_required_fields(request_type=eventarc.GetGoogleChannelConfigRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.EventarcRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_google_channel_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_google_channel_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = google_channel_config.GoogleChannelConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = google_channel_config.GoogleChannelConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_google_channel_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_google_channel_config_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_google_channel_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_google_channel_config_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EventarcRestInterceptor())
    client = EventarcClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EventarcRestInterceptor, 'post_get_google_channel_config') as post, mock.patch.object(transports.EventarcRestInterceptor, 'pre_get_google_channel_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = eventarc.GetGoogleChannelConfigRequest.pb(eventarc.GetGoogleChannelConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = google_channel_config.GoogleChannelConfig.to_json(google_channel_config.GoogleChannelConfig())
        request = eventarc.GetGoogleChannelConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = google_channel_config.GoogleChannelConfig()
        client.get_google_channel_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_google_channel_config_rest_bad_request(transport: str='rest', request_type=eventarc.GetGoogleChannelConfigRequest):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/googleChannelConfig'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_google_channel_config(request)

def test_get_google_channel_config_rest_flattened():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = google_channel_config.GoogleChannelConfig()
        sample_request = {'name': 'projects/sample1/locations/sample2/googleChannelConfig'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = google_channel_config.GoogleChannelConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_google_channel_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/googleChannelConfig}' % client.transport._host, args[1])

def test_get_google_channel_config_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_google_channel_config(eventarc.GetGoogleChannelConfigRequest(), name='name_value')

def test_get_google_channel_config_rest_error():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [eventarc.UpdateGoogleChannelConfigRequest, dict])
def test_update_google_channel_config_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'google_channel_config': {'name': 'projects/sample1/locations/sample2/googleChannelConfig'}}
    request_init['google_channel_config'] = {'name': 'projects/sample1/locations/sample2/googleChannelConfig', 'update_time': {'seconds': 751, 'nanos': 543}, 'crypto_key_name': 'crypto_key_name_value'}
    test_field = eventarc.UpdateGoogleChannelConfigRequest.meta.fields['google_channel_config']

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
    for (field, value) in request_init['google_channel_config'].items():
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
                for i in range(0, len(request_init['google_channel_config'][field])):
                    del request_init['google_channel_config'][field][i][subfield]
            else:
                del request_init['google_channel_config'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gce_google_channel_config.GoogleChannelConfig(name='name_value', crypto_key_name='crypto_key_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gce_google_channel_config.GoogleChannelConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_google_channel_config(request)
    assert isinstance(response, gce_google_channel_config.GoogleChannelConfig)
    assert response.name == 'name_value'
    assert response.crypto_key_name == 'crypto_key_name_value'

def test_update_google_channel_config_rest_required_fields(request_type=eventarc.UpdateGoogleChannelConfigRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.EventarcRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_google_channel_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_google_channel_config._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gce_google_channel_config.GoogleChannelConfig()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gce_google_channel_config.GoogleChannelConfig.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_google_channel_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_google_channel_config_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_google_channel_config._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('googleChannelConfig',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_google_channel_config_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.EventarcRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.EventarcRestInterceptor())
    client = EventarcClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.EventarcRestInterceptor, 'post_update_google_channel_config') as post, mock.patch.object(transports.EventarcRestInterceptor, 'pre_update_google_channel_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = eventarc.UpdateGoogleChannelConfigRequest.pb(eventarc.UpdateGoogleChannelConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gce_google_channel_config.GoogleChannelConfig.to_json(gce_google_channel_config.GoogleChannelConfig())
        request = eventarc.UpdateGoogleChannelConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gce_google_channel_config.GoogleChannelConfig()
        client.update_google_channel_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_google_channel_config_rest_bad_request(transport: str='rest', request_type=eventarc.UpdateGoogleChannelConfigRequest):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'google_channel_config': {'name': 'projects/sample1/locations/sample2/googleChannelConfig'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_google_channel_config(request)

def test_update_google_channel_config_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gce_google_channel_config.GoogleChannelConfig()
        sample_request = {'google_channel_config': {'name': 'projects/sample1/locations/sample2/googleChannelConfig'}}
        mock_args = dict(google_channel_config=gce_google_channel_config.GoogleChannelConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gce_google_channel_config.GoogleChannelConfig.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_google_channel_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{google_channel_config.name=projects/*/locations/*/googleChannelConfig}' % client.transport._host, args[1])

def test_update_google_channel_config_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_google_channel_config(eventarc.UpdateGoogleChannelConfigRequest(), google_channel_config=gce_google_channel_config.GoogleChannelConfig(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_google_channel_config_rest_error():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.EventarcGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.EventarcGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EventarcClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.EventarcGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = EventarcClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = EventarcClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.EventarcGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = EventarcClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.EventarcGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = EventarcClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.EventarcGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.EventarcGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.EventarcGrpcTransport, transports.EventarcGrpcAsyncIOTransport, transports.EventarcRestTransport])
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
    transport = EventarcClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.EventarcGrpcTransport)

def test_eventarc_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.EventarcTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_eventarc_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.eventarc_v1.services.eventarc.transports.EventarcTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.EventarcTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('get_trigger', 'list_triggers', 'create_trigger', 'update_trigger', 'delete_trigger', 'get_channel', 'list_channels', 'create_channel_', 'update_channel', 'delete_channel', 'get_provider', 'list_providers', 'get_channel_connection', 'list_channel_connections', 'create_channel_connection', 'delete_channel_connection', 'get_google_channel_config', 'update_google_channel_config', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'delete_operation', 'list_operations')
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

def test_eventarc_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.eventarc_v1.services.eventarc.transports.EventarcTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.EventarcTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_eventarc_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.eventarc_v1.services.eventarc.transports.EventarcTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.EventarcTransport()
        adc.assert_called_once()

def test_eventarc_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        EventarcClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.EventarcGrpcTransport, transports.EventarcGrpcAsyncIOTransport])
def test_eventarc_transport_auth_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.EventarcGrpcTransport, transports.EventarcGrpcAsyncIOTransport, transports.EventarcRestTransport])
def test_eventarc_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.EventarcGrpcTransport, grpc_helpers), (transports.EventarcGrpcAsyncIOTransport, grpc_helpers_async)])
def test_eventarc_transport_create_channel(transport_class, grpc_helpers):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('eventarc.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='eventarc.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.EventarcGrpcTransport, transports.EventarcGrpcAsyncIOTransport])
def test_eventarc_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_eventarc_http_transport_client_cert_source_for_mtls():
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.EventarcRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_eventarc_rest_lro_client():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_eventarc_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='eventarc.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('eventarc.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://eventarc.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_eventarc_host_with_port(transport_name):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='eventarc.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('eventarc.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://eventarc.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_eventarc_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = EventarcClient(credentials=creds1, transport=transport_name)
    client2 = EventarcClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.get_trigger._session
    session2 = client2.transport.get_trigger._session
    assert session1 != session2
    session1 = client1.transport.list_triggers._session
    session2 = client2.transport.list_triggers._session
    assert session1 != session2
    session1 = client1.transport.create_trigger._session
    session2 = client2.transport.create_trigger._session
    assert session1 != session2
    session1 = client1.transport.update_trigger._session
    session2 = client2.transport.update_trigger._session
    assert session1 != session2
    session1 = client1.transport.delete_trigger._session
    session2 = client2.transport.delete_trigger._session
    assert session1 != session2
    session1 = client1.transport.get_channel._session
    session2 = client2.transport.get_channel._session
    assert session1 != session2
    session1 = client1.transport.list_channels._session
    session2 = client2.transport.list_channels._session
    assert session1 != session2
    session1 = client1.transport.create_channel_._session
    session2 = client2.transport.create_channel_._session
    assert session1 != session2
    session1 = client1.transport.update_channel._session
    session2 = client2.transport.update_channel._session
    assert session1 != session2
    session1 = client1.transport.delete_channel._session
    session2 = client2.transport.delete_channel._session
    assert session1 != session2
    session1 = client1.transport.get_provider._session
    session2 = client2.transport.get_provider._session
    assert session1 != session2
    session1 = client1.transport.list_providers._session
    session2 = client2.transport.list_providers._session
    assert session1 != session2
    session1 = client1.transport.get_channel_connection._session
    session2 = client2.transport.get_channel_connection._session
    assert session1 != session2
    session1 = client1.transport.list_channel_connections._session
    session2 = client2.transport.list_channel_connections._session
    assert session1 != session2
    session1 = client1.transport.create_channel_connection._session
    session2 = client2.transport.create_channel_connection._session
    assert session1 != session2
    session1 = client1.transport.delete_channel_connection._session
    session2 = client2.transport.delete_channel_connection._session
    assert session1 != session2
    session1 = client1.transport.get_google_channel_config._session
    session2 = client2.transport.get_google_channel_config._session
    assert session1 != session2
    session1 = client1.transport.update_google_channel_config._session
    session2 = client2.transport.update_google_channel_config._session
    assert session1 != session2

def test_eventarc_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.EventarcGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_eventarc_grpc_asyncio_transport_channel():
    if False:
        print('Hello World!')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.EventarcGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.EventarcGrpcTransport, transports.EventarcGrpcAsyncIOTransport])
def test_eventarc_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.EventarcGrpcTransport, transports.EventarcGrpcAsyncIOTransport])
def test_eventarc_transport_channel_mtls_with_adc(transport_class):
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

def test_eventarc_grpc_lro_client():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_eventarc_grpc_lro_async_client():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_channel_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    channel = 'whelk'
    expected = 'projects/{project}/locations/{location}/channels/{channel}'.format(project=project, location=location, channel=channel)
    actual = EventarcClient.channel_path(project, location, channel)
    assert expected == actual

def test_parse_channel_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus', 'location': 'oyster', 'channel': 'nudibranch'}
    path = EventarcClient.channel_path(**expected)
    actual = EventarcClient.parse_channel_path(path)
    assert expected == actual

def test_channel_connection_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    location = 'mussel'
    channel_connection = 'winkle'
    expected = 'projects/{project}/locations/{location}/channelConnections/{channel_connection}'.format(project=project, location=location, channel_connection=channel_connection)
    actual = EventarcClient.channel_connection_path(project, location, channel_connection)
    assert expected == actual

def test_parse_channel_connection_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'nautilus', 'location': 'scallop', 'channel_connection': 'abalone'}
    path = EventarcClient.channel_connection_path(**expected)
    actual = EventarcClient.parse_channel_connection_path(path)
    assert expected == actual

def test_cloud_function_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    location = 'clam'
    function = 'whelk'
    expected = 'projects/{project}/locations/{location}/functions/{function}'.format(project=project, location=location, function=function)
    actual = EventarcClient.cloud_function_path(project, location, function)
    assert expected == actual

def test_parse_cloud_function_path():
    if False:
        return 10
    expected = {'project': 'octopus', 'location': 'oyster', 'function': 'nudibranch'}
    path = EventarcClient.cloud_function_path(**expected)
    actual = EventarcClient.parse_cloud_function_path(path)
    assert expected == actual

def test_crypto_key_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    location = 'mussel'
    key_ring = 'winkle'
    crypto_key = 'nautilus'
    expected = 'projects/{project}/locations/{location}/keyRings/{key_ring}/cryptoKeys/{crypto_key}'.format(project=project, location=location, key_ring=key_ring, crypto_key=crypto_key)
    actual = EventarcClient.crypto_key_path(project, location, key_ring, crypto_key)
    assert expected == actual

def test_parse_crypto_key_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'scallop', 'location': 'abalone', 'key_ring': 'squid', 'crypto_key': 'clam'}
    path = EventarcClient.crypto_key_path(**expected)
    actual = EventarcClient.parse_crypto_key_path(path)
    assert expected == actual

def test_google_channel_config_path():
    if False:
        while True:
            i = 10
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}/googleChannelConfig'.format(project=project, location=location)
    actual = EventarcClient.google_channel_config_path(project, location)
    assert expected == actual

def test_parse_google_channel_config_path():
    if False:
        print('Hello World!')
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = EventarcClient.google_channel_config_path(**expected)
    actual = EventarcClient.parse_google_channel_config_path(path)
    assert expected == actual

def test_provider_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    location = 'mussel'
    provider = 'winkle'
    expected = 'projects/{project}/locations/{location}/providers/{provider}'.format(project=project, location=location, provider=provider)
    actual = EventarcClient.provider_path(project, location, provider)
    assert expected == actual

def test_parse_provider_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'nautilus', 'location': 'scallop', 'provider': 'abalone'}
    path = EventarcClient.provider_path(**expected)
    actual = EventarcClient.parse_provider_path(path)
    assert expected == actual

def test_service_path():
    if False:
        while True:
            i = 10
    expected = '*'.format()
    actual = EventarcClient.service_path()
    assert expected == actual

def test_parse_service_path():
    if False:
        i = 10
        return i + 15
    expected = {}
    path = EventarcClient.service_path(**expected)
    actual = EventarcClient.parse_service_path(path)
    assert expected == actual

def test_service_account_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    service_account = 'clam'
    expected = 'projects/{project}/serviceAccounts/{service_account}'.format(project=project, service_account=service_account)
    actual = EventarcClient.service_account_path(project, service_account)
    assert expected == actual

def test_parse_service_account_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'whelk', 'service_account': 'octopus'}
    path = EventarcClient.service_account_path(**expected)
    actual = EventarcClient.parse_service_account_path(path)
    assert expected == actual

def test_trigger_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'oyster'
    location = 'nudibranch'
    trigger = 'cuttlefish'
    expected = 'projects/{project}/locations/{location}/triggers/{trigger}'.format(project=project, location=location, trigger=trigger)
    actual = EventarcClient.trigger_path(project, location, trigger)
    assert expected == actual

def test_parse_trigger_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'mussel', 'location': 'winkle', 'trigger': 'nautilus'}
    path = EventarcClient.trigger_path(**expected)
    actual = EventarcClient.parse_trigger_path(path)
    assert expected == actual

def test_workflow_path():
    if False:
        i = 10
        return i + 15
    project = 'scallop'
    location = 'abalone'
    workflow = 'squid'
    expected = 'projects/{project}/locations/{location}/workflows/{workflow}'.format(project=project, location=location, workflow=workflow)
    actual = EventarcClient.workflow_path(project, location, workflow)
    assert expected == actual

def test_parse_workflow_path():
    if False:
        print('Hello World!')
    expected = {'project': 'clam', 'location': 'whelk', 'workflow': 'octopus'}
    path = EventarcClient.workflow_path(**expected)
    actual = EventarcClient.parse_workflow_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    billing_account = 'oyster'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = EventarcClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'nudibranch'}
    path = EventarcClient.common_billing_account_path(**expected)
    actual = EventarcClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'cuttlefish'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = EventarcClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'mussel'}
    path = EventarcClient.common_folder_path(**expected)
    actual = EventarcClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'winkle'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = EventarcClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        i = 10
        return i + 15
    expected = {'organization': 'nautilus'}
    path = EventarcClient.common_organization_path(**expected)
    actual = EventarcClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'scallop'
    expected = 'projects/{project}'.format(project=project)
    actual = EventarcClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'abalone'}
    path = EventarcClient.common_project_path(**expected)
    actual = EventarcClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'squid'
    location = 'clam'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = EventarcClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'whelk', 'location': 'octopus'}
    path = EventarcClient.common_location_path(**expected)
    actual = EventarcClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.EventarcTransport, '_prep_wrapped_messages') as prep:
        client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.EventarcTransport, '_prep_wrapped_messages') as prep:
        transport_class = EventarcClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/triggers/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/triggers/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)

def test_set_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/triggers/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/triggers/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)

def test_test_iam_permissions_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'projects/sample1/locations/sample2/triggers/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'projects/sample1/locations/sample2/triggers/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.test_iam_permissions(request)
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)

def test_cancel_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.CancelOperationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2/operations/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.cancel_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.CancelOperationRequest, dict])
def test_cancel_operation_rest(request_type):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/operations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = '{}'
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.cancel_operation(request)
    assert response is None

def test_delete_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.DeleteOperationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/locations/sample2/operations/sample3'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.DeleteOperationRequest, dict])
def test_delete_operation_rest(request_type):
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/operations/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = '{}'
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_operation(request)
    assert response is None

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_delete_operation(transport: str='grpc'):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = None
        response = client.delete_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_delete_operation_from_dict_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_operation(request={'name': 'locations'})
        call.assert_called()

def test_cancel_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.SetIamPolicyRequest()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_set_iam_policy_async(transport: str='grpc_asyncio'):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.SetIamPolicyRequest()
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_field_headers():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

@pytest.mark.asyncio
async def test_set_iam_policy_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.SetIamPolicyRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.set_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

def test_set_iam_policy_from_dict():
    if False:
        return 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.GetIamPolicyRequest()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response = client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

@pytest.mark.asyncio
async def test_get_iam_policy_async(transport: str='grpc_asyncio'):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.GetIamPolicyRequest()
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy(version=774, etag=b'etag_blob'))
        response = await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_field_headers():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.get_iam_policy(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_iam_policy_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.GetIamPolicyRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        await client.get_iam_policy(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

def test_get_iam_policy_from_dict():
    if False:
        for i in range(10):
            print('nop')
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.TestIamPermissionsRequest()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response = client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

@pytest.mark.asyncio
async def test_test_iam_permissions_async(transport: str='grpc_asyncio'):
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = iam_policy_pb2.TestIamPermissionsRequest()
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value']))
        response = await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_field_headers():
    if False:
        while True:
            i = 10
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

@pytest.mark.asyncio
async def test_test_iam_permissions_field_headers_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = iam_policy_pb2.TestIamPermissionsRequest()
    request.resource = 'resource/value'
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        await client.test_iam_permissions(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'resource=resource/value') in kw['metadata']

def test_test_iam_permissions_from_dict():
    if False:
        i = 10
        return i + 15
    client = EventarcClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = EventarcAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = EventarcClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(EventarcClient, transports.EventarcGrpcTransport), (EventarcAsyncClient, transports.EventarcGrpcAsyncIOTransport)])
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