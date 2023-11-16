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
from google.protobuf import struct_pb2
from google.protobuf import timestamp_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.dialogflow_v2.services.answer_records import AnswerRecordsAsyncClient, AnswerRecordsClient, pagers, transports
from google.cloud.dialogflow_v2.types import context, intent, participant, session
from google.cloud.dialogflow_v2.types import answer_record
from google.cloud.dialogflow_v2.types import answer_record as gcd_answer_record

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
    assert AnswerRecordsClient._get_default_mtls_endpoint(None) is None
    assert AnswerRecordsClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert AnswerRecordsClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert AnswerRecordsClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert AnswerRecordsClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert AnswerRecordsClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(AnswerRecordsClient, 'grpc'), (AnswerRecordsAsyncClient, 'grpc_asyncio'), (AnswerRecordsClient, 'rest')])
def test_answer_records_client_from_service_account_info(client_class, transport_name):
    if False:
        print('Hello World!')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.AnswerRecordsGrpcTransport, 'grpc'), (transports.AnswerRecordsGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.AnswerRecordsRestTransport, 'rest')])
def test_answer_records_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(AnswerRecordsClient, 'grpc'), (AnswerRecordsAsyncClient, 'grpc_asyncio'), (AnswerRecordsClient, 'rest')])
def test_answer_records_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

def test_answer_records_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = AnswerRecordsClient.get_transport_class()
    available_transports = [transports.AnswerRecordsGrpcTransport, transports.AnswerRecordsRestTransport]
    assert transport in available_transports
    transport = AnswerRecordsClient.get_transport_class('grpc')
    assert transport == transports.AnswerRecordsGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AnswerRecordsClient, transports.AnswerRecordsGrpcTransport, 'grpc'), (AnswerRecordsAsyncClient, transports.AnswerRecordsGrpcAsyncIOTransport, 'grpc_asyncio'), (AnswerRecordsClient, transports.AnswerRecordsRestTransport, 'rest')])
@mock.patch.object(AnswerRecordsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AnswerRecordsClient))
@mock.patch.object(AnswerRecordsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AnswerRecordsAsyncClient))
def test_answer_records_client_client_options(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(AnswerRecordsClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(AnswerRecordsClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(AnswerRecordsClient, transports.AnswerRecordsGrpcTransport, 'grpc', 'true'), (AnswerRecordsAsyncClient, transports.AnswerRecordsGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (AnswerRecordsClient, transports.AnswerRecordsGrpcTransport, 'grpc', 'false'), (AnswerRecordsAsyncClient, transports.AnswerRecordsGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (AnswerRecordsClient, transports.AnswerRecordsRestTransport, 'rest', 'true'), (AnswerRecordsClient, transports.AnswerRecordsRestTransport, 'rest', 'false')])
@mock.patch.object(AnswerRecordsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AnswerRecordsClient))
@mock.patch.object(AnswerRecordsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AnswerRecordsAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_answer_records_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [AnswerRecordsClient, AnswerRecordsAsyncClient])
@mock.patch.object(AnswerRecordsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AnswerRecordsClient))
@mock.patch.object(AnswerRecordsAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(AnswerRecordsAsyncClient))
def test_answer_records_client_get_mtls_endpoint_and_cert_source(client_class):
    if False:
        for i in range(10):
            print('nop')
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(AnswerRecordsClient, transports.AnswerRecordsGrpcTransport, 'grpc'), (AnswerRecordsAsyncClient, transports.AnswerRecordsGrpcAsyncIOTransport, 'grpc_asyncio'), (AnswerRecordsClient, transports.AnswerRecordsRestTransport, 'rest')])
def test_answer_records_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AnswerRecordsClient, transports.AnswerRecordsGrpcTransport, 'grpc', grpc_helpers), (AnswerRecordsAsyncClient, transports.AnswerRecordsGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (AnswerRecordsClient, transports.AnswerRecordsRestTransport, 'rest', None)])
def test_answer_records_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_answer_records_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.dialogflow_v2.services.answer_records.transports.AnswerRecordsGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = AnswerRecordsClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(AnswerRecordsClient, transports.AnswerRecordsGrpcTransport, 'grpc', grpc_helpers), (AnswerRecordsAsyncClient, transports.AnswerRecordsGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_answer_records_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('dialogflow.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), scopes=None, default_host='dialogflow.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [answer_record.ListAnswerRecordsRequest, dict])
def test_list_answer_records(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_answer_records), '__call__') as call:
        call.return_value = answer_record.ListAnswerRecordsResponse(next_page_token='next_page_token_value')
        response = client.list_answer_records(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == answer_record.ListAnswerRecordsRequest()
    assert isinstance(response, pagers.ListAnswerRecordsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_answer_records_empty_call():
    if False:
        i = 10
        return i + 15
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_answer_records), '__call__') as call:
        client.list_answer_records()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == answer_record.ListAnswerRecordsRequest()

@pytest.mark.asyncio
async def test_list_answer_records_async(transport: str='grpc_asyncio', request_type=answer_record.ListAnswerRecordsRequest):
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_answer_records), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(answer_record.ListAnswerRecordsResponse(next_page_token='next_page_token_value'))
        response = await client.list_answer_records(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == answer_record.ListAnswerRecordsRequest()
    assert isinstance(response, pagers.ListAnswerRecordsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_answer_records_async_from_dict():
    await test_list_answer_records_async(request_type=dict)

def test_list_answer_records_field_headers():
    if False:
        print('Hello World!')
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials())
    request = answer_record.ListAnswerRecordsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_answer_records), '__call__') as call:
        call.return_value = answer_record.ListAnswerRecordsResponse()
        client.list_answer_records(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_answer_records_field_headers_async():
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = answer_record.ListAnswerRecordsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_answer_records), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(answer_record.ListAnswerRecordsResponse())
        await client.list_answer_records(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_answer_records_flattened():
    if False:
        i = 10
        return i + 15
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_answer_records), '__call__') as call:
        call.return_value = answer_record.ListAnswerRecordsResponse()
        client.list_answer_records(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_answer_records_flattened_error():
    if False:
        return 10
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_answer_records(answer_record.ListAnswerRecordsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_answer_records_flattened_async():
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_answer_records), '__call__') as call:
        call.return_value = answer_record.ListAnswerRecordsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(answer_record.ListAnswerRecordsResponse())
        response = await client.list_answer_records(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_answer_records_flattened_error_async():
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_answer_records(answer_record.ListAnswerRecordsRequest(), parent='parent_value')

def test_list_answer_records_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_answer_records), '__call__') as call:
        call.side_effect = (answer_record.ListAnswerRecordsResponse(answer_records=[answer_record.AnswerRecord(), answer_record.AnswerRecord(), answer_record.AnswerRecord()], next_page_token='abc'), answer_record.ListAnswerRecordsResponse(answer_records=[], next_page_token='def'), answer_record.ListAnswerRecordsResponse(answer_records=[answer_record.AnswerRecord()], next_page_token='ghi'), answer_record.ListAnswerRecordsResponse(answer_records=[answer_record.AnswerRecord(), answer_record.AnswerRecord()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_answer_records(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, answer_record.AnswerRecord) for i in results))

def test_list_answer_records_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_answer_records), '__call__') as call:
        call.side_effect = (answer_record.ListAnswerRecordsResponse(answer_records=[answer_record.AnswerRecord(), answer_record.AnswerRecord(), answer_record.AnswerRecord()], next_page_token='abc'), answer_record.ListAnswerRecordsResponse(answer_records=[], next_page_token='def'), answer_record.ListAnswerRecordsResponse(answer_records=[answer_record.AnswerRecord()], next_page_token='ghi'), answer_record.ListAnswerRecordsResponse(answer_records=[answer_record.AnswerRecord(), answer_record.AnswerRecord()]), RuntimeError)
        pages = list(client.list_answer_records(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_answer_records_async_pager():
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_answer_records), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (answer_record.ListAnswerRecordsResponse(answer_records=[answer_record.AnswerRecord(), answer_record.AnswerRecord(), answer_record.AnswerRecord()], next_page_token='abc'), answer_record.ListAnswerRecordsResponse(answer_records=[], next_page_token='def'), answer_record.ListAnswerRecordsResponse(answer_records=[answer_record.AnswerRecord()], next_page_token='ghi'), answer_record.ListAnswerRecordsResponse(answer_records=[answer_record.AnswerRecord(), answer_record.AnswerRecord()]), RuntimeError)
        async_pager = await client.list_answer_records(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, answer_record.AnswerRecord) for i in responses))

@pytest.mark.asyncio
async def test_list_answer_records_async_pages():
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_answer_records), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (answer_record.ListAnswerRecordsResponse(answer_records=[answer_record.AnswerRecord(), answer_record.AnswerRecord(), answer_record.AnswerRecord()], next_page_token='abc'), answer_record.ListAnswerRecordsResponse(answer_records=[], next_page_token='def'), answer_record.ListAnswerRecordsResponse(answer_records=[answer_record.AnswerRecord()], next_page_token='ghi'), answer_record.ListAnswerRecordsResponse(answer_records=[answer_record.AnswerRecord(), answer_record.AnswerRecord()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_answer_records(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [gcd_answer_record.UpdateAnswerRecordRequest, dict])
def test_update_answer_record(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_answer_record), '__call__') as call:
        call.return_value = gcd_answer_record.AnswerRecord(name='name_value')
        response = client.update_answer_record(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_answer_record.UpdateAnswerRecordRequest()
    assert isinstance(response, gcd_answer_record.AnswerRecord)
    assert response.name == 'name_value'

def test_update_answer_record_empty_call():
    if False:
        i = 10
        return i + 15
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_answer_record), '__call__') as call:
        client.update_answer_record()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_answer_record.UpdateAnswerRecordRequest()

@pytest.mark.asyncio
async def test_update_answer_record_async(transport: str='grpc_asyncio', request_type=gcd_answer_record.UpdateAnswerRecordRequest):
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_answer_record), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_answer_record.AnswerRecord(name='name_value'))
        response = await client.update_answer_record(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == gcd_answer_record.UpdateAnswerRecordRequest()
    assert isinstance(response, gcd_answer_record.AnswerRecord)
    assert response.name == 'name_value'

@pytest.mark.asyncio
async def test_update_answer_record_async_from_dict():
    await test_update_answer_record_async(request_type=dict)

def test_update_answer_record_field_headers():
    if False:
        print('Hello World!')
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_answer_record.UpdateAnswerRecordRequest()
    request.answer_record.name = 'name_value'
    with mock.patch.object(type(client.transport.update_answer_record), '__call__') as call:
        call.return_value = gcd_answer_record.AnswerRecord()
        client.update_answer_record(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'answer_record.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_answer_record_field_headers_async():
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = gcd_answer_record.UpdateAnswerRecordRequest()
    request.answer_record.name = 'name_value'
    with mock.patch.object(type(client.transport.update_answer_record), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_answer_record.AnswerRecord())
        await client.update_answer_record(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'answer_record.name=name_value') in kw['metadata']

def test_update_answer_record_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_answer_record), '__call__') as call:
        call.return_value = gcd_answer_record.AnswerRecord()
        client.update_answer_record(answer_record=gcd_answer_record.AnswerRecord(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].answer_record
        mock_val = gcd_answer_record.AnswerRecord(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

def test_update_answer_record_flattened_error():
    if False:
        return 10
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_answer_record(gcd_answer_record.UpdateAnswerRecordRequest(), answer_record=gcd_answer_record.AnswerRecord(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.asyncio
async def test_update_answer_record_flattened_async():
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_answer_record), '__call__') as call:
        call.return_value = gcd_answer_record.AnswerRecord()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gcd_answer_record.AnswerRecord())
        response = await client.update_answer_record(answer_record=gcd_answer_record.AnswerRecord(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].answer_record
        mock_val = gcd_answer_record.AnswerRecord(name='name_value')
        assert arg == mock_val
        arg = args[0].update_mask
        mock_val = field_mask_pb2.FieldMask(paths=['paths_value'])
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_answer_record_flattened_error_async():
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_answer_record(gcd_answer_record.UpdateAnswerRecordRequest(), answer_record=gcd_answer_record.AnswerRecord(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

@pytest.mark.parametrize('request_type', [answer_record.ListAnswerRecordsRequest, dict])
def test_list_answer_records_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = answer_record.ListAnswerRecordsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = answer_record.ListAnswerRecordsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_answer_records(request)
    assert isinstance(response, pagers.ListAnswerRecordsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_answer_records_rest_required_fields(request_type=answer_record.ListAnswerRecordsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.AnswerRecordsRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_answer_records._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_answer_records._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = answer_record.ListAnswerRecordsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = answer_record.ListAnswerRecordsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_answer_records(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_answer_records_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.AnswerRecordsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_answer_records._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_answer_records_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AnswerRecordsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AnswerRecordsRestInterceptor())
    client = AnswerRecordsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AnswerRecordsRestInterceptor, 'post_list_answer_records') as post, mock.patch.object(transports.AnswerRecordsRestInterceptor, 'pre_list_answer_records') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = answer_record.ListAnswerRecordsRequest.pb(answer_record.ListAnswerRecordsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = answer_record.ListAnswerRecordsResponse.to_json(answer_record.ListAnswerRecordsResponse())
        request = answer_record.ListAnswerRecordsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = answer_record.ListAnswerRecordsResponse()
        client.list_answer_records(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_answer_records_rest_bad_request(transport: str='rest', request_type=answer_record.ListAnswerRecordsRequest):
    if False:
        print('Hello World!')
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_answer_records(request)

def test_list_answer_records_rest_flattened():
    if False:
        return 10
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = answer_record.ListAnswerRecordsResponse()
        sample_request = {'parent': 'projects/sample1'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = answer_record.ListAnswerRecordsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_answer_records(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{parent=projects/*}/answerRecords' % client.transport._host, args[1])

def test_list_answer_records_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_answer_records(answer_record.ListAnswerRecordsRequest(), parent='parent_value')

def test_list_answer_records_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (answer_record.ListAnswerRecordsResponse(answer_records=[answer_record.AnswerRecord(), answer_record.AnswerRecord(), answer_record.AnswerRecord()], next_page_token='abc'), answer_record.ListAnswerRecordsResponse(answer_records=[], next_page_token='def'), answer_record.ListAnswerRecordsResponse(answer_records=[answer_record.AnswerRecord()], next_page_token='ghi'), answer_record.ListAnswerRecordsResponse(answer_records=[answer_record.AnswerRecord(), answer_record.AnswerRecord()]))
        response = response + response
        response = tuple((answer_record.ListAnswerRecordsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1'}
        pager = client.list_answer_records(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, answer_record.AnswerRecord) for i in results))
        pages = list(client.list_answer_records(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [gcd_answer_record.UpdateAnswerRecordRequest, dict])
def test_update_answer_record_rest(request_type):
    if False:
        return 10
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'answer_record': {'name': 'projects/sample1/answerRecords/sample2'}}
    request_init['answer_record'] = {'name': 'projects/sample1/answerRecords/sample2', 'answer_feedback': {'correctness_level': 1, 'agent_assistant_detail_feedback': {'answer_relevance': 1, 'document_correctness': 1, 'document_efficiency': 1, 'summarization_feedback': {'start_time': {'seconds': 751, 'nanos': 543}, 'submit_time': {}, 'summary_text': 'summary_text_value'}, 'knowledge_search_feedback': {'answer_copied': True, 'clicked_uris': ['clicked_uris_value1', 'clicked_uris_value2']}}, 'clicked': True, 'click_time': {}, 'displayed': True, 'display_time': {}}, 'agent_assistant_record': {'article_suggestion_answer': {'title': 'title_value', 'uri': 'uri_value', 'snippets': ['snippets_value1', 'snippets_value2'], 'confidence': 0.1038, 'metadata': {}, 'answer_record': 'answer_record_value'}, 'faq_answer': {'answer': 'answer_value', 'confidence': 0.1038, 'question': 'question_value', 'source': 'source_value', 'metadata': {}, 'answer_record': 'answer_record_value'}, 'dialogflow_assist_answer': {'query_result': {'query_text': 'query_text_value', 'language_code': 'language_code_value', 'speech_recognition_confidence': 0.3045, 'action': 'action_value', 'parameters': {'fields': {}}, 'all_required_params_present': True, 'cancels_slot_filling': True, 'fulfillment_text': 'fulfillment_text_value', 'fulfillment_messages': [{'text': {'text': ['text_value1', 'text_value2']}, 'image': {'image_uri': 'image_uri_value', 'accessibility_text': 'accessibility_text_value'}, 'quick_replies': {'title': 'title_value', 'quick_replies': ['quick_replies_value1', 'quick_replies_value2']}, 'card': {'title': 'title_value', 'subtitle': 'subtitle_value', 'image_uri': 'image_uri_value', 'buttons': [{'text': 'text_value', 'postback': 'postback_value'}]}, 'payload': {}, 'simple_responses': {'simple_responses': [{'text_to_speech': 'text_to_speech_value', 'ssml': 'ssml_value', 'display_text': 'display_text_value'}]}, 'basic_card': {'title': 'title_value', 'subtitle': 'subtitle_value', 'formatted_text': 'formatted_text_value', 'image': {}, 'buttons': [{'title': 'title_value', 'open_uri_action': {'uri': 'uri_value'}}]}, 'suggestions': {'suggestions': [{'title': 'title_value'}]}, 'link_out_suggestion': {'destination_name': 'destination_name_value', 'uri': 'uri_value'}, 'list_select': {'title': 'title_value', 'items': [{'info': {'key': 'key_value', 'synonyms': ['synonyms_value1', 'synonyms_value2']}, 'title': 'title_value', 'description': 'description_value', 'image': {}}], 'subtitle': 'subtitle_value'}, 'carousel_select': {'items': [{'info': {}, 'title': 'title_value', 'description': 'description_value', 'image': {}}]}, 'browse_carousel_card': {'items': [{'open_uri_action': {'url': 'url_value', 'url_type_hint': 1}, 'title': 'title_value', 'description': 'description_value', 'image': {}, 'footer': 'footer_value'}], 'image_display_options': 1}, 'table_card': {'title': 'title_value', 'subtitle': 'subtitle_value', 'image': {}, 'column_properties': [{'header': 'header_value', 'horizontal_alignment': 1}], 'rows': [{'cells': [{'text': 'text_value'}], 'divider_after': True}], 'buttons': {}}, 'media_content': {'media_type': 1, 'media_objects': [{'name': 'name_value', 'description': 'description_value', 'large_image': {}, 'icon': {}, 'content_url': 'content_url_value'}]}, 'platform': 1}], 'webhook_source': 'webhook_source_value', 'webhook_payload': {}, 'output_contexts': [{'name': 'name_value', 'lifespan_count': 1498, 'parameters': {}}], 'intent': {'name': 'name_value', 'display_name': 'display_name_value', 'webhook_state': 1, 'priority': 898, 'is_fallback': True, 'ml_disabled': True, 'live_agent_handoff': True, 'end_interaction': True, 'input_context_names': ['input_context_names_value1', 'input_context_names_value2'], 'events': ['events_value1', 'events_value2'], 'training_phrases': [{'name': 'name_value', 'type_': 1, 'parts': [{'text': 'text_value', 'entity_type': 'entity_type_value', 'alias': 'alias_value', 'user_defined': True}], 'times_added_count': 1787}], 'action': 'action_value', 'output_contexts': {}, 'reset_contexts': True, 'parameters': [{'name': 'name_value', 'display_name': 'display_name_value', 'value': 'value_value', 'default_value': 'default_value_value', 'entity_type_display_name': 'entity_type_display_name_value', 'mandatory': True, 'prompts': ['prompts_value1', 'prompts_value2'], 'is_list': True}], 'messages': {}, 'default_response_platforms': [1], 'root_followup_intent_name': 'root_followup_intent_name_value', 'parent_followup_intent_name': 'parent_followup_intent_name_value', 'followup_intent_info': [{'followup_intent_name': 'followup_intent_name_value', 'parent_followup_intent_name': 'parent_followup_intent_name_value'}]}, 'intent_detection_confidence': 0.28450000000000003, 'diagnostic_info': {}, 'sentiment_analysis_result': {'query_text_sentiment': {'score': 0.54, 'magnitude': 0.9580000000000001}}}, 'intent_suggestion': {'display_name': 'display_name_value', 'intent_v2': 'intent_v2_value', 'description': 'description_value'}, 'answer_record': 'answer_record_value'}}}
    test_field = gcd_answer_record.UpdateAnswerRecordRequest.meta.fields['answer_record']

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
    for (field, value) in request_init['answer_record'].items():
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
                for i in range(0, len(request_init['answer_record'][field])):
                    del request_init['answer_record'][field][i][subfield]
            else:
                del request_init['answer_record'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcd_answer_record.AnswerRecord(name='name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = gcd_answer_record.AnswerRecord.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_answer_record(request)
    assert isinstance(response, gcd_answer_record.AnswerRecord)
    assert response.name == 'name_value'

def test_update_answer_record_rest_required_fields(request_type=gcd_answer_record.UpdateAnswerRecordRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.AnswerRecordsRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_answer_record._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_answer_record._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gcd_answer_record.AnswerRecord()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gcd_answer_record.AnswerRecord.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_answer_record(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_answer_record_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AnswerRecordsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_answer_record._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('answerRecord', 'updateMask'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_answer_record_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.AnswerRecordsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.AnswerRecordsRestInterceptor())
    client = AnswerRecordsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.AnswerRecordsRestInterceptor, 'post_update_answer_record') as post, mock.patch.object(transports.AnswerRecordsRestInterceptor, 'pre_update_answer_record') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = gcd_answer_record.UpdateAnswerRecordRequest.pb(gcd_answer_record.UpdateAnswerRecordRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gcd_answer_record.AnswerRecord.to_json(gcd_answer_record.AnswerRecord())
        request = gcd_answer_record.UpdateAnswerRecordRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gcd_answer_record.AnswerRecord()
        client.update_answer_record(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_answer_record_rest_bad_request(transport: str='rest', request_type=gcd_answer_record.UpdateAnswerRecordRequest):
    if False:
        i = 10
        return i + 15
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'answer_record': {'name': 'projects/sample1/answerRecords/sample2'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_answer_record(request)

def test_update_answer_record_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gcd_answer_record.AnswerRecord()
        sample_request = {'answer_record': {'name': 'projects/sample1/answerRecords/sample2'}}
        mock_args = dict(answer_record=gcd_answer_record.AnswerRecord(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gcd_answer_record.AnswerRecord.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_answer_record(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v2/{answer_record.name=projects/*/answerRecords/*}' % client.transport._host, args[1])

def test_update_answer_record_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_answer_record(gcd_answer_record.UpdateAnswerRecordRequest(), answer_record=gcd_answer_record.AnswerRecord(name='name_value'), update_mask=field_mask_pb2.FieldMask(paths=['paths_value']))

def test_update_answer_record_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.AnswerRecordsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.AnswerRecordsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AnswerRecordsClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.AnswerRecordsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AnswerRecordsClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = AnswerRecordsClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.AnswerRecordsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = AnswerRecordsClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AnswerRecordsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = AnswerRecordsClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.AnswerRecordsGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.AnswerRecordsGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.AnswerRecordsGrpcTransport, transports.AnswerRecordsGrpcAsyncIOTransport, transports.AnswerRecordsRestTransport])
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
    transport = AnswerRecordsClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        return 10
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.AnswerRecordsGrpcTransport)

def test_answer_records_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.AnswerRecordsTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_answer_records_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.dialogflow_v2.services.answer_records.transports.AnswerRecordsTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.AnswerRecordsTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_answer_records', 'update_answer_record', 'get_location', 'list_locations', 'get_operation', 'cancel_operation', 'list_operations')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_answer_records_base_transport_with_credentials_file():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.dialogflow_v2.services.answer_records.transports.AnswerRecordsTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AnswerRecordsTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id='octopus')

def test_answer_records_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.dialogflow_v2.services.answer_records.transports.AnswerRecordsTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.AnswerRecordsTransport()
        adc.assert_called_once()

def test_answer_records_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        AnswerRecordsClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.AnswerRecordsGrpcTransport, transports.AnswerRecordsGrpcAsyncIOTransport])
def test_answer_records_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.AnswerRecordsGrpcTransport, transports.AnswerRecordsGrpcAsyncIOTransport, transports.AnswerRecordsRestTransport])
def test_answer_records_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.AnswerRecordsGrpcTransport, grpc_helpers), (transports.AnswerRecordsGrpcAsyncIOTransport, grpc_helpers_async)])
def test_answer_records_transport_create_channel(transport_class, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('dialogflow.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/dialogflow'), scopes=['1', '2'], default_host='dialogflow.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.AnswerRecordsGrpcTransport, transports.AnswerRecordsGrpcAsyncIOTransport])
def test_answer_records_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_answer_records_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.AnswerRecordsRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_answer_records_host_no_port(transport_name):
    if False:
        return 10
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dialogflow.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('dialogflow.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_answer_records_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='dialogflow.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('dialogflow.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://dialogflow.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_answer_records_client_transport_session_collision(transport_name):
    if False:
        print('Hello World!')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = AnswerRecordsClient(credentials=creds1, transport=transport_name)
    client2 = AnswerRecordsClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_answer_records._session
    session2 = client2.transport.list_answer_records._session
    assert session1 != session2
    session1 = client1.transport.update_answer_record._session
    session2 = client2.transport.update_answer_record._session
    assert session1 != session2

def test_answer_records_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AnswerRecordsGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_answer_records_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.AnswerRecordsGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.AnswerRecordsGrpcTransport, transports.AnswerRecordsGrpcAsyncIOTransport])
def test_answer_records_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.AnswerRecordsGrpcTransport, transports.AnswerRecordsGrpcAsyncIOTransport])
def test_answer_records_transport_channel_mtls_with_adc(transport_class):
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

def test_answer_record_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    answer_record = 'clam'
    expected = 'projects/{project}/answerRecords/{answer_record}'.format(project=project, answer_record=answer_record)
    actual = AnswerRecordsClient.answer_record_path(project, answer_record)
    assert expected == actual

def test_parse_answer_record_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'whelk', 'answer_record': 'octopus'}
    path = AnswerRecordsClient.answer_record_path(**expected)
    actual = AnswerRecordsClient.parse_answer_record_path(path)
    assert expected == actual

def test_context_path():
    if False:
        i = 10
        return i + 15
    project = 'oyster'
    session = 'nudibranch'
    context = 'cuttlefish'
    expected = 'projects/{project}/agent/sessions/{session}/contexts/{context}'.format(project=project, session=session, context=context)
    actual = AnswerRecordsClient.context_path(project, session, context)
    assert expected == actual

def test_parse_context_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'mussel', 'session': 'winkle', 'context': 'nautilus'}
    path = AnswerRecordsClient.context_path(**expected)
    actual = AnswerRecordsClient.parse_context_path(path)
    assert expected == actual

def test_intent_path():
    if False:
        while True:
            i = 10
    project = 'scallop'
    intent = 'abalone'
    expected = 'projects/{project}/agent/intents/{intent}'.format(project=project, intent=intent)
    actual = AnswerRecordsClient.intent_path(project, intent)
    assert expected == actual

def test_parse_intent_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'squid', 'intent': 'clam'}
    path = AnswerRecordsClient.intent_path(**expected)
    actual = AnswerRecordsClient.parse_intent_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = AnswerRecordsClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'octopus'}
    path = AnswerRecordsClient.common_billing_account_path(**expected)
    actual = AnswerRecordsClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        print('Hello World!')
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = AnswerRecordsClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'nudibranch'}
    path = AnswerRecordsClient.common_folder_path(**expected)
    actual = AnswerRecordsClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = AnswerRecordsClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'mussel'}
    path = AnswerRecordsClient.common_organization_path(**expected)
    actual = AnswerRecordsClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = AnswerRecordsClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nautilus'}
    path = AnswerRecordsClient.common_project_path(**expected)
    actual = AnswerRecordsClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        while True:
            i = 10
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = AnswerRecordsClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'squid', 'location': 'clam'}
    path = AnswerRecordsClient.common_location_path(**expected)
    actual = AnswerRecordsClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.AnswerRecordsTransport, '_prep_wrapped_messages') as prep:
        client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.AnswerRecordsTransport, '_prep_wrapped_messages') as prep:
        transport_class = AnswerRecordsClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_location_rest_bad_request(transport: str='rest', request_type=locations_pb2.GetLocationRequest):
    if False:
        return 10
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_cancel_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.CancelOperationRequest):
    if False:
        i = 10
        return i + 15
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/operations/sample2'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.cancel_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.CancelOperationRequest, dict])
def test_cancel_operation_rest(request_type):
    if False:
        print('Hello World!')
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/operations/sample2'}
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

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        for i in range(10):
            print('nop')
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1/operations/sample2'}, request)
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
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/operations/sample2'}
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
        return 10
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'projects/sample1'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_operations(request)

@pytest.mark.parametrize('request_type', [operations_pb2.ListOperationsRequest, dict])
def test_list_operations_rest(request_type):
    if False:
        print('Hello World!')
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1'}
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

def test_cancel_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = None
        response = client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_cancel_operation_from_dict_async():
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.cancel_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.cancel_operation(request={'name': 'locations'})
        call.assert_called()

def test_get_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_list_locations(transport: str='grpc'):
    if False:
        return 10
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.ListLocationsResponse()
        response = client.list_locations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_locations_from_dict_async():
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.ListLocationsResponse())
        response = await client.list_locations(request={'name': 'locations'})
        call.assert_called()

def test_get_location(transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        i = 10
        return i + 15
    client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = locations_pb2.Location()
        response = client.get_location(request={'name': 'locations/abc'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_location_from_dict_async():
    client = AnswerRecordsAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_locations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(locations_pb2.Location())
        response = await client.get_location(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        return 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        for i in range(10):
            print('nop')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = AnswerRecordsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(AnswerRecordsClient, transports.AnswerRecordsGrpcTransport), (AnswerRecordsAsyncClient, transports.AnswerRecordsGrpcAsyncIOTransport)])
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