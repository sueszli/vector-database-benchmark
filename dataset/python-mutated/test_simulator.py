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
from google.iam.v1 import policy_pb2
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import json_format
from google.type import date_pb2
from google.type import expr_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.policysimulator_v1.services.simulator import SimulatorAsyncClient, SimulatorClient, pagers, transports
from google.cloud.policysimulator_v1.types import simulator

def client_cert_source_callback():
    if False:
        while True:
            i = 10
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        for i in range(10):
            print('nop')
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
    assert SimulatorClient._get_default_mtls_endpoint(None) is None
    assert SimulatorClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert SimulatorClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert SimulatorClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert SimulatorClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert SimulatorClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(SimulatorClient, 'grpc'), (SimulatorAsyncClient, 'grpc_asyncio'), (SimulatorClient, 'rest')])
def test_simulator_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('policysimulator.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://policysimulator.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.SimulatorGrpcTransport, 'grpc'), (transports.SimulatorGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.SimulatorRestTransport, 'rest')])
def test_simulator_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(SimulatorClient, 'grpc'), (SimulatorAsyncClient, 'grpc_asyncio'), (SimulatorClient, 'rest')])
def test_simulator_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('policysimulator.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://policysimulator.googleapis.com')

def test_simulator_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = SimulatorClient.get_transport_class()
    available_transports = [transports.SimulatorGrpcTransport, transports.SimulatorRestTransport]
    assert transport in available_transports
    transport = SimulatorClient.get_transport_class('grpc')
    assert transport == transports.SimulatorGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(SimulatorClient, transports.SimulatorGrpcTransport, 'grpc'), (SimulatorAsyncClient, transports.SimulatorGrpcAsyncIOTransport, 'grpc_asyncio'), (SimulatorClient, transports.SimulatorRestTransport, 'rest')])
@mock.patch.object(SimulatorClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SimulatorClient))
@mock.patch.object(SimulatorAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SimulatorAsyncClient))
def test_simulator_client_client_options(client_class, transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(SimulatorClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(SimulatorClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(SimulatorClient, transports.SimulatorGrpcTransport, 'grpc', 'true'), (SimulatorAsyncClient, transports.SimulatorGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (SimulatorClient, transports.SimulatorGrpcTransport, 'grpc', 'false'), (SimulatorAsyncClient, transports.SimulatorGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (SimulatorClient, transports.SimulatorRestTransport, 'rest', 'true'), (SimulatorClient, transports.SimulatorRestTransport, 'rest', 'false')])
@mock.patch.object(SimulatorClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SimulatorClient))
@mock.patch.object(SimulatorAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SimulatorAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_simulator_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [SimulatorClient, SimulatorAsyncClient])
@mock.patch.object(SimulatorClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SimulatorClient))
@mock.patch.object(SimulatorAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(SimulatorAsyncClient))
def test_simulator_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(SimulatorClient, transports.SimulatorGrpcTransport, 'grpc'), (SimulatorAsyncClient, transports.SimulatorGrpcAsyncIOTransport, 'grpc_asyncio'), (SimulatorClient, transports.SimulatorRestTransport, 'rest')])
def test_simulator_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(SimulatorClient, transports.SimulatorGrpcTransport, 'grpc', grpc_helpers), (SimulatorAsyncClient, transports.SimulatorGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (SimulatorClient, transports.SimulatorRestTransport, 'rest', None)])
def test_simulator_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_simulator_client_client_options_from_dict():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.policysimulator_v1.services.simulator.transports.SimulatorGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = SimulatorClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(SimulatorClient, transports.SimulatorGrpcTransport, 'grpc', grpc_helpers), (SimulatorAsyncClient, transports.SimulatorGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_simulator_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('policysimulator.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=None, default_host='policysimulator.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [simulator.GetReplayRequest, dict])
def test_get_replay(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_replay), '__call__') as call:
        call.return_value = simulator.Replay(name='name_value', state=simulator.Replay.State.PENDING)
        response = client.get_replay(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == simulator.GetReplayRequest()
    assert isinstance(response, simulator.Replay)
    assert response.name == 'name_value'
    assert response.state == simulator.Replay.State.PENDING

def test_get_replay_empty_call():
    if False:
        return 10
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_replay), '__call__') as call:
        client.get_replay()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == simulator.GetReplayRequest()

@pytest.mark.asyncio
async def test_get_replay_async(transport: str='grpc_asyncio', request_type=simulator.GetReplayRequest):
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_replay), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(simulator.Replay(name='name_value', state=simulator.Replay.State.PENDING))
        response = await client.get_replay(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == simulator.GetReplayRequest()
    assert isinstance(response, simulator.Replay)
    assert response.name == 'name_value'
    assert response.state == simulator.Replay.State.PENDING

@pytest.mark.asyncio
async def test_get_replay_async_from_dict():
    await test_get_replay_async(request_type=dict)

def test_get_replay_field_headers():
    if False:
        return 10
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials())
    request = simulator.GetReplayRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_replay), '__call__') as call:
        call.return_value = simulator.Replay()
        client.get_replay(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_replay_field_headers_async():
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = simulator.GetReplayRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_replay), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(simulator.Replay())
        await client.get_replay(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_replay_flattened():
    if False:
        i = 10
        return i + 15
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_replay), '__call__') as call:
        call.return_value = simulator.Replay()
        client.get_replay(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_replay_flattened_error():
    if False:
        print('Hello World!')
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_replay(simulator.GetReplayRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_replay_flattened_async():
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_replay), '__call__') as call:
        call.return_value = simulator.Replay()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(simulator.Replay())
        response = await client.get_replay(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_replay_flattened_error_async():
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_replay(simulator.GetReplayRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [simulator.CreateReplayRequest, dict])
def test_create_replay(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_replay), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_replay(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == simulator.CreateReplayRequest()
    assert isinstance(response, future.Future)

def test_create_replay_empty_call():
    if False:
        print('Hello World!')
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_replay), '__call__') as call:
        client.create_replay()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == simulator.CreateReplayRequest()

@pytest.mark.asyncio
async def test_create_replay_async(transport: str='grpc_asyncio', request_type=simulator.CreateReplayRequest):
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_replay), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_replay(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == simulator.CreateReplayRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_replay_async_from_dict():
    await test_create_replay_async(request_type=dict)

def test_create_replay_field_headers():
    if False:
        print('Hello World!')
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials())
    request = simulator.CreateReplayRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_replay), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_replay(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_replay_field_headers_async():
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = simulator.CreateReplayRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_replay), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_replay(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_replay_flattened():
    if False:
        i = 10
        return i + 15
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_replay), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_replay(parent='parent_value', replay=simulator.Replay(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].replay
        mock_val = simulator.Replay(name='name_value')
        assert arg == mock_val

def test_create_replay_flattened_error():
    if False:
        i = 10
        return i + 15
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_replay(simulator.CreateReplayRequest(), parent='parent_value', replay=simulator.Replay(name='name_value'))

@pytest.mark.asyncio
async def test_create_replay_flattened_async():
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_replay), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_replay(parent='parent_value', replay=simulator.Replay(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].replay
        mock_val = simulator.Replay(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_replay_flattened_error_async():
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_replay(simulator.CreateReplayRequest(), parent='parent_value', replay=simulator.Replay(name='name_value'))

@pytest.mark.parametrize('request_type', [simulator.ListReplayResultsRequest, dict])
def test_list_replay_results(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_replay_results), '__call__') as call:
        call.return_value = simulator.ListReplayResultsResponse(next_page_token='next_page_token_value')
        response = client.list_replay_results(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == simulator.ListReplayResultsRequest()
    assert isinstance(response, pagers.ListReplayResultsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_replay_results_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_replay_results), '__call__') as call:
        client.list_replay_results()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == simulator.ListReplayResultsRequest()

@pytest.mark.asyncio
async def test_list_replay_results_async(transport: str='grpc_asyncio', request_type=simulator.ListReplayResultsRequest):
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_replay_results), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(simulator.ListReplayResultsResponse(next_page_token='next_page_token_value'))
        response = await client.list_replay_results(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == simulator.ListReplayResultsRequest()
    assert isinstance(response, pagers.ListReplayResultsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_replay_results_async_from_dict():
    await test_list_replay_results_async(request_type=dict)

def test_list_replay_results_field_headers():
    if False:
        i = 10
        return i + 15
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials())
    request = simulator.ListReplayResultsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_replay_results), '__call__') as call:
        call.return_value = simulator.ListReplayResultsResponse()
        client.list_replay_results(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_replay_results_field_headers_async():
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = simulator.ListReplayResultsRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_replay_results), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(simulator.ListReplayResultsResponse())
        await client.list_replay_results(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_replay_results_flattened():
    if False:
        i = 10
        return i + 15
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_replay_results), '__call__') as call:
        call.return_value = simulator.ListReplayResultsResponse()
        client.list_replay_results(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_replay_results_flattened_error():
    if False:
        print('Hello World!')
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_replay_results(simulator.ListReplayResultsRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_replay_results_flattened_async():
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_replay_results), '__call__') as call:
        call.return_value = simulator.ListReplayResultsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(simulator.ListReplayResultsResponse())
        response = await client.list_replay_results(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_replay_results_flattened_error_async():
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_replay_results(simulator.ListReplayResultsRequest(), parent='parent_value')

def test_list_replay_results_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_replay_results), '__call__') as call:
        call.side_effect = (simulator.ListReplayResultsResponse(replay_results=[simulator.ReplayResult(), simulator.ReplayResult(), simulator.ReplayResult()], next_page_token='abc'), simulator.ListReplayResultsResponse(replay_results=[], next_page_token='def'), simulator.ListReplayResultsResponse(replay_results=[simulator.ReplayResult()], next_page_token='ghi'), simulator.ListReplayResultsResponse(replay_results=[simulator.ReplayResult(), simulator.ReplayResult()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_replay_results(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, simulator.ReplayResult) for i in results))

def test_list_replay_results_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_replay_results), '__call__') as call:
        call.side_effect = (simulator.ListReplayResultsResponse(replay_results=[simulator.ReplayResult(), simulator.ReplayResult(), simulator.ReplayResult()], next_page_token='abc'), simulator.ListReplayResultsResponse(replay_results=[], next_page_token='def'), simulator.ListReplayResultsResponse(replay_results=[simulator.ReplayResult()], next_page_token='ghi'), simulator.ListReplayResultsResponse(replay_results=[simulator.ReplayResult(), simulator.ReplayResult()]), RuntimeError)
        pages = list(client.list_replay_results(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_replay_results_async_pager():
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_replay_results), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (simulator.ListReplayResultsResponse(replay_results=[simulator.ReplayResult(), simulator.ReplayResult(), simulator.ReplayResult()], next_page_token='abc'), simulator.ListReplayResultsResponse(replay_results=[], next_page_token='def'), simulator.ListReplayResultsResponse(replay_results=[simulator.ReplayResult()], next_page_token='ghi'), simulator.ListReplayResultsResponse(replay_results=[simulator.ReplayResult(), simulator.ReplayResult()]), RuntimeError)
        async_pager = await client.list_replay_results(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, simulator.ReplayResult) for i in responses))

@pytest.mark.asyncio
async def test_list_replay_results_async_pages():
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_replay_results), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (simulator.ListReplayResultsResponse(replay_results=[simulator.ReplayResult(), simulator.ReplayResult(), simulator.ReplayResult()], next_page_token='abc'), simulator.ListReplayResultsResponse(replay_results=[], next_page_token='def'), simulator.ListReplayResultsResponse(replay_results=[simulator.ReplayResult()], next_page_token='ghi'), simulator.ListReplayResultsResponse(replay_results=[simulator.ReplayResult(), simulator.ReplayResult()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_replay_results(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [simulator.GetReplayRequest, dict])
def test_get_replay_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/locations/sample2/replays/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = simulator.Replay(name='name_value', state=simulator.Replay.State.PENDING)
        response_value = Response()
        response_value.status_code = 200
        return_value = simulator.Replay.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_replay(request)
    assert isinstance(response, simulator.Replay)
    assert response.name == 'name_value'
    assert response.state == simulator.Replay.State.PENDING

def test_get_replay_rest_required_fields(request_type=simulator.GetReplayRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.SimulatorRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_replay._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_replay._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = simulator.Replay()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = simulator.Replay.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_replay(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_replay_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.SimulatorRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_replay._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_replay_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.SimulatorRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SimulatorRestInterceptor())
    client = SimulatorClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SimulatorRestInterceptor, 'post_get_replay') as post, mock.patch.object(transports.SimulatorRestInterceptor, 'pre_get_replay') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = simulator.GetReplayRequest.pb(simulator.GetReplayRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = simulator.Replay.to_json(simulator.Replay())
        request = simulator.GetReplayRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = simulator.Replay()
        client.get_replay(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_replay_rest_bad_request(transport: str='rest', request_type=simulator.GetReplayRequest):
    if False:
        for i in range(10):
            print('nop')
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/locations/sample2/replays/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_replay(request)

def test_get_replay_rest_flattened():
    if False:
        print('Hello World!')
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = simulator.Replay()
        sample_request = {'name': 'projects/sample1/locations/sample2/replays/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = simulator.Replay.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_replay(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*/locations/*/replays/*}' % client.transport._host, args[1])

def test_get_replay_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_replay(simulator.GetReplayRequest(), name='name_value')

def test_get_replay_rest_error():
    if False:
        while True:
            i = 10
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [simulator.CreateReplayRequest, dict])
def test_create_replay_rest(request_type):
    if False:
        print('Hello World!')
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request_init['replay'] = {'name': 'name_value', 'state': 1, 'config': {'policy_overlay': {}, 'log_source': 1}, 'results_summary': {'log_count': 970, 'unchanged_count': 1589, 'difference_count': 1683, 'error_count': 1202, 'oldest_date': {'year': 433, 'month': 550, 'day': 318}, 'newest_date': {}}}
    test_field = simulator.CreateReplayRequest.meta.fields['replay']

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
    for (field, value) in request_init['replay'].items():
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
                for i in range(0, len(request_init['replay'][field])):
                    del request_init['replay'][field][i][subfield]
            else:
                del request_init['replay'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_replay(request)
    assert response.operation.name == 'operations/spam'

def test_create_replay_rest_required_fields(request_type=simulator.CreateReplayRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.SimulatorRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_replay._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_replay._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_replay(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_replay_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.SimulatorRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_replay._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'replay'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_replay_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SimulatorRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SimulatorRestInterceptor())
    client = SimulatorClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.SimulatorRestInterceptor, 'post_create_replay') as post, mock.patch.object(transports.SimulatorRestInterceptor, 'pre_create_replay') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = simulator.CreateReplayRequest.pb(simulator.CreateReplayRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = simulator.CreateReplayRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_replay(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_replay_rest_bad_request(transport: str='rest', request_type=simulator.CreateReplayRequest):
    if False:
        print('Hello World!')
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_replay(request)

def test_create_replay_rest_flattened():
    if False:
        return 10
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'parent': 'projects/sample1/locations/sample2'}
        mock_args = dict(parent='parent_value', replay=simulator.Replay(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_replay(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*}/replays' % client.transport._host, args[1])

def test_create_replay_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_replay(simulator.CreateReplayRequest(), parent='parent_value', replay=simulator.Replay(name='name_value'))

def test_create_replay_rest_error():
    if False:
        return 10
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [simulator.ListReplayResultsRequest, dict])
def test_list_replay_results_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/locations/sample2/replays/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = simulator.ListReplayResultsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = simulator.ListReplayResultsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_replay_results(request)
    assert isinstance(response, pagers.ListReplayResultsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_replay_results_rest_required_fields(request_type=simulator.ListReplayResultsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.SimulatorRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_replay_results._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_replay_results._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = simulator.ListReplayResultsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = simulator.ListReplayResultsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_replay_results(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_replay_results_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.SimulatorRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_replay_results._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_replay_results_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.SimulatorRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.SimulatorRestInterceptor())
    client = SimulatorClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.SimulatorRestInterceptor, 'post_list_replay_results') as post, mock.patch.object(transports.SimulatorRestInterceptor, 'pre_list_replay_results') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = simulator.ListReplayResultsRequest.pb(simulator.ListReplayResultsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = simulator.ListReplayResultsResponse.to_json(simulator.ListReplayResultsResponse())
        request = simulator.ListReplayResultsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = simulator.ListReplayResultsResponse()
        client.list_replay_results(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_replay_results_rest_bad_request(transport: str='rest', request_type=simulator.ListReplayResultsRequest):
    if False:
        i = 10
        return i + 15
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/locations/sample2/replays/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_replay_results(request)

def test_list_replay_results_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = simulator.ListReplayResultsResponse()
        sample_request = {'parent': 'projects/sample1/locations/sample2/replays/sample3'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = simulator.ListReplayResultsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_replay_results(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{parent=projects/*/locations/*/replays/*}/results' % client.transport._host, args[1])

def test_list_replay_results_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_replay_results(simulator.ListReplayResultsRequest(), parent='parent_value')

def test_list_replay_results_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (simulator.ListReplayResultsResponse(replay_results=[simulator.ReplayResult(), simulator.ReplayResult(), simulator.ReplayResult()], next_page_token='abc'), simulator.ListReplayResultsResponse(replay_results=[], next_page_token='def'), simulator.ListReplayResultsResponse(replay_results=[simulator.ReplayResult()], next_page_token='ghi'), simulator.ListReplayResultsResponse(replay_results=[simulator.ReplayResult(), simulator.ReplayResult()]))
        response = response + response
        response = tuple((simulator.ListReplayResultsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/locations/sample2/replays/sample3'}
        pager = client.list_replay_results(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, simulator.ReplayResult) for i in results))
        pages = list(client.list_replay_results(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.SimulatorGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.SimulatorGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SimulatorClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.SimulatorGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = SimulatorClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = SimulatorClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.SimulatorGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = SimulatorClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.SimulatorGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = SimulatorClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        while True:
            i = 10
    transport = transports.SimulatorGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.SimulatorGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.SimulatorGrpcTransport, transports.SimulatorGrpcAsyncIOTransport, transports.SimulatorRestTransport])
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
        while True:
            i = 10
    transport = SimulatorClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.SimulatorGrpcTransport)

def test_simulator_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.SimulatorTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_simulator_base_transport():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.policysimulator_v1.services.simulator.transports.SimulatorTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.SimulatorTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('get_replay', 'create_replay', 'list_replay_results', 'get_operation', 'list_operations')
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

def test_simulator_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.policysimulator_v1.services.simulator.transports.SimulatorTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.SimulatorTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

def test_simulator_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.policysimulator_v1.services.simulator.transports.SimulatorTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.SimulatorTransport()
        adc.assert_called_once()

def test_simulator_auth_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        SimulatorClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.SimulatorGrpcTransport, transports.SimulatorGrpcAsyncIOTransport])
def test_simulator_transport_auth_adc(transport_class):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform',), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.SimulatorGrpcTransport, transports.SimulatorGrpcAsyncIOTransport, transports.SimulatorRestTransport])
def test_simulator_transport_auth_gdch_credentials(transport_class):
    if False:
        while True:
            i = 10
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.SimulatorGrpcTransport, grpc_helpers), (transports.SimulatorGrpcAsyncIOTransport, grpc_helpers_async)])
def test_simulator_transport_create_channel(transport_class, grpc_helpers):
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('policysimulator.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform',), scopes=['1', '2'], default_host='policysimulator.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.SimulatorGrpcTransport, transports.SimulatorGrpcAsyncIOTransport])
def test_simulator_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_simulator_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.SimulatorRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_simulator_rest_lro_client():
    if False:
        while True:
            i = 10
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_simulator_host_no_port(transport_name):
    if False:
        return 10
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='policysimulator.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('policysimulator.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://policysimulator.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_simulator_host_with_port(transport_name):
    if False:
        print('Hello World!')
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='policysimulator.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('policysimulator.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://policysimulator.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_simulator_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = SimulatorClient(credentials=creds1, transport=transport_name)
    client2 = SimulatorClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.get_replay._session
    session2 = client2.transport.get_replay._session
    assert session1 != session2
    session1 = client1.transport.create_replay._session
    session2 = client2.transport.create_replay._session
    assert session1 != session2
    session1 = client1.transport.list_replay_results._session
    session2 = client2.transport.list_replay_results._session
    assert session1 != session2

def test_simulator_grpc_transport_channel():
    if False:
        return 10
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.SimulatorGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_simulator_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.SimulatorGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.SimulatorGrpcTransport, transports.SimulatorGrpcAsyncIOTransport])
def test_simulator_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.SimulatorGrpcTransport, transports.SimulatorGrpcAsyncIOTransport])
def test_simulator_transport_channel_mtls_with_adc(transport_class):
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

def test_simulator_grpc_lro_client():
    if False:
        return 10
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_simulator_grpc_lro_async_client():
    if False:
        for i in range(10):
            print('nop')
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_replay_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'squid'
    location = 'clam'
    replay = 'whelk'
    expected = 'projects/{project}/locations/{location}/replays/{replay}'.format(project=project, location=location, replay=replay)
    actual = SimulatorClient.replay_path(project, location, replay)
    assert expected == actual

def test_parse_replay_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus', 'location': 'oyster', 'replay': 'nudibranch'}
    path = SimulatorClient.replay_path(**expected)
    actual = SimulatorClient.parse_replay_path(path)
    assert expected == actual

def test_replay_result_path():
    if False:
        i = 10
        return i + 15
    project = 'cuttlefish'
    location = 'mussel'
    replay = 'winkle'
    replay_result = 'nautilus'
    expected = 'projects/{project}/locations/{location}/replays/{replay}/results/{replay_result}'.format(project=project, location=location, replay=replay, replay_result=replay_result)
    actual = SimulatorClient.replay_result_path(project, location, replay, replay_result)
    assert expected == actual

def test_parse_replay_result_path():
    if False:
        return 10
    expected = {'project': 'scallop', 'location': 'abalone', 'replay': 'squid', 'replay_result': 'clam'}
    path = SimulatorClient.replay_result_path(**expected)
    actual = SimulatorClient.parse_replay_result_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        return 10
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = SimulatorClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'octopus'}
    path = SimulatorClient.common_billing_account_path(**expected)
    actual = SimulatorClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = SimulatorClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'nudibranch'}
    path = SimulatorClient.common_folder_path(**expected)
    actual = SimulatorClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = SimulatorClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'mussel'}
    path = SimulatorClient.common_organization_path(**expected)
    actual = SimulatorClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = SimulatorClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'nautilus'}
    path = SimulatorClient.common_project_path(**expected)
    actual = SimulatorClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = SimulatorClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'squid', 'location': 'clam'}
    path = SimulatorClient.common_location_path(**expected)
    actual = SimulatorClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        for i in range(10):
            print('nop')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.SimulatorTransport, '_prep_wrapped_messages') as prep:
        client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.SimulatorTransport, '_prep_wrapped_messages') as prep:
        transport_class = SimulatorClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        i = 10
        return i + 15
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'operations/sample1'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_operation(request)

@pytest.mark.parametrize('request_type', [operations_pb2.GetOperationRequest, dict])
def test_get_operation_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'operations/sample1'}
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
        i = 10
        return i + 15
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'name': 'operations'}, request)
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
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'operations'}
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
        for i in range(10):
            print('nop')
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        return 10
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_list_operations(transport: str='grpc'):
    if False:
        return 10
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = SimulatorAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = SimulatorClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(SimulatorClient, transports.SimulatorGrpcTransport), (SimulatorAsyncClient, transports.SimulatorGrpcAsyncIOTransport)])
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