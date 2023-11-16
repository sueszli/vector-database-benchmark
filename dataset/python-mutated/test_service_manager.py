import os
try:
    from unittest import mock
    from unittest.mock import AsyncMock
except ImportError:
    import mock
from collections.abc import Iterable
import json
import math
from google.api import auth_pb2
from google.api import backend_pb2
from google.api import billing_pb2
from google.api import client_pb2
from google.api import context_pb2
from google.api import control_pb2
from google.api import documentation_pb2
from google.api import endpoint_pb2
from google.api import http_pb2
from google.api import label_pb2
from google.api import launch_stage_pb2
from google.api import log_pb2
from google.api import logging_pb2
from google.api import metric_pb2
from google.api import monitored_resource_pb2
from google.api import monitoring_pb2
from google.api import policy_pb2
from google.api import quota_pb2
from google.api import service_pb2
from google.api import source_info_pb2
from google.api import system_parameter_pb2
from google.api import usage_pb2
from google.api_core import future, gapic_v1, grpc_helpers, grpc_helpers_async, operation, operations_v1, path_template
from google.api_core import client_options
from google.api_core import exceptions as core_exceptions
from google.api_core import operation_async
import google.auth
from google.auth import credentials as ga_credentials
from google.auth.exceptions import MutualTLSChannelError
from google.iam.v1 import iam_policy_pb2
from google.iam.v1 import options_pb2
from google.iam.v1 import policy_pb2
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import any_pb2
from google.protobuf import api_pb2
from google.protobuf import duration_pb2
from google.protobuf import empty_pb2
from google.protobuf import json_format
from google.protobuf import source_context_pb2
from google.protobuf import timestamp_pb2
from google.protobuf import type_pb2
from google.protobuf import wrappers_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.servicemanagement_v1.services.service_manager import ServiceManagerAsyncClient, ServiceManagerClient, pagers, transports
from google.cloud.servicemanagement_v1.types import resources, servicemanager

def client_cert_source_callback():
    if False:
        return 10
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
    assert ServiceManagerClient._get_default_mtls_endpoint(None) is None
    assert ServiceManagerClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert ServiceManagerClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert ServiceManagerClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert ServiceManagerClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert ServiceManagerClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(ServiceManagerClient, 'grpc'), (ServiceManagerAsyncClient, 'grpc_asyncio'), (ServiceManagerClient, 'rest')])
def test_service_manager_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('servicemanagement.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://servicemanagement.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.ServiceManagerGrpcTransport, 'grpc'), (transports.ServiceManagerGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.ServiceManagerRestTransport, 'rest')])
def test_service_manager_client_service_account_always_use_jwt(transport_class, transport_name):
    if False:
        print('Hello World!')
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=True)
        use_jwt.assert_called_once_with(True)
    with mock.patch.object(service_account.Credentials, 'with_always_use_jwt_access', create=True) as use_jwt:
        creds = service_account.Credentials(None, None, None)
        transport = transport_class(credentials=creds, always_use_jwt_access=False)
        use_jwt.assert_not_called()

@pytest.mark.parametrize('client_class,transport_name', [(ServiceManagerClient, 'grpc'), (ServiceManagerAsyncClient, 'grpc_asyncio'), (ServiceManagerClient, 'rest')])
def test_service_manager_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('servicemanagement.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://servicemanagement.googleapis.com')

def test_service_manager_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = ServiceManagerClient.get_transport_class()
    available_transports = [transports.ServiceManagerGrpcTransport, transports.ServiceManagerRestTransport]
    assert transport in available_transports
    transport = ServiceManagerClient.get_transport_class('grpc')
    assert transport == transports.ServiceManagerGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ServiceManagerClient, transports.ServiceManagerGrpcTransport, 'grpc'), (ServiceManagerAsyncClient, transports.ServiceManagerGrpcAsyncIOTransport, 'grpc_asyncio'), (ServiceManagerClient, transports.ServiceManagerRestTransport, 'rest')])
@mock.patch.object(ServiceManagerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceManagerClient))
@mock.patch.object(ServiceManagerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceManagerAsyncClient))
def test_service_manager_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(ServiceManagerClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(ServiceManagerClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(ServiceManagerClient, transports.ServiceManagerGrpcTransport, 'grpc', 'true'), (ServiceManagerAsyncClient, transports.ServiceManagerGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (ServiceManagerClient, transports.ServiceManagerGrpcTransport, 'grpc', 'false'), (ServiceManagerAsyncClient, transports.ServiceManagerGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (ServiceManagerClient, transports.ServiceManagerRestTransport, 'rest', 'true'), (ServiceManagerClient, transports.ServiceManagerRestTransport, 'rest', 'false')])
@mock.patch.object(ServiceManagerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceManagerClient))
@mock.patch.object(ServiceManagerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceManagerAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_service_manager_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [ServiceManagerClient, ServiceManagerAsyncClient])
@mock.patch.object(ServiceManagerClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceManagerClient))
@mock.patch.object(ServiceManagerAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(ServiceManagerAsyncClient))
def test_service_manager_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(ServiceManagerClient, transports.ServiceManagerGrpcTransport, 'grpc'), (ServiceManagerAsyncClient, transports.ServiceManagerGrpcAsyncIOTransport, 'grpc_asyncio'), (ServiceManagerClient, transports.ServiceManagerRestTransport, 'rest')])
def test_service_manager_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ServiceManagerClient, transports.ServiceManagerGrpcTransport, 'grpc', grpc_helpers), (ServiceManagerAsyncClient, transports.ServiceManagerGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (ServiceManagerClient, transports.ServiceManagerRestTransport, 'rest', None)])
def test_service_manager_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_service_manager_client_client_options_from_dict():
    if False:
        return 10
    with mock.patch('google.cloud.servicemanagement_v1.services.service_manager.transports.ServiceManagerGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = ServiceManagerClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(ServiceManagerClient, transports.ServiceManagerGrpcTransport, 'grpc', grpc_helpers), (ServiceManagerAsyncClient, transports.ServiceManagerGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_service_manager_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('servicemanagement.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only', 'https://www.googleapis.com/auth/service.management', 'https://www.googleapis.com/auth/service.management.readonly'), scopes=None, default_host='servicemanagement.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [servicemanager.ListServicesRequest, dict])
def test_list_services(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = servicemanager.ListServicesResponse(next_page_token='next_page_token_value')
        response = client.list_services(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.ListServicesRequest()
    assert isinstance(response, pagers.ListServicesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_services_empty_call():
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        client.list_services()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.ListServicesRequest()

@pytest.mark.asyncio
async def test_list_services_async(transport: str='grpc_asyncio', request_type=servicemanager.ListServicesRequest):
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(servicemanager.ListServicesResponse(next_page_token='next_page_token_value'))
        response = await client.list_services(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.ListServicesRequest()
    assert isinstance(response, pagers.ListServicesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_services_async_from_dict():
    await test_list_services_async(request_type=dict)

def test_list_services_flattened():
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = servicemanager.ListServicesResponse()
        client.list_services(producer_project_id='producer_project_id_value', consumer_id='consumer_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].producer_project_id
        mock_val = 'producer_project_id_value'
        assert arg == mock_val
        arg = args[0].consumer_id
        mock_val = 'consumer_id_value'
        assert arg == mock_val

def test_list_services_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_services(servicemanager.ListServicesRequest(), producer_project_id='producer_project_id_value', consumer_id='consumer_id_value')

@pytest.mark.asyncio
async def test_list_services_flattened_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.return_value = servicemanager.ListServicesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(servicemanager.ListServicesResponse())
        response = await client.list_services(producer_project_id='producer_project_id_value', consumer_id='consumer_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].producer_project_id
        mock_val = 'producer_project_id_value'
        assert arg == mock_val
        arg = args[0].consumer_id
        mock_val = 'consumer_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_services_flattened_error_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_services(servicemanager.ListServicesRequest(), producer_project_id='producer_project_id_value', consumer_id='consumer_id_value')

def test_list_services_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.side_effect = (servicemanager.ListServicesResponse(services=[resources.ManagedService(), resources.ManagedService(), resources.ManagedService()], next_page_token='abc'), servicemanager.ListServicesResponse(services=[], next_page_token='def'), servicemanager.ListServicesResponse(services=[resources.ManagedService()], next_page_token='ghi'), servicemanager.ListServicesResponse(services=[resources.ManagedService(), resources.ManagedService()]), RuntimeError)
        metadata = ()
        pager = client.list_services(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.ManagedService) for i in results))

def test_list_services_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_services), '__call__') as call:
        call.side_effect = (servicemanager.ListServicesResponse(services=[resources.ManagedService(), resources.ManagedService(), resources.ManagedService()], next_page_token='abc'), servicemanager.ListServicesResponse(services=[], next_page_token='def'), servicemanager.ListServicesResponse(services=[resources.ManagedService()], next_page_token='ghi'), servicemanager.ListServicesResponse(services=[resources.ManagedService(), resources.ManagedService()]), RuntimeError)
        pages = list(client.list_services(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_services_async_pager():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_services), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (servicemanager.ListServicesResponse(services=[resources.ManagedService(), resources.ManagedService(), resources.ManagedService()], next_page_token='abc'), servicemanager.ListServicesResponse(services=[], next_page_token='def'), servicemanager.ListServicesResponse(services=[resources.ManagedService()], next_page_token='ghi'), servicemanager.ListServicesResponse(services=[resources.ManagedService(), resources.ManagedService()]), RuntimeError)
        async_pager = await client.list_services(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.ManagedService) for i in responses))

@pytest.mark.asyncio
async def test_list_services_async_pages():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_services), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (servicemanager.ListServicesResponse(services=[resources.ManagedService(), resources.ManagedService(), resources.ManagedService()], next_page_token='abc'), servicemanager.ListServicesResponse(services=[], next_page_token='def'), servicemanager.ListServicesResponse(services=[resources.ManagedService()], next_page_token='ghi'), servicemanager.ListServicesResponse(services=[resources.ManagedService(), resources.ManagedService()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_services(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [servicemanager.GetServiceRequest, dict])
def test_get_service(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = resources.ManagedService(service_name='service_name_value', producer_project_id='producer_project_id_value')
        response = client.get_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.GetServiceRequest()
    assert isinstance(response, resources.ManagedService)
    assert response.service_name == 'service_name_value'
    assert response.producer_project_id == 'producer_project_id_value'

def test_get_service_empty_call():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        client.get_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.GetServiceRequest()

@pytest.mark.asyncio
async def test_get_service_async(transport: str='grpc_asyncio', request_type=servicemanager.GetServiceRequest):
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.ManagedService(service_name='service_name_value', producer_project_id='producer_project_id_value'))
        response = await client.get_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.GetServiceRequest()
    assert isinstance(response, resources.ManagedService)
    assert response.service_name == 'service_name_value'
    assert response.producer_project_id == 'producer_project_id_value'

@pytest.mark.asyncio
async def test_get_service_async_from_dict():
    await test_get_service_async(request_type=dict)

def test_get_service_field_headers():
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.GetServiceRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = resources.ManagedService()
        client.get_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_service_field_headers_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.GetServiceRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.ManagedService())
        await client.get_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

def test_get_service_flattened():
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = resources.ManagedService()
        client.get_service(service_name='service_name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val

def test_get_service_flattened_error():
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_service(servicemanager.GetServiceRequest(), service_name='service_name_value')

@pytest.mark.asyncio
async def test_get_service_flattened_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_service), '__call__') as call:
        call.return_value = resources.ManagedService()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.ManagedService())
        response = await client.get_service(service_name='service_name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_service_flattened_error_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_service(servicemanager.GetServiceRequest(), service_name='service_name_value')

@pytest.mark.parametrize('request_type', [servicemanager.CreateServiceRequest, dict])
def test_create_service(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.CreateServiceRequest()
    assert isinstance(response, future.Future)

def test_create_service_empty_call():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        client.create_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.CreateServiceRequest()

@pytest.mark.asyncio
async def test_create_service_async(transport: str='grpc_asyncio', request_type=servicemanager.CreateServiceRequest):
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.CreateServiceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_service_async_from_dict():
    await test_create_service_async(request_type=dict)

def test_create_service_flattened():
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_service(service=resources.ManagedService(service_name='service_name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service
        mock_val = resources.ManagedService(service_name='service_name_value')
        assert arg == mock_val

def test_create_service_flattened_error():
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_service(servicemanager.CreateServiceRequest(), service=resources.ManagedService(service_name='service_name_value'))

@pytest.mark.asyncio
async def test_create_service_flattened_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_service(service=resources.ManagedService(service_name='service_name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service
        mock_val = resources.ManagedService(service_name='service_name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_service_flattened_error_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_service(servicemanager.CreateServiceRequest(), service=resources.ManagedService(service_name='service_name_value'))

@pytest.mark.parametrize('request_type', [servicemanager.DeleteServiceRequest, dict])
def test_delete_service(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.delete_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.DeleteServiceRequest()
    assert isinstance(response, future.Future)

def test_delete_service_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        client.delete_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.DeleteServiceRequest()

@pytest.mark.asyncio
async def test_delete_service_async(transport: str='grpc_asyncio', request_type=servicemanager.DeleteServiceRequest):
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.DeleteServiceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_delete_service_async_from_dict():
    await test_delete_service_async(request_type=dict)

def test_delete_service_field_headers():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.DeleteServiceRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_service_field_headers_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.DeleteServiceRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.delete_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

def test_delete_service_flattened():
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.delete_service(service_name='service_name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val

def test_delete_service_flattened_error():
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_service(servicemanager.DeleteServiceRequest(), service_name='service_name_value')

@pytest.mark.asyncio
async def test_delete_service_flattened_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.delete_service(service_name='service_name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_service_flattened_error_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_service(servicemanager.DeleteServiceRequest(), service_name='service_name_value')

@pytest.mark.parametrize('request_type', [servicemanager.UndeleteServiceRequest, dict])
def test_undelete_service(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.undelete_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.undelete_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.UndeleteServiceRequest()
    assert isinstance(response, future.Future)

def test_undelete_service_empty_call():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.undelete_service), '__call__') as call:
        client.undelete_service()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.UndeleteServiceRequest()

@pytest.mark.asyncio
async def test_undelete_service_async(transport: str='grpc_asyncio', request_type=servicemanager.UndeleteServiceRequest):
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.undelete_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.undelete_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.UndeleteServiceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_undelete_service_async_from_dict():
    await test_undelete_service_async(request_type=dict)

def test_undelete_service_field_headers():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.UndeleteServiceRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.undelete_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.undelete_service(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_undelete_service_field_headers_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.UndeleteServiceRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.undelete_service), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.undelete_service(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

def test_undelete_service_flattened():
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.undelete_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.undelete_service(service_name='service_name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val

def test_undelete_service_flattened_error():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.undelete_service(servicemanager.UndeleteServiceRequest(), service_name='service_name_value')

@pytest.mark.asyncio
async def test_undelete_service_flattened_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.undelete_service), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.undelete_service(service_name='service_name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_undelete_service_flattened_error_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.undelete_service(servicemanager.UndeleteServiceRequest(), service_name='service_name_value')

@pytest.mark.parametrize('request_type', [servicemanager.ListServiceConfigsRequest, dict])
def test_list_service_configs(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_service_configs), '__call__') as call:
        call.return_value = servicemanager.ListServiceConfigsResponse(next_page_token='next_page_token_value')
        response = client.list_service_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.ListServiceConfigsRequest()
    assert isinstance(response, pagers.ListServiceConfigsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_service_configs_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_service_configs), '__call__') as call:
        client.list_service_configs()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.ListServiceConfigsRequest()

@pytest.mark.asyncio
async def test_list_service_configs_async(transport: str='grpc_asyncio', request_type=servicemanager.ListServiceConfigsRequest):
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_service_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(servicemanager.ListServiceConfigsResponse(next_page_token='next_page_token_value'))
        response = await client.list_service_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.ListServiceConfigsRequest()
    assert isinstance(response, pagers.ListServiceConfigsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_service_configs_async_from_dict():
    await test_list_service_configs_async(request_type=dict)

def test_list_service_configs_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.ListServiceConfigsRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.list_service_configs), '__call__') as call:
        call.return_value = servicemanager.ListServiceConfigsResponse()
        client.list_service_configs(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_service_configs_field_headers_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.ListServiceConfigsRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.list_service_configs), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(servicemanager.ListServiceConfigsResponse())
        await client.list_service_configs(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

def test_list_service_configs_flattened():
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_service_configs), '__call__') as call:
        call.return_value = servicemanager.ListServiceConfigsResponse()
        client.list_service_configs(service_name='service_name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val

def test_list_service_configs_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_service_configs(servicemanager.ListServiceConfigsRequest(), service_name='service_name_value')

@pytest.mark.asyncio
async def test_list_service_configs_flattened_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_service_configs), '__call__') as call:
        call.return_value = servicemanager.ListServiceConfigsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(servicemanager.ListServiceConfigsResponse())
        response = await client.list_service_configs(service_name='service_name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_service_configs_flattened_error_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_service_configs(servicemanager.ListServiceConfigsRequest(), service_name='service_name_value')

def test_list_service_configs_pager(transport_name: str='grpc'):
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_service_configs), '__call__') as call:
        call.side_effect = (servicemanager.ListServiceConfigsResponse(service_configs=[service_pb2.Service(), service_pb2.Service(), service_pb2.Service()], next_page_token='abc'), servicemanager.ListServiceConfigsResponse(service_configs=[], next_page_token='def'), servicemanager.ListServiceConfigsResponse(service_configs=[service_pb2.Service()], next_page_token='ghi'), servicemanager.ListServiceConfigsResponse(service_configs=[service_pb2.Service(), service_pb2.Service()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('service_name', ''),)),)
        pager = client.list_service_configs(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service_pb2.Service) for i in results))

def test_list_service_configs_pages(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_service_configs), '__call__') as call:
        call.side_effect = (servicemanager.ListServiceConfigsResponse(service_configs=[service_pb2.Service(), service_pb2.Service(), service_pb2.Service()], next_page_token='abc'), servicemanager.ListServiceConfigsResponse(service_configs=[], next_page_token='def'), servicemanager.ListServiceConfigsResponse(service_configs=[service_pb2.Service()], next_page_token='ghi'), servicemanager.ListServiceConfigsResponse(service_configs=[service_pb2.Service(), service_pb2.Service()]), RuntimeError)
        pages = list(client.list_service_configs(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_service_configs_async_pager():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_service_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (servicemanager.ListServiceConfigsResponse(service_configs=[service_pb2.Service(), service_pb2.Service(), service_pb2.Service()], next_page_token='abc'), servicemanager.ListServiceConfigsResponse(service_configs=[], next_page_token='def'), servicemanager.ListServiceConfigsResponse(service_configs=[service_pb2.Service()], next_page_token='ghi'), servicemanager.ListServiceConfigsResponse(service_configs=[service_pb2.Service(), service_pb2.Service()]), RuntimeError)
        async_pager = await client.list_service_configs(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, service_pb2.Service) for i in responses))

@pytest.mark.asyncio
async def test_list_service_configs_async_pages():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_service_configs), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (servicemanager.ListServiceConfigsResponse(service_configs=[service_pb2.Service(), service_pb2.Service(), service_pb2.Service()], next_page_token='abc'), servicemanager.ListServiceConfigsResponse(service_configs=[], next_page_token='def'), servicemanager.ListServiceConfigsResponse(service_configs=[service_pb2.Service()], next_page_token='ghi'), servicemanager.ListServiceConfigsResponse(service_configs=[service_pb2.Service(), service_pb2.Service()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_service_configs(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [servicemanager.GetServiceConfigRequest, dict])
def test_get_service_config(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_service_config), '__call__') as call:
        call.return_value = service_pb2.Service(name='name_value', title='title_value', producer_project_id='producer_project_id_value', id='id_value')
        response = client.get_service_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.GetServiceConfigRequest()
    assert isinstance(response, service_pb2.Service)
    assert response.name == 'name_value'
    assert response.title == 'title_value'
    assert response.producer_project_id == 'producer_project_id_value'
    assert response.id == 'id_value'

def test_get_service_config_empty_call():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_service_config), '__call__') as call:
        client.get_service_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.GetServiceConfigRequest()

@pytest.mark.asyncio
async def test_get_service_config_async(transport: str='grpc_asyncio', request_type=servicemanager.GetServiceConfigRequest):
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_service_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_pb2.Service(name='name_value', title='title_value', producer_project_id='producer_project_id_value', id='id_value'))
        response = await client.get_service_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.GetServiceConfigRequest()
    assert isinstance(response, service_pb2.Service)
    assert response.name == 'name_value'
    assert response.title == 'title_value'
    assert response.producer_project_id == 'producer_project_id_value'
    assert response.id == 'id_value'

@pytest.mark.asyncio
async def test_get_service_config_async_from_dict():
    await test_get_service_config_async(request_type=dict)

def test_get_service_config_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.GetServiceConfigRequest()
    request.service_name = 'service_name_value'
    request.config_id = 'config_id_value'
    with mock.patch.object(type(client.transport.get_service_config), '__call__') as call:
        call.return_value = service_pb2.Service()
        client.get_service_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value&config_id=config_id_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_service_config_field_headers_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.GetServiceConfigRequest()
    request.service_name = 'service_name_value'
    request.config_id = 'config_id_value'
    with mock.patch.object(type(client.transport.get_service_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_pb2.Service())
        await client.get_service_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value&config_id=config_id_value') in kw['metadata']

def test_get_service_config_flattened():
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_service_config), '__call__') as call:
        call.return_value = service_pb2.Service()
        client.get_service_config(service_name='service_name_value', config_id='config_id_value', view=servicemanager.GetServiceConfigRequest.ConfigView.FULL)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val
        arg = args[0].config_id
        mock_val = 'config_id_value'
        assert arg == mock_val
        arg = args[0].view
        mock_val = servicemanager.GetServiceConfigRequest.ConfigView.FULL
        assert arg == mock_val

def test_get_service_config_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_service_config(servicemanager.GetServiceConfigRequest(), service_name='service_name_value', config_id='config_id_value', view=servicemanager.GetServiceConfigRequest.ConfigView.FULL)

@pytest.mark.asyncio
async def test_get_service_config_flattened_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_service_config), '__call__') as call:
        call.return_value = service_pb2.Service()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_pb2.Service())
        response = await client.get_service_config(service_name='service_name_value', config_id='config_id_value', view=servicemanager.GetServiceConfigRequest.ConfigView.FULL)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val
        arg = args[0].config_id
        mock_val = 'config_id_value'
        assert arg == mock_val
        arg = args[0].view
        mock_val = servicemanager.GetServiceConfigRequest.ConfigView.FULL
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_service_config_flattened_error_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_service_config(servicemanager.GetServiceConfigRequest(), service_name='service_name_value', config_id='config_id_value', view=servicemanager.GetServiceConfigRequest.ConfigView.FULL)

@pytest.mark.parametrize('request_type', [servicemanager.CreateServiceConfigRequest, dict])
def test_create_service_config(request_type, transport: str='grpc'):
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_service_config), '__call__') as call:
        call.return_value = service_pb2.Service(name='name_value', title='title_value', producer_project_id='producer_project_id_value', id='id_value')
        response = client.create_service_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.CreateServiceConfigRequest()
    assert isinstance(response, service_pb2.Service)
    assert response.name == 'name_value'
    assert response.title == 'title_value'
    assert response.producer_project_id == 'producer_project_id_value'
    assert response.id == 'id_value'

def test_create_service_config_empty_call():
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_service_config), '__call__') as call:
        client.create_service_config()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.CreateServiceConfigRequest()

@pytest.mark.asyncio
async def test_create_service_config_async(transport: str='grpc_asyncio', request_type=servicemanager.CreateServiceConfigRequest):
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_service_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_pb2.Service(name='name_value', title='title_value', producer_project_id='producer_project_id_value', id='id_value'))
        response = await client.create_service_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.CreateServiceConfigRequest()
    assert isinstance(response, service_pb2.Service)
    assert response.name == 'name_value'
    assert response.title == 'title_value'
    assert response.producer_project_id == 'producer_project_id_value'
    assert response.id == 'id_value'

@pytest.mark.asyncio
async def test_create_service_config_async_from_dict():
    await test_create_service_config_async(request_type=dict)

def test_create_service_config_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.CreateServiceConfigRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.create_service_config), '__call__') as call:
        call.return_value = service_pb2.Service()
        client.create_service_config(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_service_config_field_headers_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.CreateServiceConfigRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.create_service_config), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_pb2.Service())
        await client.create_service_config(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

def test_create_service_config_flattened():
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_service_config), '__call__') as call:
        call.return_value = service_pb2.Service()
        client.create_service_config(service_name='service_name_value', service_config=service_pb2.Service(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val
        arg = args[0].service_config
        mock_val = service_pb2.Service(name='name_value')
        assert arg == mock_val

def test_create_service_config_flattened_error():
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_service_config(servicemanager.CreateServiceConfigRequest(), service_name='service_name_value', service_config=service_pb2.Service(name='name_value'))

@pytest.mark.asyncio
async def test_create_service_config_flattened_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_service_config), '__call__') as call:
        call.return_value = service_pb2.Service()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(service_pb2.Service())
        response = await client.create_service_config(service_name='service_name_value', service_config=service_pb2.Service(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val
        arg = args[0].service_config
        mock_val = service_pb2.Service(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_service_config_flattened_error_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_service_config(servicemanager.CreateServiceConfigRequest(), service_name='service_name_value', service_config=service_pb2.Service(name='name_value'))

@pytest.mark.parametrize('request_type', [servicemanager.SubmitConfigSourceRequest, dict])
def test_submit_config_source(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.submit_config_source), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.submit_config_source(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.SubmitConfigSourceRequest()
    assert isinstance(response, future.Future)

def test_submit_config_source_empty_call():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.submit_config_source), '__call__') as call:
        client.submit_config_source()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.SubmitConfigSourceRequest()

@pytest.mark.asyncio
async def test_submit_config_source_async(transport: str='grpc_asyncio', request_type=servicemanager.SubmitConfigSourceRequest):
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.submit_config_source), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.submit_config_source(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.SubmitConfigSourceRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_submit_config_source_async_from_dict():
    await test_submit_config_source_async(request_type=dict)

def test_submit_config_source_field_headers():
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.SubmitConfigSourceRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.submit_config_source), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.submit_config_source(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_submit_config_source_field_headers_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.SubmitConfigSourceRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.submit_config_source), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.submit_config_source(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

def test_submit_config_source_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.submit_config_source), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.submit_config_source(service_name='service_name_value', config_source=resources.ConfigSource(id='id_value'), validate_only=True)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val
        arg = args[0].config_source
        mock_val = resources.ConfigSource(id='id_value')
        assert arg == mock_val
        arg = args[0].validate_only
        mock_val = True
        assert arg == mock_val

def test_submit_config_source_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.submit_config_source(servicemanager.SubmitConfigSourceRequest(), service_name='service_name_value', config_source=resources.ConfigSource(id='id_value'), validate_only=True)

@pytest.mark.asyncio
async def test_submit_config_source_flattened_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.submit_config_source), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.submit_config_source(service_name='service_name_value', config_source=resources.ConfigSource(id='id_value'), validate_only=True)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val
        arg = args[0].config_source
        mock_val = resources.ConfigSource(id='id_value')
        assert arg == mock_val
        arg = args[0].validate_only
        mock_val = True
        assert arg == mock_val

@pytest.mark.asyncio
async def test_submit_config_source_flattened_error_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.submit_config_source(servicemanager.SubmitConfigSourceRequest(), service_name='service_name_value', config_source=resources.ConfigSource(id='id_value'), validate_only=True)

@pytest.mark.parametrize('request_type', [servicemanager.ListServiceRolloutsRequest, dict])
def test_list_service_rollouts(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_service_rollouts), '__call__') as call:
        call.return_value = servicemanager.ListServiceRolloutsResponse(next_page_token='next_page_token_value')
        response = client.list_service_rollouts(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.ListServiceRolloutsRequest()
    assert isinstance(response, pagers.ListServiceRolloutsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_service_rollouts_empty_call():
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_service_rollouts), '__call__') as call:
        client.list_service_rollouts()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.ListServiceRolloutsRequest()

@pytest.mark.asyncio
async def test_list_service_rollouts_async(transport: str='grpc_asyncio', request_type=servicemanager.ListServiceRolloutsRequest):
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_service_rollouts), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(servicemanager.ListServiceRolloutsResponse(next_page_token='next_page_token_value'))
        response = await client.list_service_rollouts(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.ListServiceRolloutsRequest()
    assert isinstance(response, pagers.ListServiceRolloutsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_service_rollouts_async_from_dict():
    await test_list_service_rollouts_async(request_type=dict)

def test_list_service_rollouts_field_headers():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.ListServiceRolloutsRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.list_service_rollouts), '__call__') as call:
        call.return_value = servicemanager.ListServiceRolloutsResponse()
        client.list_service_rollouts(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_service_rollouts_field_headers_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.ListServiceRolloutsRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.list_service_rollouts), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(servicemanager.ListServiceRolloutsResponse())
        await client.list_service_rollouts(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

def test_list_service_rollouts_flattened():
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_service_rollouts), '__call__') as call:
        call.return_value = servicemanager.ListServiceRolloutsResponse()
        client.list_service_rollouts(service_name='service_name_value', filter='filter_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

def test_list_service_rollouts_flattened_error():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_service_rollouts(servicemanager.ListServiceRolloutsRequest(), service_name='service_name_value', filter='filter_value')

@pytest.mark.asyncio
async def test_list_service_rollouts_flattened_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_service_rollouts), '__call__') as call:
        call.return_value = servicemanager.ListServiceRolloutsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(servicemanager.ListServiceRolloutsResponse())
        response = await client.list_service_rollouts(service_name='service_name_value', filter='filter_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val
        arg = args[0].filter
        mock_val = 'filter_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_service_rollouts_flattened_error_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_service_rollouts(servicemanager.ListServiceRolloutsRequest(), service_name='service_name_value', filter='filter_value')

def test_list_service_rollouts_pager(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_service_rollouts), '__call__') as call:
        call.side_effect = (servicemanager.ListServiceRolloutsResponse(rollouts=[resources.Rollout(), resources.Rollout(), resources.Rollout()], next_page_token='abc'), servicemanager.ListServiceRolloutsResponse(rollouts=[], next_page_token='def'), servicemanager.ListServiceRolloutsResponse(rollouts=[resources.Rollout()], next_page_token='ghi'), servicemanager.ListServiceRolloutsResponse(rollouts=[resources.Rollout(), resources.Rollout()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('service_name', ''),)),)
        pager = client.list_service_rollouts(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Rollout) for i in results))

def test_list_service_rollouts_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_service_rollouts), '__call__') as call:
        call.side_effect = (servicemanager.ListServiceRolloutsResponse(rollouts=[resources.Rollout(), resources.Rollout(), resources.Rollout()], next_page_token='abc'), servicemanager.ListServiceRolloutsResponse(rollouts=[], next_page_token='def'), servicemanager.ListServiceRolloutsResponse(rollouts=[resources.Rollout()], next_page_token='ghi'), servicemanager.ListServiceRolloutsResponse(rollouts=[resources.Rollout(), resources.Rollout()]), RuntimeError)
        pages = list(client.list_service_rollouts(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_service_rollouts_async_pager():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_service_rollouts), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (servicemanager.ListServiceRolloutsResponse(rollouts=[resources.Rollout(), resources.Rollout(), resources.Rollout()], next_page_token='abc'), servicemanager.ListServiceRolloutsResponse(rollouts=[], next_page_token='def'), servicemanager.ListServiceRolloutsResponse(rollouts=[resources.Rollout()], next_page_token='ghi'), servicemanager.ListServiceRolloutsResponse(rollouts=[resources.Rollout(), resources.Rollout()]), RuntimeError)
        async_pager = await client.list_service_rollouts(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, resources.Rollout) for i in responses))

@pytest.mark.asyncio
async def test_list_service_rollouts_async_pages():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_service_rollouts), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (servicemanager.ListServiceRolloutsResponse(rollouts=[resources.Rollout(), resources.Rollout(), resources.Rollout()], next_page_token='abc'), servicemanager.ListServiceRolloutsResponse(rollouts=[], next_page_token='def'), servicemanager.ListServiceRolloutsResponse(rollouts=[resources.Rollout()], next_page_token='ghi'), servicemanager.ListServiceRolloutsResponse(rollouts=[resources.Rollout(), resources.Rollout()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_service_rollouts(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [servicemanager.GetServiceRolloutRequest, dict])
def test_get_service_rollout(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_service_rollout), '__call__') as call:
        call.return_value = resources.Rollout(rollout_id='rollout_id_value', created_by='created_by_value', status=resources.Rollout.RolloutStatus.IN_PROGRESS, service_name='service_name_value')
        response = client.get_service_rollout(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.GetServiceRolloutRequest()
    assert isinstance(response, resources.Rollout)
    assert response.rollout_id == 'rollout_id_value'
    assert response.created_by == 'created_by_value'
    assert response.status == resources.Rollout.RolloutStatus.IN_PROGRESS
    assert response.service_name == 'service_name_value'

def test_get_service_rollout_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_service_rollout), '__call__') as call:
        client.get_service_rollout()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.GetServiceRolloutRequest()

@pytest.mark.asyncio
async def test_get_service_rollout_async(transport: str='grpc_asyncio', request_type=servicemanager.GetServiceRolloutRequest):
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_service_rollout), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Rollout(rollout_id='rollout_id_value', created_by='created_by_value', status=resources.Rollout.RolloutStatus.IN_PROGRESS, service_name='service_name_value'))
        response = await client.get_service_rollout(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.GetServiceRolloutRequest()
    assert isinstance(response, resources.Rollout)
    assert response.rollout_id == 'rollout_id_value'
    assert response.created_by == 'created_by_value'
    assert response.status == resources.Rollout.RolloutStatus.IN_PROGRESS
    assert response.service_name == 'service_name_value'

@pytest.mark.asyncio
async def test_get_service_rollout_async_from_dict():
    await test_get_service_rollout_async(request_type=dict)

def test_get_service_rollout_field_headers():
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.GetServiceRolloutRequest()
    request.service_name = 'service_name_value'
    request.rollout_id = 'rollout_id_value'
    with mock.patch.object(type(client.transport.get_service_rollout), '__call__') as call:
        call.return_value = resources.Rollout()
        client.get_service_rollout(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value&rollout_id=rollout_id_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_service_rollout_field_headers_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.GetServiceRolloutRequest()
    request.service_name = 'service_name_value'
    request.rollout_id = 'rollout_id_value'
    with mock.patch.object(type(client.transport.get_service_rollout), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Rollout())
        await client.get_service_rollout(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value&rollout_id=rollout_id_value') in kw['metadata']

def test_get_service_rollout_flattened():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_service_rollout), '__call__') as call:
        call.return_value = resources.Rollout()
        client.get_service_rollout(service_name='service_name_value', rollout_id='rollout_id_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val
        arg = args[0].rollout_id
        mock_val = 'rollout_id_value'
        assert arg == mock_val

def test_get_service_rollout_flattened_error():
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_service_rollout(servicemanager.GetServiceRolloutRequest(), service_name='service_name_value', rollout_id='rollout_id_value')

@pytest.mark.asyncio
async def test_get_service_rollout_flattened_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_service_rollout), '__call__') as call:
        call.return_value = resources.Rollout()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(resources.Rollout())
        response = await client.get_service_rollout(service_name='service_name_value', rollout_id='rollout_id_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val
        arg = args[0].rollout_id
        mock_val = 'rollout_id_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_service_rollout_flattened_error_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_service_rollout(servicemanager.GetServiceRolloutRequest(), service_name='service_name_value', rollout_id='rollout_id_value')

@pytest.mark.parametrize('request_type', [servicemanager.CreateServiceRolloutRequest, dict])
def test_create_service_rollout(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_service_rollout), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/spam')
        response = client.create_service_rollout(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.CreateServiceRolloutRequest()
    assert isinstance(response, future.Future)

def test_create_service_rollout_empty_call():
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_service_rollout), '__call__') as call:
        client.create_service_rollout()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.CreateServiceRolloutRequest()

@pytest.mark.asyncio
async def test_create_service_rollout_async(transport: str='grpc_asyncio', request_type=servicemanager.CreateServiceRolloutRequest):
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_service_rollout), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_service_rollout(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.CreateServiceRolloutRequest()
    assert isinstance(response, future.Future)

@pytest.mark.asyncio
async def test_create_service_rollout_async_from_dict():
    await test_create_service_rollout_async(request_type=dict)

def test_create_service_rollout_field_headers():
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.CreateServiceRolloutRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.create_service_rollout), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_service_rollout(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_service_rollout_field_headers_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = servicemanager.CreateServiceRolloutRequest()
    request.service_name = 'service_name_value'
    with mock.patch.object(type(client.transport.create_service_rollout), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/op'))
        await client.create_service_rollout(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'service_name=service_name_value') in kw['metadata']

def test_create_service_rollout_flattened():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_service_rollout), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        client.create_service_rollout(service_name='service_name_value', rollout=resources.Rollout(rollout_id='rollout_id_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val
        arg = args[0].rollout
        mock_val = resources.Rollout(rollout_id='rollout_id_value')
        assert arg == mock_val

def test_create_service_rollout_flattened_error():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_service_rollout(servicemanager.CreateServiceRolloutRequest(), service_name='service_name_value', rollout=resources.Rollout(rollout_id='rollout_id_value'))

@pytest.mark.asyncio
async def test_create_service_rollout_flattened_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_service_rollout), '__call__') as call:
        call.return_value = operations_pb2.Operation(name='operations/op')
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation(name='operations/spam'))
        response = await client.create_service_rollout(service_name='service_name_value', rollout=resources.Rollout(rollout_id='rollout_id_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].service_name
        mock_val = 'service_name_value'
        assert arg == mock_val
        arg = args[0].rollout
        mock_val = resources.Rollout(rollout_id='rollout_id_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_service_rollout_flattened_error_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_service_rollout(servicemanager.CreateServiceRolloutRequest(), service_name='service_name_value', rollout=resources.Rollout(rollout_id='rollout_id_value'))

@pytest.mark.parametrize('request_type', [servicemanager.GenerateConfigReportRequest, dict])
def test_generate_config_report(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_config_report), '__call__') as call:
        call.return_value = servicemanager.GenerateConfigReportResponse(service_name='service_name_value', id='id_value')
        response = client.generate_config_report(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.GenerateConfigReportRequest()
    assert isinstance(response, servicemanager.GenerateConfigReportResponse)
    assert response.service_name == 'service_name_value'
    assert response.id == 'id_value'

def test_generate_config_report_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.generate_config_report), '__call__') as call:
        client.generate_config_report()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.GenerateConfigReportRequest()

@pytest.mark.asyncio
async def test_generate_config_report_async(transport: str='grpc_asyncio', request_type=servicemanager.GenerateConfigReportRequest):
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.generate_config_report), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(servicemanager.GenerateConfigReportResponse(service_name='service_name_value', id='id_value'))
        response = await client.generate_config_report(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == servicemanager.GenerateConfigReportRequest()
    assert isinstance(response, servicemanager.GenerateConfigReportResponse)
    assert response.service_name == 'service_name_value'
    assert response.id == 'id_value'

@pytest.mark.asyncio
async def test_generate_config_report_async_from_dict():
    await test_generate_config_report_async(request_type=dict)

def test_generate_config_report_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.generate_config_report), '__call__') as call:
        call.return_value = servicemanager.GenerateConfigReportResponse()
        client.generate_config_report(new_config=any_pb2.Any(type_url='type.googleapis.com/google.protobuf.Empty'), old_config=any_pb2.Any(type_url='type.googleapis.com/google.protobuf.Empty'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].new_config
        mock_val = any_pb2.Any(type_url='type.googleapis.com/google.protobuf.Empty')
        assert arg == mock_val
        arg = args[0].old_config
        mock_val = any_pb2.Any(type_url='type.googleapis.com/google.protobuf.Empty')
        assert arg == mock_val

def test_generate_config_report_flattened_error():
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.generate_config_report(servicemanager.GenerateConfigReportRequest(), new_config=any_pb2.Any(type_url='type.googleapis.com/google.protobuf.Empty'), old_config=any_pb2.Any(type_url='type.googleapis.com/google.protobuf.Empty'))

@pytest.mark.asyncio
async def test_generate_config_report_flattened_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.generate_config_report), '__call__') as call:
        call.return_value = servicemanager.GenerateConfigReportResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(servicemanager.GenerateConfigReportResponse())
        response = await client.generate_config_report(new_config=any_pb2.Any(type_url='type.googleapis.com/google.protobuf.Empty'), old_config=any_pb2.Any(type_url='type.googleapis.com/google.protobuf.Empty'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].new_config
        mock_val = any_pb2.Any(type_url='type.googleapis.com/google.protobuf.Empty')
        assert arg == mock_val
        arg = args[0].old_config
        mock_val = any_pb2.Any(type_url='type.googleapis.com/google.protobuf.Empty')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_generate_config_report_flattened_error_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.generate_config_report(servicemanager.GenerateConfigReportRequest(), new_config=any_pb2.Any(type_url='type.googleapis.com/google.protobuf.Empty'), old_config=any_pb2.Any(type_url='type.googleapis.com/google.protobuf.Empty'))

@pytest.mark.parametrize('request_type', [servicemanager.ListServicesRequest, dict])
def test_list_services_rest(request_type):
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = servicemanager.ListServicesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = servicemanager.ListServicesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_services(request)
    assert isinstance(response, pagers.ListServicesPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_services_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceManagerRestInterceptor())
    client = ServiceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ServiceManagerRestInterceptor, 'post_list_services') as post, mock.patch.object(transports.ServiceManagerRestInterceptor, 'pre_list_services') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = servicemanager.ListServicesRequest.pb(servicemanager.ListServicesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = servicemanager.ListServicesResponse.to_json(servicemanager.ListServicesResponse())
        request = servicemanager.ListServicesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = servicemanager.ListServicesResponse()
        client.list_services(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_services_rest_bad_request(transport: str='rest', request_type=servicemanager.ListServicesRequest):
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_services(request)

def test_list_services_rest_flattened():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = servicemanager.ListServicesResponse()
        sample_request = {}
        mock_args = dict(producer_project_id='producer_project_id_value', consumer_id='consumer_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = servicemanager.ListServicesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_services(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/services' % client.transport._host, args[1])

def test_list_services_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_services(servicemanager.ListServicesRequest(), producer_project_id='producer_project_id_value', consumer_id='consumer_id_value')

def test_list_services_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (servicemanager.ListServicesResponse(services=[resources.ManagedService(), resources.ManagedService(), resources.ManagedService()], next_page_token='abc'), servicemanager.ListServicesResponse(services=[], next_page_token='def'), servicemanager.ListServicesResponse(services=[resources.ManagedService()], next_page_token='ghi'), servicemanager.ListServicesResponse(services=[resources.ManagedService(), resources.ManagedService()]))
        response = response + response
        response = tuple((servicemanager.ListServicesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {}
        pager = client.list_services(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.ManagedService) for i in results))
        pages = list(client.list_services(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [servicemanager.GetServiceRequest, dict])
def test_get_service_rest(request_type):
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'service_name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.ManagedService(service_name='service_name_value', producer_project_id='producer_project_id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.ManagedService.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_service(request)
    assert isinstance(response, resources.ManagedService)
    assert response.service_name == 'service_name_value'
    assert response.producer_project_id == 'producer_project_id_value'

def test_get_service_rest_required_fields(request_type=servicemanager.GetServiceRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ServiceManagerRestTransport
    request_init = {}
    request_init['service_name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['serviceName'] = 'service_name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'serviceName' in jsonified_request
    assert jsonified_request['serviceName'] == 'service_name_value'
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.ManagedService()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.ManagedService.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_service(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_service_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_service._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('serviceName',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_service_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceManagerRestInterceptor())
    client = ServiceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ServiceManagerRestInterceptor, 'post_get_service') as post, mock.patch.object(transports.ServiceManagerRestInterceptor, 'pre_get_service') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = servicemanager.GetServiceRequest.pb(servicemanager.GetServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.ManagedService.to_json(resources.ManagedService())
        request = servicemanager.GetServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.ManagedService()
        client.get_service(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_service_rest_bad_request(transport: str='rest', request_type=servicemanager.GetServiceRequest):
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'service_name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_service(request)

def test_get_service_rest_flattened():
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.ManagedService()
        sample_request = {'service_name': 'sample1'}
        mock_args = dict(service_name='service_name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.ManagedService.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_service(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/services/{service_name}' % client.transport._host, args[1])

def test_get_service_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_service(servicemanager.GetServiceRequest(), service_name='service_name_value')

def test_get_service_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [servicemanager.CreateServiceRequest, dict])
def test_create_service_rest(request_type):
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request_init['service'] = {'service_name': 'service_name_value', 'producer_project_id': 'producer_project_id_value'}
    test_field = servicemanager.CreateServiceRequest.meta.fields['service']

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
    for (field, value) in request_init['service'].items():
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
                for i in range(0, len(request_init['service'][field])):
                    del request_init['service'][field][i][subfield]
            else:
                del request_init['service'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_service(request)
    assert response.operation.name == 'operations/spam'

def test_create_service_rest_required_fields(request_type=servicemanager.CreateServiceRequest):
    if False:
        return 10
    transport_class = transports.ServiceManagerRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_service(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_service_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_service._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('service',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_service_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceManagerRestInterceptor())
    client = ServiceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ServiceManagerRestInterceptor, 'post_create_service') as post, mock.patch.object(transports.ServiceManagerRestInterceptor, 'pre_create_service') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = servicemanager.CreateServiceRequest.pb(servicemanager.CreateServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = servicemanager.CreateServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_service(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_service_rest_bad_request(transport: str='rest', request_type=servicemanager.CreateServiceRequest):
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_service(request)

def test_create_service_rest_flattened():
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {}
        mock_args = dict(service=resources.ManagedService(service_name='service_name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_service(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/services' % client.transport._host, args[1])

def test_create_service_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_service(servicemanager.CreateServiceRequest(), service=resources.ManagedService(service_name='service_name_value'))

def test_create_service_rest_error():
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [servicemanager.DeleteServiceRequest, dict])
def test_delete_service_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'service_name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_service(request)
    assert response.operation.name == 'operations/spam'

def test_delete_service_rest_required_fields(request_type=servicemanager.DeleteServiceRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ServiceManagerRestTransport
    request_init = {}
    request_init['service_name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['serviceName'] = 'service_name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'serviceName' in jsonified_request
    assert jsonified_request['serviceName'] == 'service_name_value'
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_service(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_service_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_service._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('serviceName',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_service_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceManagerRestInterceptor())
    client = ServiceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ServiceManagerRestInterceptor, 'post_delete_service') as post, mock.patch.object(transports.ServiceManagerRestInterceptor, 'pre_delete_service') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = servicemanager.DeleteServiceRequest.pb(servicemanager.DeleteServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = servicemanager.DeleteServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.delete_service(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_service_rest_bad_request(transport: str='rest', request_type=servicemanager.DeleteServiceRequest):
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'service_name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_service(request)

def test_delete_service_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'service_name': 'sample1'}
        mock_args = dict(service_name='service_name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_service(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/services/{service_name}' % client.transport._host, args[1])

def test_delete_service_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_service(servicemanager.DeleteServiceRequest(), service_name='service_name_value')

def test_delete_service_rest_error():
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [servicemanager.UndeleteServiceRequest, dict])
def test_undelete_service_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'service_name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.undelete_service(request)
    assert response.operation.name == 'operations/spam'

def test_undelete_service_rest_required_fields(request_type=servicemanager.UndeleteServiceRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.ServiceManagerRestTransport
    request_init = {}
    request_init['service_name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).undelete_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['serviceName'] = 'service_name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).undelete_service._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'serviceName' in jsonified_request
    assert jsonified_request['serviceName'] == 'service_name_value'
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = operations_pb2.Operation(name='operations/spam')
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.undelete_service(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_undelete_service_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.undelete_service._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('serviceName',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_undelete_service_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceManagerRestInterceptor())
    client = ServiceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ServiceManagerRestInterceptor, 'post_undelete_service') as post, mock.patch.object(transports.ServiceManagerRestInterceptor, 'pre_undelete_service') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = servicemanager.UndeleteServiceRequest.pb(servicemanager.UndeleteServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = servicemanager.UndeleteServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.undelete_service(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_undelete_service_rest_bad_request(transport: str='rest', request_type=servicemanager.UndeleteServiceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'service_name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.undelete_service(request)

def test_undelete_service_rest_flattened():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'service_name': 'sample1'}
        mock_args = dict(service_name='service_name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.undelete_service(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/services/{service_name}:undelete' % client.transport._host, args[1])

def test_undelete_service_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.undelete_service(servicemanager.UndeleteServiceRequest(), service_name='service_name_value')

def test_undelete_service_rest_error():
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [servicemanager.ListServiceConfigsRequest, dict])
def test_list_service_configs_rest(request_type):
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'service_name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = servicemanager.ListServiceConfigsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = servicemanager.ListServiceConfigsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_service_configs(request)
    assert isinstance(response, pagers.ListServiceConfigsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_service_configs_rest_required_fields(request_type=servicemanager.ListServiceConfigsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ServiceManagerRestTransport
    request_init = {}
    request_init['service_name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_service_configs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['serviceName'] = 'service_name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_service_configs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'serviceName' in jsonified_request
    assert jsonified_request['serviceName'] == 'service_name_value'
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = servicemanager.ListServiceConfigsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = servicemanager.ListServiceConfigsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_service_configs(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_service_configs_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_service_configs._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('serviceName',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_service_configs_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceManagerRestInterceptor())
    client = ServiceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ServiceManagerRestInterceptor, 'post_list_service_configs') as post, mock.patch.object(transports.ServiceManagerRestInterceptor, 'pre_list_service_configs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = servicemanager.ListServiceConfigsRequest.pb(servicemanager.ListServiceConfigsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = servicemanager.ListServiceConfigsResponse.to_json(servicemanager.ListServiceConfigsResponse())
        request = servicemanager.ListServiceConfigsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = servicemanager.ListServiceConfigsResponse()
        client.list_service_configs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_service_configs_rest_bad_request(transport: str='rest', request_type=servicemanager.ListServiceConfigsRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'service_name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_service_configs(request)

def test_list_service_configs_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = servicemanager.ListServiceConfigsResponse()
        sample_request = {'service_name': 'sample1'}
        mock_args = dict(service_name='service_name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = servicemanager.ListServiceConfigsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_service_configs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/services/{service_name}/configs' % client.transport._host, args[1])

def test_list_service_configs_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_service_configs(servicemanager.ListServiceConfigsRequest(), service_name='service_name_value')

def test_list_service_configs_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (servicemanager.ListServiceConfigsResponse(service_configs=[service_pb2.Service(), service_pb2.Service(), service_pb2.Service()], next_page_token='abc'), servicemanager.ListServiceConfigsResponse(service_configs=[], next_page_token='def'), servicemanager.ListServiceConfigsResponse(service_configs=[service_pb2.Service()], next_page_token='ghi'), servicemanager.ListServiceConfigsResponse(service_configs=[service_pb2.Service(), service_pb2.Service()]))
        response = response + response
        response = tuple((servicemanager.ListServiceConfigsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'service_name': 'sample1'}
        pager = client.list_service_configs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, service_pb2.Service) for i in results))
        pages = list(client.list_service_configs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [servicemanager.GetServiceConfigRequest, dict])
def test_get_service_config_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'service_name': 'sample1', 'config_id': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service_pb2.Service(name='name_value', title='title_value', producer_project_id='producer_project_id_value', id='id_value')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_service_config(request)
    assert isinstance(response, service_pb2.Service)
    assert response.name == 'name_value'
    assert response.title == 'title_value'
    assert response.producer_project_id == 'producer_project_id_value'
    assert response.id == 'id_value'

def test_get_service_config_rest_required_fields(request_type=servicemanager.GetServiceConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ServiceManagerRestTransport
    request_init = {}
    request_init['service_name'] = ''
    request_init['config_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_service_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['serviceName'] = 'service_name_value'
    jsonified_request['configId'] = 'config_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_service_config._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('view',))
    jsonified_request.update(unset_fields)
    assert 'serviceName' in jsonified_request
    assert jsonified_request['serviceName'] == 'service_name_value'
    assert 'configId' in jsonified_request
    assert jsonified_request['configId'] == 'config_id_value'
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service_pb2.Service()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_service_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_service_config_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_service_config._get_unset_required_fields({})
    assert set(unset_fields) == set(('view',)) & set(('serviceName', 'configId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_service_config_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceManagerRestInterceptor())
    client = ServiceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ServiceManagerRestInterceptor, 'post_get_service_config') as post, mock.patch.object(transports.ServiceManagerRestInterceptor, 'pre_get_service_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = servicemanager.GetServiceConfigRequest.pb(servicemanager.GetServiceConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(service_pb2.Service())
        request = servicemanager.GetServiceConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service_pb2.Service()
        client.get_service_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_service_config_rest_bad_request(transport: str='rest', request_type=servicemanager.GetServiceConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'service_name': 'sample1', 'config_id': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_service_config(request)

def test_get_service_config_rest_flattened():
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service_pb2.Service()
        sample_request = {'service_name': 'sample1', 'config_id': 'sample2'}
        mock_args = dict(service_name='service_name_value', config_id='config_id_value', view=servicemanager.GetServiceConfigRequest.ConfigView.FULL)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_service_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/services/{service_name}/configs/{config_id}' % client.transport._host, args[1])

def test_get_service_config_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_service_config(servicemanager.GetServiceConfigRequest(), service_name='service_name_value', config_id='config_id_value', view=servicemanager.GetServiceConfigRequest.ConfigView.FULL)

def test_get_service_config_rest_error():
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [servicemanager.CreateServiceConfigRequest, dict])
def test_create_service_config_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'service_name': 'sample1'}
    request_init['service_config'] = {'name': 'name_value', 'title': 'title_value', 'producer_project_id': 'producer_project_id_value', 'id': 'id_value', 'apis': [{'name': 'name_value', 'methods': [{'name': 'name_value', 'request_type_url': 'request_type_url_value', 'request_streaming': True, 'response_type_url': 'response_type_url_value', 'response_streaming': True, 'options': [{'name': 'name_value', 'value': {'type_url': 'type.googleapis.com/google.protobuf.Duration', 'value': b'\x08\x0c\x10\xdb\x07'}}], 'syntax': 1}], 'options': {}, 'version': 'version_value', 'source_context': {'file_name': 'file_name_value'}, 'mixins': [{'name': 'name_value', 'root': 'root_value'}], 'syntax': 1}], 'types': [{'name': 'name_value', 'fields': [{'kind': 1, 'cardinality': 1, 'number': 649, 'name': 'name_value', 'type_url': 'type.googleapis.com/google.protobuf.Empty', 'oneof_index': 1166, 'packed': True, 'options': {}, 'json_name': 'json_name_value', 'default_value': 'default_value_value'}], 'oneofs': ['oneofs_value1', 'oneofs_value2'], 'options': {}, 'source_context': {}, 'syntax': 1, 'edition': 'edition_value'}], 'enums': [{'name': 'name_value', 'enumvalue': [{'name': 'name_value', 'number': 649, 'options': {}}], 'options': {}, 'source_context': {}, 'syntax': 1, 'edition': 'edition_value'}], 'documentation': {'summary': 'summary_value', 'pages': [{'name': 'name_value', 'content': 'content_value', 'subpages': {}}], 'rules': [{'selector': 'selector_value', 'description': 'description_value', 'deprecation_description': 'deprecation_description_value'}], 'documentation_root_url': 'documentation_root_url_value', 'service_root_url': 'service_root_url_value', 'overview': 'overview_value'}, 'backend': {'rules': [{'selector': 'selector_value', 'address': 'address_value', 'deadline': 0.8220000000000001, 'min_deadline': 0.1241, 'operation_deadline': 0.1894, 'path_translation': 1, 'jwt_audience': 'jwt_audience_value', 'disable_auth': True, 'protocol': 'protocol_value', 'overrides_by_request_protocol': {}}]}, 'http': {'rules': [{'selector': 'selector_value', 'get': 'get_value', 'put': 'put_value', 'post': 'post_value', 'delete': 'delete_value', 'patch': 'patch_value', 'custom': {'kind': 'kind_value', 'path': 'path_value'}, 'body': 'body_value', 'response_body': 'response_body_value', 'additional_bindings': {}}], 'fully_decode_reserved_expansion': True}, 'quota': {'limits': [{'name': 'name_value', 'description': 'description_value', 'default_limit': 1379, 'max_limit': 964, 'free_tier': 949, 'duration': 'duration_value', 'metric': 'metric_value', 'unit': 'unit_value', 'values': {}, 'display_name': 'display_name_value'}], 'metric_rules': [{'selector': 'selector_value', 'metric_costs': {}}]}, 'authentication': {'rules': [{'selector': 'selector_value', 'oauth': {'canonical_scopes': 'canonical_scopes_value'}, 'allow_without_credential': True, 'requirements': [{'provider_id': 'provider_id_value', 'audiences': 'audiences_value'}]}], 'providers': [{'id': 'id_value', 'issuer': 'issuer_value', 'jwks_uri': 'jwks_uri_value', 'audiences': 'audiences_value', 'authorization_url': 'authorization_url_value', 'jwt_locations': [{'header': 'header_value', 'query': 'query_value', 'cookie': 'cookie_value', 'value_prefix': 'value_prefix_value'}]}]}, 'context': {'rules': [{'selector': 'selector_value', 'requested': ['requested_value1', 'requested_value2'], 'provided': ['provided_value1', 'provided_value2'], 'allowed_request_extensions': ['allowed_request_extensions_value1', 'allowed_request_extensions_value2'], 'allowed_response_extensions': ['allowed_response_extensions_value1', 'allowed_response_extensions_value2']}]}, 'usage': {'requirements': ['requirements_value1', 'requirements_value2'], 'rules': [{'selector': 'selector_value', 'allow_unregistered_calls': True, 'skip_service_control': True}], 'producer_notification_channel': 'producer_notification_channel_value'}, 'endpoints': [{'name': 'name_value', 'aliases': ['aliases_value1', 'aliases_value2'], 'target': 'target_value', 'allow_cors': True}], 'control': {'environment': 'environment_value', 'method_policies': [{'selector': 'selector_value', 'request_policies': [{'selector': 'selector_value', 'resource_permission': 'resource_permission_value', 'resource_type': 'resource_type_value'}]}]}, 'logs': [{'name': 'name_value', 'labels': [{'key': 'key_value', 'value_type': 1, 'description': 'description_value'}], 'description': 'description_value', 'display_name': 'display_name_value'}], 'metrics': [{'name': 'name_value', 'type': 'type_value', 'labels': {}, 'metric_kind': 1, 'value_type': 1, 'unit': 'unit_value', 'description': 'description_value', 'display_name': 'display_name_value', 'metadata': {'launch_stage': 6, 'sample_period': {'seconds': 751, 'nanos': 543}, 'ingest_delay': {}}, 'launch_stage': 6, 'monitored_resource_types': ['monitored_resource_types_value1', 'monitored_resource_types_value2']}], 'monitored_resources': [{'name': 'name_value', 'type': 'type_value', 'display_name': 'display_name_value', 'description': 'description_value', 'labels': {}, 'launch_stage': 6}], 'billing': {'consumer_destinations': [{'monitored_resource': 'monitored_resource_value', 'metrics': ['metrics_value1', 'metrics_value2']}]}, 'logging': {'producer_destinations': [{'monitored_resource': 'monitored_resource_value', 'logs': ['logs_value1', 'logs_value2']}], 'consumer_destinations': {}}, 'monitoring': {'producer_destinations': [{'monitored_resource': 'monitored_resource_value', 'metrics': ['metrics_value1', 'metrics_value2']}], 'consumer_destinations': {}}, 'system_parameters': {'rules': [{'selector': 'selector_value', 'parameters': [{'name': 'name_value', 'http_header': 'http_header_value', 'url_query_parameter': 'url_query_parameter_value'}]}]}, 'source_info': {'source_files': {}}, 'publishing': {'method_settings': [{'selector': 'selector_value', 'long_running': {'initial_poll_delay': {}, 'poll_delay_multiplier': 0.22510000000000002, 'max_poll_delay': {}, 'total_poll_timeout': {}}}], 'new_issue_uri': 'new_issue_uri_value', 'documentation_uri': 'documentation_uri_value', 'api_short_name': 'api_short_name_value', 'github_label': 'github_label_value', 'codeowner_github_teams': ['codeowner_github_teams_value1', 'codeowner_github_teams_value2'], 'doc_tag_prefix': 'doc_tag_prefix_value', 'organization': 1, 'library_settings': [{'version': 'version_value', 'launch_stage': 6, 'rest_numeric_enums': True, 'java_settings': {'library_package': 'library_package_value', 'service_class_names': {}, 'common': {'reference_docs_uri': 'reference_docs_uri_value', 'destinations': [10]}}, 'cpp_settings': {'common': {}}, 'php_settings': {'common': {}}, 'python_settings': {'common': {}}, 'node_settings': {'common': {}}, 'dotnet_settings': {'common': {}, 'renamed_services': {}, 'renamed_resources': {}, 'ignored_resources': ['ignored_resources_value1', 'ignored_resources_value2'], 'forced_namespace_aliases': ['forced_namespace_aliases_value1', 'forced_namespace_aliases_value2'], 'handwritten_signatures': ['handwritten_signatures_value1', 'handwritten_signatures_value2']}, 'ruby_settings': {'common': {}}, 'go_settings': {'common': {}}}], 'proto_reference_documentation_uri': 'proto_reference_documentation_uri_value'}, 'config_version': {'value': 541}}
    test_field = servicemanager.CreateServiceConfigRequest.meta.fields['service_config']

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
    for (field, value) in request_init['service_config'].items():
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
                for i in range(0, len(request_init['service_config'][field])):
                    del request_init['service_config'][field][i][subfield]
            else:
                del request_init['service_config'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service_pb2.Service(name='name_value', title='title_value', producer_project_id='producer_project_id_value', id='id_value')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_service_config(request)
    assert isinstance(response, service_pb2.Service)
    assert response.name == 'name_value'
    assert response.title == 'title_value'
    assert response.producer_project_id == 'producer_project_id_value'
    assert response.id == 'id_value'

def test_create_service_config_rest_required_fields(request_type=servicemanager.CreateServiceConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.ServiceManagerRestTransport
    request_init = {}
    request_init['service_name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_service_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['serviceName'] = 'service_name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_service_config._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'serviceName' in jsonified_request
    assert jsonified_request['serviceName'] == 'service_name_value'
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = service_pb2.Service()
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
            response = client.create_service_config(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_service_config_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_service_config._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('serviceName', 'serviceConfig'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_service_config_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceManagerRestInterceptor())
    client = ServiceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ServiceManagerRestInterceptor, 'post_create_service_config') as post, mock.patch.object(transports.ServiceManagerRestInterceptor, 'pre_create_service_config') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = servicemanager.CreateServiceConfigRequest.pb(servicemanager.CreateServiceConfigRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(service_pb2.Service())
        request = servicemanager.CreateServiceConfigRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = service_pb2.Service()
        client.create_service_config(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_service_config_rest_bad_request(transport: str='rest', request_type=servicemanager.CreateServiceConfigRequest):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'service_name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_service_config(request)

def test_create_service_config_rest_flattened():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = service_pb2.Service()
        sample_request = {'service_name': 'sample1'}
        mock_args = dict(service_name='service_name_value', service_config=service_pb2.Service(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_service_config(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/services/{service_name}/configs' % client.transport._host, args[1])

def test_create_service_config_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_service_config(servicemanager.CreateServiceConfigRequest(), service_name='service_name_value', service_config=service_pb2.Service(name='name_value'))

def test_create_service_config_rest_error():
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [servicemanager.SubmitConfigSourceRequest, dict])
def test_submit_config_source_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'service_name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.submit_config_source(request)
    assert response.operation.name == 'operations/spam'

def test_submit_config_source_rest_required_fields(request_type=servicemanager.SubmitConfigSourceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ServiceManagerRestTransport
    request_init = {}
    request_init['service_name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).submit_config_source._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['serviceName'] = 'service_name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).submit_config_source._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'serviceName' in jsonified_request
    assert jsonified_request['serviceName'] == 'service_name_value'
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.submit_config_source(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_submit_config_source_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.submit_config_source._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('serviceName', 'configSource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_submit_config_source_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceManagerRestInterceptor())
    client = ServiceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ServiceManagerRestInterceptor, 'post_submit_config_source') as post, mock.patch.object(transports.ServiceManagerRestInterceptor, 'pre_submit_config_source') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = servicemanager.SubmitConfigSourceRequest.pb(servicemanager.SubmitConfigSourceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = servicemanager.SubmitConfigSourceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.submit_config_source(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_submit_config_source_rest_bad_request(transport: str='rest', request_type=servicemanager.SubmitConfigSourceRequest):
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'service_name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.submit_config_source(request)

def test_submit_config_source_rest_flattened():
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'service_name': 'sample1'}
        mock_args = dict(service_name='service_name_value', config_source=resources.ConfigSource(id='id_value'), validate_only=True)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.submit_config_source(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/services/{service_name}/configs:submit' % client.transport._host, args[1])

def test_submit_config_source_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.submit_config_source(servicemanager.SubmitConfigSourceRequest(), service_name='service_name_value', config_source=resources.ConfigSource(id='id_value'), validate_only=True)

def test_submit_config_source_rest_error():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [servicemanager.ListServiceRolloutsRequest, dict])
def test_list_service_rollouts_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'service_name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = servicemanager.ListServiceRolloutsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = servicemanager.ListServiceRolloutsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_service_rollouts(request)
    assert isinstance(response, pagers.ListServiceRolloutsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_service_rollouts_rest_required_fields(request_type=servicemanager.ListServiceRolloutsRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ServiceManagerRestTransport
    request_init = {}
    request_init['service_name'] = ''
    request_init['filter'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'filter' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_service_rollouts._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'filter' in jsonified_request
    assert jsonified_request['filter'] == request_init['filter']
    jsonified_request['serviceName'] = 'service_name_value'
    jsonified_request['filter'] = 'filter_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_service_rollouts._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'serviceName' in jsonified_request
    assert jsonified_request['serviceName'] == 'service_name_value'
    assert 'filter' in jsonified_request
    assert jsonified_request['filter'] == 'filter_value'
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = servicemanager.ListServiceRolloutsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = servicemanager.ListServiceRolloutsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_service_rollouts(request)
            expected_params = [('filter', ''), ('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_service_rollouts_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_service_rollouts._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'pageSize', 'pageToken')) & set(('serviceName', 'filter'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_service_rollouts_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceManagerRestInterceptor())
    client = ServiceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ServiceManagerRestInterceptor, 'post_list_service_rollouts') as post, mock.patch.object(transports.ServiceManagerRestInterceptor, 'pre_list_service_rollouts') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = servicemanager.ListServiceRolloutsRequest.pb(servicemanager.ListServiceRolloutsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = servicemanager.ListServiceRolloutsResponse.to_json(servicemanager.ListServiceRolloutsResponse())
        request = servicemanager.ListServiceRolloutsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = servicemanager.ListServiceRolloutsResponse()
        client.list_service_rollouts(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_service_rollouts_rest_bad_request(transport: str='rest', request_type=servicemanager.ListServiceRolloutsRequest):
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'service_name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_service_rollouts(request)

def test_list_service_rollouts_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = servicemanager.ListServiceRolloutsResponse()
        sample_request = {'service_name': 'sample1'}
        mock_args = dict(service_name='service_name_value', filter='filter_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = servicemanager.ListServiceRolloutsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_service_rollouts(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/services/{service_name}/rollouts' % client.transport._host, args[1])

def test_list_service_rollouts_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_service_rollouts(servicemanager.ListServiceRolloutsRequest(), service_name='service_name_value', filter='filter_value')

def test_list_service_rollouts_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (servicemanager.ListServiceRolloutsResponse(rollouts=[resources.Rollout(), resources.Rollout(), resources.Rollout()], next_page_token='abc'), servicemanager.ListServiceRolloutsResponse(rollouts=[], next_page_token='def'), servicemanager.ListServiceRolloutsResponse(rollouts=[resources.Rollout()], next_page_token='ghi'), servicemanager.ListServiceRolloutsResponse(rollouts=[resources.Rollout(), resources.Rollout()]))
        response = response + response
        response = tuple((servicemanager.ListServiceRolloutsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'service_name': 'sample1'}
        pager = client.list_service_rollouts(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, resources.Rollout) for i in results))
        pages = list(client.list_service_rollouts(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [servicemanager.GetServiceRolloutRequest, dict])
def test_get_service_rollout_rest(request_type):
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'service_name': 'sample1', 'rollout_id': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Rollout(rollout_id='rollout_id_value', created_by='created_by_value', status=resources.Rollout.RolloutStatus.IN_PROGRESS, service_name='service_name_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Rollout.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_service_rollout(request)
    assert isinstance(response, resources.Rollout)
    assert response.rollout_id == 'rollout_id_value'
    assert response.created_by == 'created_by_value'
    assert response.status == resources.Rollout.RolloutStatus.IN_PROGRESS
    assert response.service_name == 'service_name_value'

def test_get_service_rollout_rest_required_fields(request_type=servicemanager.GetServiceRolloutRequest):
    if False:
        print('Hello World!')
    transport_class = transports.ServiceManagerRestTransport
    request_init = {}
    request_init['service_name'] = ''
    request_init['rollout_id'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_service_rollout._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['serviceName'] = 'service_name_value'
    jsonified_request['rolloutId'] = 'rollout_id_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_service_rollout._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'serviceName' in jsonified_request
    assert jsonified_request['serviceName'] == 'service_name_value'
    assert 'rolloutId' in jsonified_request
    assert jsonified_request['rolloutId'] == 'rollout_id_value'
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = resources.Rollout()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = resources.Rollout.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_service_rollout(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_service_rollout_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_service_rollout._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('serviceName', 'rolloutId'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_service_rollout_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceManagerRestInterceptor())
    client = ServiceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ServiceManagerRestInterceptor, 'post_get_service_rollout') as post, mock.patch.object(transports.ServiceManagerRestInterceptor, 'pre_get_service_rollout') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = servicemanager.GetServiceRolloutRequest.pb(servicemanager.GetServiceRolloutRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = resources.Rollout.to_json(resources.Rollout())
        request = servicemanager.GetServiceRolloutRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = resources.Rollout()
        client.get_service_rollout(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_service_rollout_rest_bad_request(transport: str='rest', request_type=servicemanager.GetServiceRolloutRequest):
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'service_name': 'sample1', 'rollout_id': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_service_rollout(request)

def test_get_service_rollout_rest_flattened():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = resources.Rollout()
        sample_request = {'service_name': 'sample1', 'rollout_id': 'sample2'}
        mock_args = dict(service_name='service_name_value', rollout_id='rollout_id_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = resources.Rollout.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_service_rollout(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/services/{service_name}/rollouts/{rollout_id}' % client.transport._host, args[1])

def test_get_service_rollout_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_service_rollout(servicemanager.GetServiceRolloutRequest(), service_name='service_name_value', rollout_id='rollout_id_value')

def test_get_service_rollout_rest_error():
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [servicemanager.CreateServiceRolloutRequest, dict])
def test_create_service_rollout_rest(request_type):
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'service_name': 'sample1'}
    request_init['rollout'] = {'rollout_id': 'rollout_id_value', 'create_time': {'seconds': 751, 'nanos': 543}, 'created_by': 'created_by_value', 'status': 1, 'traffic_percent_strategy': {'percentages': {}}, 'delete_service_strategy': {}, 'service_name': 'service_name_value'}
    test_field = servicemanager.CreateServiceRolloutRequest.meta.fields['rollout']

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
    for (field, value) in request_init['rollout'].items():
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
                for i in range(0, len(request_init['rollout'][field])):
                    del request_init['rollout'][field][i][subfield]
            else:
                del request_init['rollout'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_service_rollout(request)
    assert response.operation.name == 'operations/spam'

def test_create_service_rollout_rest_required_fields(request_type=servicemanager.CreateServiceRolloutRequest):
    if False:
        return 10
    transport_class = transports.ServiceManagerRestTransport
    request_init = {}
    request_init['service_name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_service_rollout._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['serviceName'] = 'service_name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_service_rollout._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'serviceName' in jsonified_request
    assert jsonified_request['serviceName'] == 'service_name_value'
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_service_rollout(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_service_rollout_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_service_rollout._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('serviceName', 'rollout'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_service_rollout_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceManagerRestInterceptor())
    client = ServiceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(operation.Operation, '_set_result_from_operation'), mock.patch.object(transports.ServiceManagerRestInterceptor, 'post_create_service_rollout') as post, mock.patch.object(transports.ServiceManagerRestInterceptor, 'pre_create_service_rollout') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = servicemanager.CreateServiceRolloutRequest.pb(servicemanager.CreateServiceRolloutRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(operations_pb2.Operation())
        request = servicemanager.CreateServiceRolloutRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = operations_pb2.Operation()
        client.create_service_rollout(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_service_rollout_rest_bad_request(transport: str='rest', request_type=servicemanager.CreateServiceRolloutRequest):
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'service_name': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_service_rollout(request)

def test_create_service_rollout_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = operations_pb2.Operation(name='operations/spam')
        sample_request = {'service_name': 'sample1'}
        mock_args = dict(service_name='service_name_value', rollout=resources.Rollout(rollout_id='rollout_id_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_service_rollout(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/services/{service_name}/rollouts' % client.transport._host, args[1])

def test_create_service_rollout_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_service_rollout(servicemanager.CreateServiceRolloutRequest(), service_name='service_name_value', rollout=resources.Rollout(rollout_id='rollout_id_value'))

def test_create_service_rollout_rest_error():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [servicemanager.GenerateConfigReportRequest, dict])
def test_generate_config_report_rest(request_type):
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = servicemanager.GenerateConfigReportResponse(service_name='service_name_value', id='id_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = servicemanager.GenerateConfigReportResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.generate_config_report(request)
    assert isinstance(response, servicemanager.GenerateConfigReportResponse)
    assert response.service_name == 'service_name_value'
    assert response.id == 'id_value'

def test_generate_config_report_rest_required_fields(request_type=servicemanager.GenerateConfigReportRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.ServiceManagerRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_config_report._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).generate_config_report._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = servicemanager.GenerateConfigReportResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = servicemanager.GenerateConfigReportResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.generate_config_report(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_generate_config_report_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.generate_config_report._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('newConfig',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_generate_config_report_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.ServiceManagerRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.ServiceManagerRestInterceptor())
    client = ServiceManagerClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.ServiceManagerRestInterceptor, 'post_generate_config_report') as post, mock.patch.object(transports.ServiceManagerRestInterceptor, 'pre_generate_config_report') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = servicemanager.GenerateConfigReportRequest.pb(servicemanager.GenerateConfigReportRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = servicemanager.GenerateConfigReportResponse.to_json(servicemanager.GenerateConfigReportResponse())
        request = servicemanager.GenerateConfigReportRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = servicemanager.GenerateConfigReportResponse()
        client.generate_config_report(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_generate_config_report_rest_bad_request(transport: str='rest', request_type=servicemanager.GenerateConfigReportRequest):
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.generate_config_report(request)

def test_generate_config_report_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = servicemanager.GenerateConfigReportResponse()
        sample_request = {}
        mock_args = dict(new_config=any_pb2.Any(type_url='type.googleapis.com/google.protobuf.Empty'), old_config=any_pb2.Any(type_url='type.googleapis.com/google.protobuf.Empty'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = servicemanager.GenerateConfigReportResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.generate_config_report(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/services:generateConfigReport' % client.transport._host, args[1])

def test_generate_config_report_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.generate_config_report(servicemanager.GenerateConfigReportRequest(), new_config=any_pb2.Any(type_url='type.googleapis.com/google.protobuf.Empty'), old_config=any_pb2.Any(type_url='type.googleapis.com/google.protobuf.Empty'))

def test_generate_config_report_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.ServiceManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.ServiceManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ServiceManagerClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.ServiceManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ServiceManagerClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = ServiceManagerClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.ServiceManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = ServiceManagerClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        return 10
    transport = transports.ServiceManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = ServiceManagerClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        i = 10
        return i + 15
    transport = transports.ServiceManagerGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.ServiceManagerGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.ServiceManagerGrpcTransport, transports.ServiceManagerGrpcAsyncIOTransport, transports.ServiceManagerRestTransport])
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
    transport = ServiceManagerClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.ServiceManagerGrpcTransport)

def test_service_manager_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.ServiceManagerTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_service_manager_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.servicemanagement_v1.services.service_manager.transports.ServiceManagerTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.ServiceManagerTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('list_services', 'get_service', 'create_service', 'delete_service', 'undelete_service', 'list_service_configs', 'get_service_config', 'create_service_config', 'submit_config_source', 'list_service_rollouts', 'get_service_rollout', 'create_service_rollout', 'generate_config_report', 'set_iam_policy', 'get_iam_policy', 'test_iam_permissions', 'list_operations')
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

def test_service_manager_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.servicemanagement_v1.services.service_manager.transports.ServiceManagerTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ServiceManagerTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only', 'https://www.googleapis.com/auth/service.management', 'https://www.googleapis.com/auth/service.management.readonly'), quota_project_id='octopus')

def test_service_manager_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.servicemanagement_v1.services.service_manager.transports.ServiceManagerTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.ServiceManagerTransport()
        adc.assert_called_once()

def test_service_manager_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        ServiceManagerClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only', 'https://www.googleapis.com/auth/service.management', 'https://www.googleapis.com/auth/service.management.readonly'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.ServiceManagerGrpcTransport, transports.ServiceManagerGrpcAsyncIOTransport])
def test_service_manager_transport_auth_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only', 'https://www.googleapis.com/auth/service.management', 'https://www.googleapis.com/auth/service.management.readonly'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.ServiceManagerGrpcTransport, transports.ServiceManagerGrpcAsyncIOTransport, transports.ServiceManagerRestTransport])
def test_service_manager_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.ServiceManagerGrpcTransport, grpc_helpers), (transports.ServiceManagerGrpcAsyncIOTransport, grpc_helpers_async)])
def test_service_manager_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('servicemanagement.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-platform.read-only', 'https://www.googleapis.com/auth/service.management', 'https://www.googleapis.com/auth/service.management.readonly'), scopes=['1', '2'], default_host='servicemanagement.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.ServiceManagerGrpcTransport, transports.ServiceManagerGrpcAsyncIOTransport])
def test_service_manager_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_service_manager_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.ServiceManagerRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

def test_service_manager_rest_lro_client():
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.AbstractOperationsClient)
    assert transport.operations_client is transport.operations_client

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_service_manager_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='servicemanagement.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('servicemanagement.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://servicemanagement.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_service_manager_host_with_port(transport_name):
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='servicemanagement.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('servicemanagement.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://servicemanagement.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_service_manager_client_transport_session_collision(transport_name):
    if False:
        while True:
            i = 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = ServiceManagerClient(credentials=creds1, transport=transport_name)
    client2 = ServiceManagerClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.list_services._session
    session2 = client2.transport.list_services._session
    assert session1 != session2
    session1 = client1.transport.get_service._session
    session2 = client2.transport.get_service._session
    assert session1 != session2
    session1 = client1.transport.create_service._session
    session2 = client2.transport.create_service._session
    assert session1 != session2
    session1 = client1.transport.delete_service._session
    session2 = client2.transport.delete_service._session
    assert session1 != session2
    session1 = client1.transport.undelete_service._session
    session2 = client2.transport.undelete_service._session
    assert session1 != session2
    session1 = client1.transport.list_service_configs._session
    session2 = client2.transport.list_service_configs._session
    assert session1 != session2
    session1 = client1.transport.get_service_config._session
    session2 = client2.transport.get_service_config._session
    assert session1 != session2
    session1 = client1.transport.create_service_config._session
    session2 = client2.transport.create_service_config._session
    assert session1 != session2
    session1 = client1.transport.submit_config_source._session
    session2 = client2.transport.submit_config_source._session
    assert session1 != session2
    session1 = client1.transport.list_service_rollouts._session
    session2 = client2.transport.list_service_rollouts._session
    assert session1 != session2
    session1 = client1.transport.get_service_rollout._session
    session2 = client2.transport.get_service_rollout._session
    assert session1 != session2
    session1 = client1.transport.create_service_rollout._session
    session2 = client2.transport.create_service_rollout._session
    assert session1 != session2
    session1 = client1.transport.generate_config_report._session
    session2 = client2.transport.generate_config_report._session
    assert session1 != session2

def test_service_manager_grpc_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ServiceManagerGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_service_manager_grpc_asyncio_transport_channel():
    if False:
        return 10
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.ServiceManagerGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.ServiceManagerGrpcTransport, transports.ServiceManagerGrpcAsyncIOTransport])
def test_service_manager_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.ServiceManagerGrpcTransport, transports.ServiceManagerGrpcAsyncIOTransport])
def test_service_manager_transport_channel_mtls_with_adc(transport_class):
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

def test_service_manager_grpc_lro_client():
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsClient)
    assert transport.operations_client is transport.operations_client

def test_service_manager_grpc_lro_async_client():
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    transport = client.transport
    assert isinstance(transport.operations_client, operations_v1.OperationsAsyncClient)
    assert transport.operations_client is transport.operations_client

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = ServiceManagerClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'clam'}
    path = ServiceManagerClient.common_billing_account_path(**expected)
    actual = ServiceManagerClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = ServiceManagerClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'octopus'}
    path = ServiceManagerClient.common_folder_path(**expected)
    actual = ServiceManagerClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = ServiceManagerClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'nudibranch'}
    path = ServiceManagerClient.common_organization_path(**expected)
    actual = ServiceManagerClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        while True:
            i = 10
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = ServiceManagerClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        return 10
    expected = {'project': 'mussel'}
    path = ServiceManagerClient.common_project_path(**expected)
    actual = ServiceManagerClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = ServiceManagerClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = ServiceManagerClient.common_location_path(**expected)
    actual = ServiceManagerClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.ServiceManagerTransport, '_prep_wrapped_messages') as prep:
        client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.ServiceManagerTransport, '_prep_wrapped_messages') as prep:
        transport_class = ServiceManagerClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'services/sample1'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'services/sample1'}
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
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'services/sample1'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'services/sample1'}
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
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({'resource': 'services/sample1'}, request)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'services/sample1'}
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

def test_list_operations_rest_bad_request(transport: str='rest', request_type=operations_pb2.ListOperationsRequest):
    if False:
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    request = json_format.ParseDict({}, request)
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
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
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

def test_list_operations(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = operations_pb2.ListOperationsResponse()
        response = client.list_operations(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_list_operations_from_dict_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_operations), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.ListOperationsResponse())
        response = await client.list_operations(request={'name': 'locations'})
        call.assert_called()

def test_set_iam_policy(transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

@pytest.mark.asyncio
async def test_set_iam_policy_from_dict_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774)})
        call.assert_called()

def test_get_iam_policy(transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_iam_policy_from_dict_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_test_iam_permissions(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

@pytest.mark.asyncio
async def test_test_iam_permissions_from_dict_async():
    client = ServiceManagerAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_transport_close():
    if False:
        return 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = ServiceManagerClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(ServiceManagerClient, transports.ServiceManagerGrpcTransport), (ServiceManagerAsyncClient, transports.ServiceManagerGrpcAsyncIOTransport)])
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