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
from google.iam.v1 import iam_policy_pb2
from google.iam.v1 import options_pb2
from google.iam.v1 import policy_pb2
from google.oauth2 import service_account
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.type import expr_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.billing_v1.services.cloud_billing import CloudBillingAsyncClient, CloudBillingClient, pagers, transports
from google.cloud.billing_v1.types import cloud_billing

def client_cert_source_callback():
    if False:
        while True:
            i = 10
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
    assert CloudBillingClient._get_default_mtls_endpoint(None) is None
    assert CloudBillingClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert CloudBillingClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert CloudBillingClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert CloudBillingClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert CloudBillingClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(CloudBillingClient, 'grpc'), (CloudBillingAsyncClient, 'grpc_asyncio'), (CloudBillingClient, 'rest')])
def test_cloud_billing_client_from_service_account_info(client_class, transport_name):
    if False:
        print('Hello World!')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('cloudbilling.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudbilling.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.CloudBillingGrpcTransport, 'grpc'), (transports.CloudBillingGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.CloudBillingRestTransport, 'rest')])
def test_cloud_billing_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(CloudBillingClient, 'grpc'), (CloudBillingAsyncClient, 'grpc_asyncio'), (CloudBillingClient, 'rest')])
def test_cloud_billing_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('cloudbilling.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudbilling.googleapis.com')

def test_cloud_billing_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = CloudBillingClient.get_transport_class()
    available_transports = [transports.CloudBillingGrpcTransport, transports.CloudBillingRestTransport]
    assert transport in available_transports
    transport = CloudBillingClient.get_transport_class('grpc')
    assert transport == transports.CloudBillingGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CloudBillingClient, transports.CloudBillingGrpcTransport, 'grpc'), (CloudBillingAsyncClient, transports.CloudBillingGrpcAsyncIOTransport, 'grpc_asyncio'), (CloudBillingClient, transports.CloudBillingRestTransport, 'rest')])
@mock.patch.object(CloudBillingClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudBillingClient))
@mock.patch.object(CloudBillingAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudBillingAsyncClient))
def test_cloud_billing_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(CloudBillingClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(CloudBillingClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(CloudBillingClient, transports.CloudBillingGrpcTransport, 'grpc', 'true'), (CloudBillingAsyncClient, transports.CloudBillingGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (CloudBillingClient, transports.CloudBillingGrpcTransport, 'grpc', 'false'), (CloudBillingAsyncClient, transports.CloudBillingGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (CloudBillingClient, transports.CloudBillingRestTransport, 'rest', 'true'), (CloudBillingClient, transports.CloudBillingRestTransport, 'rest', 'false')])
@mock.patch.object(CloudBillingClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudBillingClient))
@mock.patch.object(CloudBillingAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudBillingAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_cloud_billing_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [CloudBillingClient, CloudBillingAsyncClient])
@mock.patch.object(CloudBillingClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudBillingClient))
@mock.patch.object(CloudBillingAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CloudBillingAsyncClient))
def test_cloud_billing_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CloudBillingClient, transports.CloudBillingGrpcTransport, 'grpc'), (CloudBillingAsyncClient, transports.CloudBillingGrpcAsyncIOTransport, 'grpc_asyncio'), (CloudBillingClient, transports.CloudBillingRestTransport, 'rest')])
def test_cloud_billing_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CloudBillingClient, transports.CloudBillingGrpcTransport, 'grpc', grpc_helpers), (CloudBillingAsyncClient, transports.CloudBillingGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (CloudBillingClient, transports.CloudBillingRestTransport, 'rest', None)])
def test_cloud_billing_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_cloud_billing_client_client_options_from_dict():
    if False:
        i = 10
        return i + 15
    with mock.patch('google.cloud.billing_v1.services.cloud_billing.transports.CloudBillingGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = CloudBillingClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CloudBillingClient, transports.CloudBillingGrpcTransport, 'grpc', grpc_helpers), (CloudBillingAsyncClient, transports.CloudBillingGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_cloud_billing_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
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
        create_channel.assert_called_with('cloudbilling.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-billing', 'https://www.googleapis.com/auth/cloud-billing.readonly', 'https://www.googleapis.com/auth/cloud-platform'), scopes=None, default_host='cloudbilling.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [cloud_billing.GetBillingAccountRequest, dict])
def test_get_billing_account(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_billing_account), '__call__') as call:
        call.return_value = cloud_billing.BillingAccount(name='name_value', open_=True, display_name='display_name_value', master_billing_account='master_billing_account_value')
        response = client.get_billing_account(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.GetBillingAccountRequest()
    assert isinstance(response, cloud_billing.BillingAccount)
    assert response.name == 'name_value'
    assert response.open_ is True
    assert response.display_name == 'display_name_value'
    assert response.master_billing_account == 'master_billing_account_value'

def test_get_billing_account_empty_call():
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_billing_account), '__call__') as call:
        client.get_billing_account()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.GetBillingAccountRequest()

@pytest.mark.asyncio
async def test_get_billing_account_async(transport: str='grpc_asyncio', request_type=cloud_billing.GetBillingAccountRequest):
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_billing_account), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_billing.BillingAccount(name='name_value', open_=True, display_name='display_name_value', master_billing_account='master_billing_account_value'))
        response = await client.get_billing_account(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.GetBillingAccountRequest()
    assert isinstance(response, cloud_billing.BillingAccount)
    assert response.name == 'name_value'
    assert response.open_ is True
    assert response.display_name == 'display_name_value'
    assert response.master_billing_account == 'master_billing_account_value'

@pytest.mark.asyncio
async def test_get_billing_account_async_from_dict():
    await test_get_billing_account_async(request_type=dict)

def test_get_billing_account_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_billing.GetBillingAccountRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_billing_account), '__call__') as call:
        call.return_value = cloud_billing.BillingAccount()
        client.get_billing_account(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_billing_account_field_headers_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_billing.GetBillingAccountRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_billing_account), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_billing.BillingAccount())
        await client.get_billing_account(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_billing_account_flattened():
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_billing_account), '__call__') as call:
        call.return_value = cloud_billing.BillingAccount()
        client.get_billing_account(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_billing_account_flattened_error():
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_billing_account(cloud_billing.GetBillingAccountRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_billing_account_flattened_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_billing_account), '__call__') as call:
        call.return_value = cloud_billing.BillingAccount()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_billing.BillingAccount())
        response = await client.get_billing_account(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_billing_account_flattened_error_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_billing_account(cloud_billing.GetBillingAccountRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_billing.ListBillingAccountsRequest, dict])
def test_list_billing_accounts(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_billing_accounts), '__call__') as call:
        call.return_value = cloud_billing.ListBillingAccountsResponse(next_page_token='next_page_token_value')
        response = client.list_billing_accounts(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.ListBillingAccountsRequest()
    assert isinstance(response, pagers.ListBillingAccountsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_billing_accounts_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_billing_accounts), '__call__') as call:
        client.list_billing_accounts()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.ListBillingAccountsRequest()

@pytest.mark.asyncio
async def test_list_billing_accounts_async(transport: str='grpc_asyncio', request_type=cloud_billing.ListBillingAccountsRequest):
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_billing_accounts), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_billing.ListBillingAccountsResponse(next_page_token='next_page_token_value'))
        response = await client.list_billing_accounts(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.ListBillingAccountsRequest()
    assert isinstance(response, pagers.ListBillingAccountsAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_billing_accounts_async_from_dict():
    await test_list_billing_accounts_async(request_type=dict)

def test_list_billing_accounts_pager(transport_name: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_billing_accounts), '__call__') as call:
        call.side_effect = (cloud_billing.ListBillingAccountsResponse(billing_accounts=[cloud_billing.BillingAccount(), cloud_billing.BillingAccount(), cloud_billing.BillingAccount()], next_page_token='abc'), cloud_billing.ListBillingAccountsResponse(billing_accounts=[], next_page_token='def'), cloud_billing.ListBillingAccountsResponse(billing_accounts=[cloud_billing.BillingAccount()], next_page_token='ghi'), cloud_billing.ListBillingAccountsResponse(billing_accounts=[cloud_billing.BillingAccount(), cloud_billing.BillingAccount()]), RuntimeError)
        metadata = ()
        pager = client.list_billing_accounts(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_billing.BillingAccount) for i in results))

def test_list_billing_accounts_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_billing_accounts), '__call__') as call:
        call.side_effect = (cloud_billing.ListBillingAccountsResponse(billing_accounts=[cloud_billing.BillingAccount(), cloud_billing.BillingAccount(), cloud_billing.BillingAccount()], next_page_token='abc'), cloud_billing.ListBillingAccountsResponse(billing_accounts=[], next_page_token='def'), cloud_billing.ListBillingAccountsResponse(billing_accounts=[cloud_billing.BillingAccount()], next_page_token='ghi'), cloud_billing.ListBillingAccountsResponse(billing_accounts=[cloud_billing.BillingAccount(), cloud_billing.BillingAccount()]), RuntimeError)
        pages = list(client.list_billing_accounts(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_billing_accounts_async_pager():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_billing_accounts), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_billing.ListBillingAccountsResponse(billing_accounts=[cloud_billing.BillingAccount(), cloud_billing.BillingAccount(), cloud_billing.BillingAccount()], next_page_token='abc'), cloud_billing.ListBillingAccountsResponse(billing_accounts=[], next_page_token='def'), cloud_billing.ListBillingAccountsResponse(billing_accounts=[cloud_billing.BillingAccount()], next_page_token='ghi'), cloud_billing.ListBillingAccountsResponse(billing_accounts=[cloud_billing.BillingAccount(), cloud_billing.BillingAccount()]), RuntimeError)
        async_pager = await client.list_billing_accounts(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, cloud_billing.BillingAccount) for i in responses))

@pytest.mark.asyncio
async def test_list_billing_accounts_async_pages():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_billing_accounts), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_billing.ListBillingAccountsResponse(billing_accounts=[cloud_billing.BillingAccount(), cloud_billing.BillingAccount(), cloud_billing.BillingAccount()], next_page_token='abc'), cloud_billing.ListBillingAccountsResponse(billing_accounts=[], next_page_token='def'), cloud_billing.ListBillingAccountsResponse(billing_accounts=[cloud_billing.BillingAccount()], next_page_token='ghi'), cloud_billing.ListBillingAccountsResponse(billing_accounts=[cloud_billing.BillingAccount(), cloud_billing.BillingAccount()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_billing_accounts(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_billing.UpdateBillingAccountRequest, dict])
def test_update_billing_account(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_billing_account), '__call__') as call:
        call.return_value = cloud_billing.BillingAccount(name='name_value', open_=True, display_name='display_name_value', master_billing_account='master_billing_account_value')
        response = client.update_billing_account(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.UpdateBillingAccountRequest()
    assert isinstance(response, cloud_billing.BillingAccount)
    assert response.name == 'name_value'
    assert response.open_ is True
    assert response.display_name == 'display_name_value'
    assert response.master_billing_account == 'master_billing_account_value'

def test_update_billing_account_empty_call():
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_billing_account), '__call__') as call:
        client.update_billing_account()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.UpdateBillingAccountRequest()

@pytest.mark.asyncio
async def test_update_billing_account_async(transport: str='grpc_asyncio', request_type=cloud_billing.UpdateBillingAccountRequest):
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_billing_account), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_billing.BillingAccount(name='name_value', open_=True, display_name='display_name_value', master_billing_account='master_billing_account_value'))
        response = await client.update_billing_account(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.UpdateBillingAccountRequest()
    assert isinstance(response, cloud_billing.BillingAccount)
    assert response.name == 'name_value'
    assert response.open_ is True
    assert response.display_name == 'display_name_value'
    assert response.master_billing_account == 'master_billing_account_value'

@pytest.mark.asyncio
async def test_update_billing_account_async_from_dict():
    await test_update_billing_account_async(request_type=dict)

def test_update_billing_account_field_headers():
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_billing.UpdateBillingAccountRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_billing_account), '__call__') as call:
        call.return_value = cloud_billing.BillingAccount()
        client.update_billing_account(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_billing_account_field_headers_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_billing.UpdateBillingAccountRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_billing_account), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_billing.BillingAccount())
        await client.update_billing_account(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_update_billing_account_flattened():
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_billing_account), '__call__') as call:
        call.return_value = cloud_billing.BillingAccount()
        client.update_billing_account(name='name_value', account=cloud_billing.BillingAccount(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].account
        mock_val = cloud_billing.BillingAccount(name='name_value')
        assert arg == mock_val

def test_update_billing_account_flattened_error():
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_billing_account(cloud_billing.UpdateBillingAccountRequest(), name='name_value', account=cloud_billing.BillingAccount(name='name_value'))

@pytest.mark.asyncio
async def test_update_billing_account_flattened_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_billing_account), '__call__') as call:
        call.return_value = cloud_billing.BillingAccount()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_billing.BillingAccount())
        response = await client.update_billing_account(name='name_value', account=cloud_billing.BillingAccount(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].account
        mock_val = cloud_billing.BillingAccount(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_billing_account_flattened_error_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_billing_account(cloud_billing.UpdateBillingAccountRequest(), name='name_value', account=cloud_billing.BillingAccount(name='name_value'))

@pytest.mark.parametrize('request_type', [cloud_billing.CreateBillingAccountRequest, dict])
def test_create_billing_account(request_type, transport: str='grpc'):
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_billing_account), '__call__') as call:
        call.return_value = cloud_billing.BillingAccount(name='name_value', open_=True, display_name='display_name_value', master_billing_account='master_billing_account_value')
        response = client.create_billing_account(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.CreateBillingAccountRequest()
    assert isinstance(response, cloud_billing.BillingAccount)
    assert response.name == 'name_value'
    assert response.open_ is True
    assert response.display_name == 'display_name_value'
    assert response.master_billing_account == 'master_billing_account_value'

def test_create_billing_account_empty_call():
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_billing_account), '__call__') as call:
        client.create_billing_account()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.CreateBillingAccountRequest()

@pytest.mark.asyncio
async def test_create_billing_account_async(transport: str='grpc_asyncio', request_type=cloud_billing.CreateBillingAccountRequest):
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_billing_account), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_billing.BillingAccount(name='name_value', open_=True, display_name='display_name_value', master_billing_account='master_billing_account_value'))
        response = await client.create_billing_account(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.CreateBillingAccountRequest()
    assert isinstance(response, cloud_billing.BillingAccount)
    assert response.name == 'name_value'
    assert response.open_ is True
    assert response.display_name == 'display_name_value'
    assert response.master_billing_account == 'master_billing_account_value'

@pytest.mark.asyncio
async def test_create_billing_account_async_from_dict():
    await test_create_billing_account_async(request_type=dict)

def test_create_billing_account_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_billing_account), '__call__') as call:
        call.return_value = cloud_billing.BillingAccount()
        client.create_billing_account(billing_account=cloud_billing.BillingAccount(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].billing_account
        mock_val = cloud_billing.BillingAccount(name='name_value')
        assert arg == mock_val

def test_create_billing_account_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_billing_account(cloud_billing.CreateBillingAccountRequest(), billing_account=cloud_billing.BillingAccount(name='name_value'))

@pytest.mark.asyncio
async def test_create_billing_account_flattened_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_billing_account), '__call__') as call:
        call.return_value = cloud_billing.BillingAccount()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_billing.BillingAccount())
        response = await client.create_billing_account(billing_account=cloud_billing.BillingAccount(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].billing_account
        mock_val = cloud_billing.BillingAccount(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_billing_account_flattened_error_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_billing_account(cloud_billing.CreateBillingAccountRequest(), billing_account=cloud_billing.BillingAccount(name='name_value'))

@pytest.mark.parametrize('request_type', [cloud_billing.ListProjectBillingInfoRequest, dict])
def test_list_project_billing_info(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_project_billing_info), '__call__') as call:
        call.return_value = cloud_billing.ListProjectBillingInfoResponse(next_page_token='next_page_token_value')
        response = client.list_project_billing_info(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.ListProjectBillingInfoRequest()
    assert isinstance(response, pagers.ListProjectBillingInfoPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_project_billing_info_empty_call():
    if False:
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_project_billing_info), '__call__') as call:
        client.list_project_billing_info()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.ListProjectBillingInfoRequest()

@pytest.mark.asyncio
async def test_list_project_billing_info_async(transport: str='grpc_asyncio', request_type=cloud_billing.ListProjectBillingInfoRequest):
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_project_billing_info), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_billing.ListProjectBillingInfoResponse(next_page_token='next_page_token_value'))
        response = await client.list_project_billing_info(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.ListProjectBillingInfoRequest()
    assert isinstance(response, pagers.ListProjectBillingInfoAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_project_billing_info_async_from_dict():
    await test_list_project_billing_info_async(request_type=dict)

def test_list_project_billing_info_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_billing.ListProjectBillingInfoRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.list_project_billing_info), '__call__') as call:
        call.return_value = cloud_billing.ListProjectBillingInfoResponse()
        client.list_project_billing_info(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_project_billing_info_field_headers_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_billing.ListProjectBillingInfoRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.list_project_billing_info), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_billing.ListProjectBillingInfoResponse())
        await client.list_project_billing_info(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_list_project_billing_info_flattened():
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_project_billing_info), '__call__') as call:
        call.return_value = cloud_billing.ListProjectBillingInfoResponse()
        client.list_project_billing_info(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_list_project_billing_info_flattened_error():
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_project_billing_info(cloud_billing.ListProjectBillingInfoRequest(), name='name_value')

@pytest.mark.asyncio
async def test_list_project_billing_info_flattened_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_project_billing_info), '__call__') as call:
        call.return_value = cloud_billing.ListProjectBillingInfoResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_billing.ListProjectBillingInfoResponse())
        response = await client.list_project_billing_info(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_project_billing_info_flattened_error_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_project_billing_info(cloud_billing.ListProjectBillingInfoRequest(), name='name_value')

def test_list_project_billing_info_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_project_billing_info), '__call__') as call:
        call.side_effect = (cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[cloud_billing.ProjectBillingInfo(), cloud_billing.ProjectBillingInfo(), cloud_billing.ProjectBillingInfo()], next_page_token='abc'), cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[], next_page_token='def'), cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[cloud_billing.ProjectBillingInfo()], next_page_token='ghi'), cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[cloud_billing.ProjectBillingInfo(), cloud_billing.ProjectBillingInfo()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('name', ''),)),)
        pager = client.list_project_billing_info(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_billing.ProjectBillingInfo) for i in results))

def test_list_project_billing_info_pages(transport_name: str='grpc'):
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_project_billing_info), '__call__') as call:
        call.side_effect = (cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[cloud_billing.ProjectBillingInfo(), cloud_billing.ProjectBillingInfo(), cloud_billing.ProjectBillingInfo()], next_page_token='abc'), cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[], next_page_token='def'), cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[cloud_billing.ProjectBillingInfo()], next_page_token='ghi'), cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[cloud_billing.ProjectBillingInfo(), cloud_billing.ProjectBillingInfo()]), RuntimeError)
        pages = list(client.list_project_billing_info(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_project_billing_info_async_pager():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_project_billing_info), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[cloud_billing.ProjectBillingInfo(), cloud_billing.ProjectBillingInfo(), cloud_billing.ProjectBillingInfo()], next_page_token='abc'), cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[], next_page_token='def'), cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[cloud_billing.ProjectBillingInfo()], next_page_token='ghi'), cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[cloud_billing.ProjectBillingInfo(), cloud_billing.ProjectBillingInfo()]), RuntimeError)
        async_pager = await client.list_project_billing_info(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, cloud_billing.ProjectBillingInfo) for i in responses))

@pytest.mark.asyncio
async def test_list_project_billing_info_async_pages():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_project_billing_info), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[cloud_billing.ProjectBillingInfo(), cloud_billing.ProjectBillingInfo(), cloud_billing.ProjectBillingInfo()], next_page_token='abc'), cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[], next_page_token='def'), cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[cloud_billing.ProjectBillingInfo()], next_page_token='ghi'), cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[cloud_billing.ProjectBillingInfo(), cloud_billing.ProjectBillingInfo()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_project_billing_info(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_billing.GetProjectBillingInfoRequest, dict])
def test_get_project_billing_info(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_project_billing_info), '__call__') as call:
        call.return_value = cloud_billing.ProjectBillingInfo(name='name_value', project_id='project_id_value', billing_account_name='billing_account_name_value', billing_enabled=True)
        response = client.get_project_billing_info(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.GetProjectBillingInfoRequest()
    assert isinstance(response, cloud_billing.ProjectBillingInfo)
    assert response.name == 'name_value'
    assert response.project_id == 'project_id_value'
    assert response.billing_account_name == 'billing_account_name_value'
    assert response.billing_enabled is True

def test_get_project_billing_info_empty_call():
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_project_billing_info), '__call__') as call:
        client.get_project_billing_info()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.GetProjectBillingInfoRequest()

@pytest.mark.asyncio
async def test_get_project_billing_info_async(transport: str='grpc_asyncio', request_type=cloud_billing.GetProjectBillingInfoRequest):
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_project_billing_info), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_billing.ProjectBillingInfo(name='name_value', project_id='project_id_value', billing_account_name='billing_account_name_value', billing_enabled=True))
        response = await client.get_project_billing_info(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.GetProjectBillingInfoRequest()
    assert isinstance(response, cloud_billing.ProjectBillingInfo)
    assert response.name == 'name_value'
    assert response.project_id == 'project_id_value'
    assert response.billing_account_name == 'billing_account_name_value'
    assert response.billing_enabled is True

@pytest.mark.asyncio
async def test_get_project_billing_info_async_from_dict():
    await test_get_project_billing_info_async(request_type=dict)

def test_get_project_billing_info_field_headers():
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_billing.GetProjectBillingInfoRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_project_billing_info), '__call__') as call:
        call.return_value = cloud_billing.ProjectBillingInfo()
        client.get_project_billing_info(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_project_billing_info_field_headers_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_billing.GetProjectBillingInfoRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_project_billing_info), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_billing.ProjectBillingInfo())
        await client.get_project_billing_info(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_project_billing_info_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_project_billing_info), '__call__') as call:
        call.return_value = cloud_billing.ProjectBillingInfo()
        client.get_project_billing_info(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_project_billing_info_flattened_error():
    if False:
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_project_billing_info(cloud_billing.GetProjectBillingInfoRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_project_billing_info_flattened_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_project_billing_info), '__call__') as call:
        call.return_value = cloud_billing.ProjectBillingInfo()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_billing.ProjectBillingInfo())
        response = await client.get_project_billing_info(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_project_billing_info_flattened_error_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_project_billing_info(cloud_billing.GetProjectBillingInfoRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [cloud_billing.UpdateProjectBillingInfoRequest, dict])
def test_update_project_billing_info(request_type, transport: str='grpc'):
    if False:
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_project_billing_info), '__call__') as call:
        call.return_value = cloud_billing.ProjectBillingInfo(name='name_value', project_id='project_id_value', billing_account_name='billing_account_name_value', billing_enabled=True)
        response = client.update_project_billing_info(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.UpdateProjectBillingInfoRequest()
    assert isinstance(response, cloud_billing.ProjectBillingInfo)
    assert response.name == 'name_value'
    assert response.project_id == 'project_id_value'
    assert response.billing_account_name == 'billing_account_name_value'
    assert response.billing_enabled is True

def test_update_project_billing_info_empty_call():
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_project_billing_info), '__call__') as call:
        client.update_project_billing_info()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.UpdateProjectBillingInfoRequest()

@pytest.mark.asyncio
async def test_update_project_billing_info_async(transport: str='grpc_asyncio', request_type=cloud_billing.UpdateProjectBillingInfoRequest):
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_project_billing_info), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_billing.ProjectBillingInfo(name='name_value', project_id='project_id_value', billing_account_name='billing_account_name_value', billing_enabled=True))
        response = await client.update_project_billing_info(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == cloud_billing.UpdateProjectBillingInfoRequest()
    assert isinstance(response, cloud_billing.ProjectBillingInfo)
    assert response.name == 'name_value'
    assert response.project_id == 'project_id_value'
    assert response.billing_account_name == 'billing_account_name_value'
    assert response.billing_enabled is True

@pytest.mark.asyncio
async def test_update_project_billing_info_async_from_dict():
    await test_update_project_billing_info_async(request_type=dict)

def test_update_project_billing_info_field_headers():
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_billing.UpdateProjectBillingInfoRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_project_billing_info), '__call__') as call:
        call.return_value = cloud_billing.ProjectBillingInfo()
        client.update_project_billing_info(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_project_billing_info_field_headers_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = cloud_billing.UpdateProjectBillingInfoRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.update_project_billing_info), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_billing.ProjectBillingInfo())
        await client.update_project_billing_info(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_update_project_billing_info_flattened():
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_project_billing_info), '__call__') as call:
        call.return_value = cloud_billing.ProjectBillingInfo()
        client.update_project_billing_info(name='name_value', project_billing_info=cloud_billing.ProjectBillingInfo(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].project_billing_info
        mock_val = cloud_billing.ProjectBillingInfo(name='name_value')
        assert arg == mock_val

def test_update_project_billing_info_flattened_error():
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_project_billing_info(cloud_billing.UpdateProjectBillingInfoRequest(), name='name_value', project_billing_info=cloud_billing.ProjectBillingInfo(name='name_value'))

@pytest.mark.asyncio
async def test_update_project_billing_info_flattened_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_project_billing_info), '__call__') as call:
        call.return_value = cloud_billing.ProjectBillingInfo()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(cloud_billing.ProjectBillingInfo())
        response = await client.update_project_billing_info(name='name_value', project_billing_info=cloud_billing.ProjectBillingInfo(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val
        arg = args[0].project_billing_info
        mock_val = cloud_billing.ProjectBillingInfo(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_project_billing_info_flattened_error_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_project_billing_info(cloud_billing.UpdateProjectBillingInfoRequest(), name='name_value', project_billing_info=cloud_billing.ProjectBillingInfo(name='name_value'))

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        client.get_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.GetIamPolicyRequest()

@pytest.mark.asyncio
async def test_get_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.GetIamPolicyRequest):
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.get_iam_policy(request={'resource': 'resource_value', 'options': options_pb2.GetPolicyOptions(requested_policy_version=2598)})
        call.assert_called()

def test_get_iam_policy_flattened():
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.get_iam_policy(resource='resource_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

def test_get_iam_policy_flattened_error():
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_iam_policy(iam_policy_pb2.GetIamPolicyRequest(), resource='resource_value')

@pytest.mark.asyncio
async def test_get_iam_policy_flattened_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.get_iam_policy(resource='resource_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_iam_policy_flattened_error_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_iam_policy(iam_policy_pb2.GetIamPolicyRequest(), resource='resource_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        client.set_iam_policy()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.SetIamPolicyRequest()

@pytest.mark.asyncio
async def test_set_iam_policy_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.SetIamPolicyRequest):
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        for i in range(10):
            print('nop')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        response = client.set_iam_policy(request={'resource': 'resource_value', 'policy': policy_pb2.Policy(version=774), 'update_mask': field_mask_pb2.FieldMask(paths=['paths_value'])})
        call.assert_called()

def test_set_iam_policy_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        client.set_iam_policy(resource='resource_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

def test_set_iam_policy_flattened_error():
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.set_iam_policy(iam_policy_pb2.SetIamPolicyRequest(), resource='resource_value')

@pytest.mark.asyncio
async def test_set_iam_policy_flattened_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.set_iam_policy), '__call__') as call:
        call.return_value = policy_pb2.Policy()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(policy_pb2.Policy())
        response = await client.set_iam_policy(resource='resource_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_set_iam_policy_flattened_error_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.set_iam_policy(iam_policy_pb2.SetIamPolicyRequest(), resource='resource_value')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        client.test_iam_permissions()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == iam_policy_pb2.TestIamPermissionsRequest()

@pytest.mark.asyncio
async def test_test_iam_permissions_async(transport: str='grpc_asyncio', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        response = client.test_iam_permissions(request={'resource': 'resource_value', 'permissions': ['permissions_value']})
        call.assert_called()

def test_test_iam_permissions_flattened():
    if False:
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(resource='resource_value', permissions=['permissions_value'])
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val
        arg = args[0].permissions
        mock_val = ['permissions_value']
        assert arg == mock_val

def test_test_iam_permissions_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.test_iam_permissions(iam_policy_pb2.TestIamPermissionsRequest(), resource='resource_value', permissions=['permissions_value'])

@pytest.mark.asyncio
async def test_test_iam_permissions_flattened_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.test_iam_permissions), '__call__') as call:
        call.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(iam_policy_pb2.TestIamPermissionsResponse())
        response = await client.test_iam_permissions(resource='resource_value', permissions=['permissions_value'])
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].resource
        mock_val = 'resource_value'
        assert arg == mock_val
        arg = args[0].permissions
        mock_val = ['permissions_value']
        assert arg == mock_val

@pytest.mark.asyncio
async def test_test_iam_permissions_flattened_error_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.test_iam_permissions(iam_policy_pb2.TestIamPermissionsRequest(), resource='resource_value', permissions=['permissions_value'])

@pytest.mark.parametrize('request_type', [cloud_billing.GetBillingAccountRequest, dict])
def test_get_billing_account_rest(request_type):
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'billingAccounts/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_billing.BillingAccount(name='name_value', open_=True, display_name='display_name_value', master_billing_account='master_billing_account_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_billing.BillingAccount.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_billing_account(request)
    assert isinstance(response, cloud_billing.BillingAccount)
    assert response.name == 'name_value'
    assert response.open_ is True
    assert response.display_name == 'display_name_value'
    assert response.master_billing_account == 'master_billing_account_value'

def test_get_billing_account_rest_required_fields(request_type=cloud_billing.GetBillingAccountRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudBillingRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_billing_account._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_billing_account._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_billing.BillingAccount()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_billing.BillingAccount.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_billing_account(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_billing_account_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_billing_account._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_billing_account_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudBillingRestInterceptor())
    client = CloudBillingClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudBillingRestInterceptor, 'post_get_billing_account') as post, mock.patch.object(transports.CloudBillingRestInterceptor, 'pre_get_billing_account') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_billing.GetBillingAccountRequest.pb(cloud_billing.GetBillingAccountRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_billing.BillingAccount.to_json(cloud_billing.BillingAccount())
        request = cloud_billing.GetBillingAccountRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_billing.BillingAccount()
        client.get_billing_account(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_billing_account_rest_bad_request(transport: str='rest', request_type=cloud_billing.GetBillingAccountRequest):
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'billingAccounts/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_billing_account(request)

def test_get_billing_account_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_billing.BillingAccount()
        sample_request = {'name': 'billingAccounts/sample1'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_billing.BillingAccount.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_billing_account(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=billingAccounts/*}' % client.transport._host, args[1])

def test_get_billing_account_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_billing_account(cloud_billing.GetBillingAccountRequest(), name='name_value')

def test_get_billing_account_rest_error():
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_billing.ListBillingAccountsRequest, dict])
def test_list_billing_accounts_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_billing.ListBillingAccountsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_billing.ListBillingAccountsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_billing_accounts(request)
    assert isinstance(response, pagers.ListBillingAccountsPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_billing_accounts_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudBillingRestInterceptor())
    client = CloudBillingClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudBillingRestInterceptor, 'post_list_billing_accounts') as post, mock.patch.object(transports.CloudBillingRestInterceptor, 'pre_list_billing_accounts') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_billing.ListBillingAccountsRequest.pb(cloud_billing.ListBillingAccountsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_billing.ListBillingAccountsResponse.to_json(cloud_billing.ListBillingAccountsResponse())
        request = cloud_billing.ListBillingAccountsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_billing.ListBillingAccountsResponse()
        client.list_billing_accounts(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_billing_accounts_rest_bad_request(transport: str='rest', request_type=cloud_billing.ListBillingAccountsRequest):
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_billing_accounts(request)

def test_list_billing_accounts_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (cloud_billing.ListBillingAccountsResponse(billing_accounts=[cloud_billing.BillingAccount(), cloud_billing.BillingAccount(), cloud_billing.BillingAccount()], next_page_token='abc'), cloud_billing.ListBillingAccountsResponse(billing_accounts=[], next_page_token='def'), cloud_billing.ListBillingAccountsResponse(billing_accounts=[cloud_billing.BillingAccount()], next_page_token='ghi'), cloud_billing.ListBillingAccountsResponse(billing_accounts=[cloud_billing.BillingAccount(), cloud_billing.BillingAccount()]))
        response = response + response
        response = tuple((cloud_billing.ListBillingAccountsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {}
        pager = client.list_billing_accounts(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_billing.BillingAccount) for i in results))
        pages = list(client.list_billing_accounts(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_billing.UpdateBillingAccountRequest, dict])
def test_update_billing_account_rest(request_type):
    if False:
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'billingAccounts/sample1'}
    request_init['account'] = {'name': 'name_value', 'open_': True, 'display_name': 'display_name_value', 'master_billing_account': 'master_billing_account_value'}
    test_field = cloud_billing.UpdateBillingAccountRequest.meta.fields['account']

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
    for (field, value) in request_init['account'].items():
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
                for i in range(0, len(request_init['account'][field])):
                    del request_init['account'][field][i][subfield]
            else:
                del request_init['account'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_billing.BillingAccount(name='name_value', open_=True, display_name='display_name_value', master_billing_account='master_billing_account_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_billing.BillingAccount.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_billing_account(request)
    assert isinstance(response, cloud_billing.BillingAccount)
    assert response.name == 'name_value'
    assert response.open_ is True
    assert response.display_name == 'display_name_value'
    assert response.master_billing_account == 'master_billing_account_value'

def test_update_billing_account_rest_required_fields(request_type=cloud_billing.UpdateBillingAccountRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudBillingRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_billing_account._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_billing_account._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('update_mask',))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_billing.BillingAccount()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_billing.BillingAccount.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_billing_account(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_billing_account_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_billing_account._get_unset_required_fields({})
    assert set(unset_fields) == set(('updateMask',)) & set(('name', 'account'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_billing_account_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudBillingRestInterceptor())
    client = CloudBillingClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudBillingRestInterceptor, 'post_update_billing_account') as post, mock.patch.object(transports.CloudBillingRestInterceptor, 'pre_update_billing_account') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_billing.UpdateBillingAccountRequest.pb(cloud_billing.UpdateBillingAccountRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_billing.BillingAccount.to_json(cloud_billing.BillingAccount())
        request = cloud_billing.UpdateBillingAccountRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_billing.BillingAccount()
        client.update_billing_account(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_billing_account_rest_bad_request(transport: str='rest', request_type=cloud_billing.UpdateBillingAccountRequest):
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'billingAccounts/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_billing_account(request)

def test_update_billing_account_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_billing.BillingAccount()
        sample_request = {'name': 'billingAccounts/sample1'}
        mock_args = dict(name='name_value', account=cloud_billing.BillingAccount(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_billing.BillingAccount.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_billing_account(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=billingAccounts/*}' % client.transport._host, args[1])

def test_update_billing_account_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_billing_account(cloud_billing.UpdateBillingAccountRequest(), name='name_value', account=cloud_billing.BillingAccount(name='name_value'))

def test_update_billing_account_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_billing.CreateBillingAccountRequest, dict])
def test_create_billing_account_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {}
    request_init['billing_account'] = {'name': 'name_value', 'open_': True, 'display_name': 'display_name_value', 'master_billing_account': 'master_billing_account_value'}
    test_field = cloud_billing.CreateBillingAccountRequest.meta.fields['billing_account']

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
    for (field, value) in request_init['billing_account'].items():
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
                for i in range(0, len(request_init['billing_account'][field])):
                    del request_init['billing_account'][field][i][subfield]
            else:
                del request_init['billing_account'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_billing.BillingAccount(name='name_value', open_=True, display_name='display_name_value', master_billing_account='master_billing_account_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_billing.BillingAccount.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_billing_account(request)
    assert isinstance(response, cloud_billing.BillingAccount)
    assert response.name == 'name_value'
    assert response.open_ is True
    assert response.display_name == 'display_name_value'
    assert response.master_billing_account == 'master_billing_account_value'

def test_create_billing_account_rest_required_fields(request_type=cloud_billing.CreateBillingAccountRequest):
    if False:
        print('Hello World!')
    transport_class = transports.CloudBillingRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_billing_account._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_billing_account._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_billing.BillingAccount()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_billing.BillingAccount.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_billing_account(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_billing_account_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_billing_account._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('billingAccount',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_billing_account_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudBillingRestInterceptor())
    client = CloudBillingClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudBillingRestInterceptor, 'post_create_billing_account') as post, mock.patch.object(transports.CloudBillingRestInterceptor, 'pre_create_billing_account') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_billing.CreateBillingAccountRequest.pb(cloud_billing.CreateBillingAccountRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_billing.BillingAccount.to_json(cloud_billing.BillingAccount())
        request = cloud_billing.CreateBillingAccountRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_billing.BillingAccount()
        client.create_billing_account(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_billing_account_rest_bad_request(transport: str='rest', request_type=cloud_billing.CreateBillingAccountRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_billing_account(request)

def test_create_billing_account_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_billing.BillingAccount()
        sample_request = {}
        mock_args = dict(billing_account=cloud_billing.BillingAccount(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_billing.BillingAccount.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_billing_account(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/billingAccounts' % client.transport._host, args[1])

def test_create_billing_account_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_billing_account(cloud_billing.CreateBillingAccountRequest(), billing_account=cloud_billing.BillingAccount(name='name_value'))

def test_create_billing_account_rest_error():
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_billing.ListProjectBillingInfoRequest, dict])
def test_list_project_billing_info_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'billingAccounts/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_billing.ListProjectBillingInfoResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_billing.ListProjectBillingInfoResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_project_billing_info(request)
    assert isinstance(response, pagers.ListProjectBillingInfoPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_project_billing_info_rest_required_fields(request_type=cloud_billing.ListProjectBillingInfoRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CloudBillingRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_project_billing_info._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_project_billing_info._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token'))
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_billing.ListProjectBillingInfoResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_billing.ListProjectBillingInfoResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_project_billing_info(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_project_billing_info_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_project_billing_info._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken')) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_project_billing_info_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudBillingRestInterceptor())
    client = CloudBillingClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudBillingRestInterceptor, 'post_list_project_billing_info') as post, mock.patch.object(transports.CloudBillingRestInterceptor, 'pre_list_project_billing_info') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_billing.ListProjectBillingInfoRequest.pb(cloud_billing.ListProjectBillingInfoRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_billing.ListProjectBillingInfoResponse.to_json(cloud_billing.ListProjectBillingInfoResponse())
        request = cloud_billing.ListProjectBillingInfoRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_billing.ListProjectBillingInfoResponse()
        client.list_project_billing_info(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_project_billing_info_rest_bad_request(transport: str='rest', request_type=cloud_billing.ListProjectBillingInfoRequest):
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'billingAccounts/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_project_billing_info(request)

def test_list_project_billing_info_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_billing.ListProjectBillingInfoResponse()
        sample_request = {'name': 'billingAccounts/sample1'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_billing.ListProjectBillingInfoResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_project_billing_info(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=billingAccounts/*}/projects' % client.transport._host, args[1])

def test_list_project_billing_info_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_project_billing_info(cloud_billing.ListProjectBillingInfoRequest(), name='name_value')

def test_list_project_billing_info_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[cloud_billing.ProjectBillingInfo(), cloud_billing.ProjectBillingInfo(), cloud_billing.ProjectBillingInfo()], next_page_token='abc'), cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[], next_page_token='def'), cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[cloud_billing.ProjectBillingInfo()], next_page_token='ghi'), cloud_billing.ListProjectBillingInfoResponse(project_billing_info=[cloud_billing.ProjectBillingInfo(), cloud_billing.ProjectBillingInfo()]))
        response = response + response
        response = tuple((cloud_billing.ListProjectBillingInfoResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'name': 'billingAccounts/sample1'}
        pager = client.list_project_billing_info(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, cloud_billing.ProjectBillingInfo) for i in results))
        pages = list(client.list_project_billing_info(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [cloud_billing.GetProjectBillingInfoRequest, dict])
def test_get_project_billing_info_rest(request_type):
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_billing.ProjectBillingInfo(name='name_value', project_id='project_id_value', billing_account_name='billing_account_name_value', billing_enabled=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_billing.ProjectBillingInfo.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_project_billing_info(request)
    assert isinstance(response, cloud_billing.ProjectBillingInfo)
    assert response.name == 'name_value'
    assert response.project_id == 'project_id_value'
    assert response.billing_account_name == 'billing_account_name_value'
    assert response.billing_enabled is True

def test_get_project_billing_info_rest_required_fields(request_type=cloud_billing.GetProjectBillingInfoRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudBillingRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_project_billing_info._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_project_billing_info._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_billing.ProjectBillingInfo()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_billing.ProjectBillingInfo.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_project_billing_info(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_project_billing_info_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_project_billing_info._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_project_billing_info_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudBillingRestInterceptor())
    client = CloudBillingClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudBillingRestInterceptor, 'post_get_project_billing_info') as post, mock.patch.object(transports.CloudBillingRestInterceptor, 'pre_get_project_billing_info') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_billing.GetProjectBillingInfoRequest.pb(cloud_billing.GetProjectBillingInfoRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_billing.ProjectBillingInfo.to_json(cloud_billing.ProjectBillingInfo())
        request = cloud_billing.GetProjectBillingInfoRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_billing.ProjectBillingInfo()
        client.get_project_billing_info(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_project_billing_info_rest_bad_request(transport: str='rest', request_type=cloud_billing.GetProjectBillingInfoRequest):
    if False:
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_project_billing_info(request)

def test_get_project_billing_info_rest_flattened():
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_billing.ProjectBillingInfo()
        sample_request = {'name': 'projects/sample1'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_billing.ProjectBillingInfo.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_project_billing_info(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*}/billingInfo' % client.transport._host, args[1])

def test_get_project_billing_info_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_project_billing_info(cloud_billing.GetProjectBillingInfoRequest(), name='name_value')

def test_get_project_billing_info_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [cloud_billing.UpdateProjectBillingInfoRequest, dict])
def test_update_project_billing_info_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1'}
    request_init['project_billing_info'] = {'name': 'name_value', 'project_id': 'project_id_value', 'billing_account_name': 'billing_account_name_value', 'billing_enabled': True}
    test_field = cloud_billing.UpdateProjectBillingInfoRequest.meta.fields['project_billing_info']

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
    for (field, value) in request_init['project_billing_info'].items():
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
                for i in range(0, len(request_init['project_billing_info'][field])):
                    del request_init['project_billing_info'][field][i][subfield]
            else:
                del request_init['project_billing_info'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_billing.ProjectBillingInfo(name='name_value', project_id='project_id_value', billing_account_name='billing_account_name_value', billing_enabled=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_billing.ProjectBillingInfo.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_project_billing_info(request)
    assert isinstance(response, cloud_billing.ProjectBillingInfo)
    assert response.name == 'name_value'
    assert response.project_id == 'project_id_value'
    assert response.billing_account_name == 'billing_account_name_value'
    assert response.billing_enabled is True

def test_update_project_billing_info_rest_required_fields(request_type=cloud_billing.UpdateProjectBillingInfoRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudBillingRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_project_billing_info._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_project_billing_info._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = cloud_billing.ProjectBillingInfo()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'put', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = cloud_billing.ProjectBillingInfo.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_project_billing_info(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_project_billing_info_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_project_billing_info._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_project_billing_info_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudBillingRestInterceptor())
    client = CloudBillingClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudBillingRestInterceptor, 'post_update_project_billing_info') as post, mock.patch.object(transports.CloudBillingRestInterceptor, 'pre_update_project_billing_info') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = cloud_billing.UpdateProjectBillingInfoRequest.pb(cloud_billing.UpdateProjectBillingInfoRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = cloud_billing.ProjectBillingInfo.to_json(cloud_billing.ProjectBillingInfo())
        request = cloud_billing.UpdateProjectBillingInfoRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = cloud_billing.ProjectBillingInfo()
        client.update_project_billing_info(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_project_billing_info_rest_bad_request(transport: str='rest', request_type=cloud_billing.UpdateProjectBillingInfoRequest):
    if False:
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_project_billing_info(request)

def test_update_project_billing_info_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = cloud_billing.ProjectBillingInfo()
        sample_request = {'name': 'projects/sample1'}
        mock_args = dict(name='name_value', project_billing_info=cloud_billing.ProjectBillingInfo(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = cloud_billing.ProjectBillingInfo.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_project_billing_info(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{name=projects/*}/billingInfo' % client.transport._host, args[1])

def test_update_project_billing_info_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_project_billing_info(cloud_billing.UpdateProjectBillingInfoRequest(), name='name_value', project_billing_info=cloud_billing.ProjectBillingInfo(name='name_value'))

def test_update_project_billing_info_rest_error():
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.GetIamPolicyRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'billingAccounts/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_get_iam_policy_rest_required_fields(request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudBillingRestTransport
    request_init = {}
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('options',))
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = policy_pb2.Policy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_iam_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_iam_policy_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(('options',)) & set(('resource',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_iam_policy_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudBillingRestInterceptor())
    client = CloudBillingClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudBillingRestInterceptor, 'post_get_iam_policy') as post, mock.patch.object(transports.CloudBillingRestInterceptor, 'pre_get_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.GetIamPolicyRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(policy_pb2.Policy())
        request = iam_policy_pb2.GetIamPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = policy_pb2.Policy()
        client.get_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.GetIamPolicyRequest):
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'billingAccounts/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

def test_get_iam_policy_rest_flattened():
    if False:
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        sample_request = {'resource': 'billingAccounts/sample1'}
        mock_args = dict(resource='resource_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_iam_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{resource=billingAccounts/*}:getIamPolicy' % client.transport._host, args[1])

def test_get_iam_policy_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_iam_policy(iam_policy_pb2.GetIamPolicyRequest(), resource='resource_value')

def test_get_iam_policy_rest_error():
    if False:
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.SetIamPolicyRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'billingAccounts/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy(version=774, etag=b'etag_blob')
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_iam_policy(request)
    assert isinstance(response, policy_pb2.Policy)
    assert response.version == 774
    assert response.etag == b'etag_blob'

def test_set_iam_policy_rest_required_fields(request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.CloudBillingRestTransport
    request_init = {}
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = policy_pb2.Policy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.set_iam_policy(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_iam_policy_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'policy'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_iam_policy_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudBillingRestInterceptor())
    client = CloudBillingClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudBillingRestInterceptor, 'post_set_iam_policy') as post, mock.patch.object(transports.CloudBillingRestInterceptor, 'pre_set_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.SetIamPolicyRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(policy_pb2.Policy())
        request = iam_policy_pb2.SetIamPolicyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = policy_pb2.Policy()
        client.set_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_iam_policy_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.SetIamPolicyRequest):
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'billingAccounts/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

def test_set_iam_policy_rest_flattened():
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = policy_pb2.Policy()
        sample_request = {'resource': 'billingAccounts/sample1'}
        mock_args = dict(resource='resource_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_iam_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{resource=billingAccounts/*}:setIamPolicy' % client.transport._host, args[1])

def test_set_iam_policy_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_iam_policy(iam_policy_pb2.SetIamPolicyRequest(), resource='resource_value')

def test_set_iam_policy_rest_error():
    if False:
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [iam_policy_pb2.TestIamPermissionsRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'resource': 'billingAccounts/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse(permissions=['permissions_value'])
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.test_iam_permissions(request)
    assert isinstance(response, iam_policy_pb2.TestIamPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_rest_required_fields(request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        return 10
    transport_class = transports.CloudBillingRestTransport
    request_init = {}
    request_init['resource'] = ''
    request_init['permissions'] = ''
    request = request_type(**request_init)
    pb_request = request
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['resource'] = 'resource_value'
    jsonified_request['permissions'] = 'permissions_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    assert 'permissions' in jsonified_request
    assert jsonified_request['permissions'] == 'permissions_value'
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = iam_policy_pb2.TestIamPermissionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.test_iam_permissions(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_test_iam_permissions_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.test_iam_permissions._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('resource', 'permissions'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_test_iam_permissions_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.CloudBillingRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CloudBillingRestInterceptor())
    client = CloudBillingClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CloudBillingRestInterceptor, 'post_test_iam_permissions') as post, mock.patch.object(transports.CloudBillingRestInterceptor, 'pre_test_iam_permissions') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = iam_policy_pb2.TestIamPermissionsRequest()
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = json_format.MessageToJson(iam_policy_pb2.TestIamPermissionsResponse())
        request = iam_policy_pb2.TestIamPermissionsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = iam_policy_pb2.TestIamPermissionsResponse()
        client.test_iam_permissions(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_test_iam_permissions_rest_bad_request(transport: str='rest', request_type=iam_policy_pb2.TestIamPermissionsRequest):
    if False:
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'resource': 'billingAccounts/sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

def test_test_iam_permissions_rest_flattened():
    if False:
        print('Hello World!')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = iam_policy_pb2.TestIamPermissionsResponse()
        sample_request = {'resource': 'billingAccounts/sample1'}
        mock_args = dict(resource='resource_value', permissions=['permissions_value'])
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.test_iam_permissions(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v1/{resource=billingAccounts/*}:testIamPermissions' % client.transport._host, args[1])

def test_test_iam_permissions_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.test_iam_permissions(iam_policy_pb2.TestIamPermissionsRequest(), resource='resource_value', permissions=['permissions_value'])

def test_test_iam_permissions_rest_error():
    if False:
        return 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.CloudBillingGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.CloudBillingGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudBillingClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.CloudBillingGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CloudBillingClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CloudBillingClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.CloudBillingGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CloudBillingClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        print('Hello World!')
    transport = transports.CloudBillingGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = CloudBillingClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        print('Hello World!')
    transport = transports.CloudBillingGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.CloudBillingGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.CloudBillingGrpcTransport, transports.CloudBillingGrpcAsyncIOTransport, transports.CloudBillingRestTransport])
def test_transport_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['grpc', 'rest'])
def test_transport_kind(transport_name):
    if False:
        while True:
            i = 10
    transport = CloudBillingClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.CloudBillingGrpcTransport)

def test_cloud_billing_base_transport_error():
    if False:
        i = 10
        return i + 15
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.CloudBillingTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_cloud_billing_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.billing_v1.services.cloud_billing.transports.CloudBillingTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.CloudBillingTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('get_billing_account', 'list_billing_accounts', 'update_billing_account', 'create_billing_account', 'list_project_billing_info', 'get_project_billing_info', 'update_project_billing_info', 'get_iam_policy', 'set_iam_policy', 'test_iam_permissions')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_cloud_billing_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.billing_v1.services.cloud_billing.transports.CloudBillingTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CloudBillingTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-billing', 'https://www.googleapis.com/auth/cloud-billing.readonly', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id='octopus')

def test_cloud_billing_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.billing_v1.services.cloud_billing.transports.CloudBillingTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CloudBillingTransport()
        adc.assert_called_once()

def test_cloud_billing_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        CloudBillingClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-billing', 'https://www.googleapis.com/auth/cloud-billing.readonly', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.CloudBillingGrpcTransport, transports.CloudBillingGrpcAsyncIOTransport])
def test_cloud_billing_transport_auth_adc(transport_class):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-billing', 'https://www.googleapis.com/auth/cloud-billing.readonly', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.CloudBillingGrpcTransport, transports.CloudBillingGrpcAsyncIOTransport, transports.CloudBillingRestTransport])
def test_cloud_billing_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.CloudBillingGrpcTransport, grpc_helpers), (transports.CloudBillingGrpcAsyncIOTransport, grpc_helpers_async)])
def test_cloud_billing_transport_create_channel(transport_class, grpc_helpers):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('cloudbilling.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-billing', 'https://www.googleapis.com/auth/cloud-billing.readonly', 'https://www.googleapis.com/auth/cloud-platform'), scopes=['1', '2'], default_host='cloudbilling.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.CloudBillingGrpcTransport, transports.CloudBillingGrpcAsyncIOTransport])
def test_cloud_billing_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_cloud_billing_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.CloudBillingRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_cloud_billing_host_no_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudbilling.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('cloudbilling.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudbilling.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_cloud_billing_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='cloudbilling.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('cloudbilling.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://cloudbilling.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_cloud_billing_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = CloudBillingClient(credentials=creds1, transport=transport_name)
    client2 = CloudBillingClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.get_billing_account._session
    session2 = client2.transport.get_billing_account._session
    assert session1 != session2
    session1 = client1.transport.list_billing_accounts._session
    session2 = client2.transport.list_billing_accounts._session
    assert session1 != session2
    session1 = client1.transport.update_billing_account._session
    session2 = client2.transport.update_billing_account._session
    assert session1 != session2
    session1 = client1.transport.create_billing_account._session
    session2 = client2.transport.create_billing_account._session
    assert session1 != session2
    session1 = client1.transport.list_project_billing_info._session
    session2 = client2.transport.list_project_billing_info._session
    assert session1 != session2
    session1 = client1.transport.get_project_billing_info._session
    session2 = client2.transport.get_project_billing_info._session
    assert session1 != session2
    session1 = client1.transport.update_project_billing_info._session
    session2 = client2.transport.update_project_billing_info._session
    assert session1 != session2
    session1 = client1.transport.get_iam_policy._session
    session2 = client2.transport.get_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.set_iam_policy._session
    session2 = client2.transport.set_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.test_iam_permissions._session
    session2 = client2.transport.test_iam_permissions._session
    assert session1 != session2

def test_cloud_billing_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CloudBillingGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_cloud_billing_grpc_asyncio_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CloudBillingGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.CloudBillingGrpcTransport, transports.CloudBillingGrpcAsyncIOTransport])
def test_cloud_billing_transport_channel_mtls_with_client_cert_source(transport_class):
    if False:
        while True:
            i = 10
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

@pytest.mark.parametrize('transport_class', [transports.CloudBillingGrpcTransport, transports.CloudBillingGrpcAsyncIOTransport])
def test_cloud_billing_transport_channel_mtls_with_adc(transport_class):
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

def test_project_billing_info_path():
    if False:
        return 10
    project = 'squid'
    expected = 'projects/{project}/billingInfo'.format(project=project)
    actual = CloudBillingClient.project_billing_info_path(project)
    assert expected == actual

def test_parse_project_billing_info_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'clam'}
    path = CloudBillingClient.project_billing_info_path(**expected)
    actual = CloudBillingClient.parse_project_billing_info_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'whelk'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = CloudBillingClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'octopus'}
    path = CloudBillingClient.common_billing_account_path(**expected)
    actual = CloudBillingClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'oyster'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = CloudBillingClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'nudibranch'}
    path = CloudBillingClient.common_folder_path(**expected)
    actual = CloudBillingClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'cuttlefish'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = CloudBillingClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        i = 10
        return i + 15
    expected = {'organization': 'mussel'}
    path = CloudBillingClient.common_organization_path(**expected)
    actual = CloudBillingClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'winkle'
    expected = 'projects/{project}'.format(project=project)
    actual = CloudBillingClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'nautilus'}
    path = CloudBillingClient.common_project_path(**expected)
    actual = CloudBillingClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'scallop'
    location = 'abalone'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = CloudBillingClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'squid', 'location': 'clam'}
    path = CloudBillingClient.common_location_path(**expected)
    actual = CloudBillingClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        return 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.CloudBillingTransport, '_prep_wrapped_messages') as prep:
        client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.CloudBillingTransport, '_prep_wrapped_messages') as prep:
        transport_class = CloudBillingClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = CloudBillingAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = CloudBillingClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(CloudBillingClient, transports.CloudBillingGrpcTransport), (CloudBillingAsyncClient, transports.CloudBillingGrpcAsyncIOTransport)])
def test_api_key_credentials(client_class, transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth._default, 'get_api_key_credentials', create=True) as get_api_key_credentials:
        mock_cred = mock.Mock()
        get_api_key_credentials.return_value = mock_cred
        options = client_options.ClientOptions()
        options.api_key = 'api_key'
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options)
            patched.assert_called_once_with(credentials=mock_cred, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)