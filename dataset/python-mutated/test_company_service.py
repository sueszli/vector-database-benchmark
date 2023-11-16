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
from google.longrunning import operations_pb2
from google.oauth2 import service_account
from google.protobuf import field_mask_pb2
from google.protobuf import json_format
from google.type import latlng_pb2
from google.type import postal_address_pb2
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.talent_v4beta1.services.company_service import CompanyServiceAsyncClient, CompanyServiceClient, pagers, transports
from google.cloud.talent_v4beta1.types import common
from google.cloud.talent_v4beta1.types import company
from google.cloud.talent_v4beta1.types import company as gct_company
from google.cloud.talent_v4beta1.types import company_service

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
        for i in range(10):
            print('nop')
    api_endpoint = 'example.googleapis.com'
    api_mtls_endpoint = 'example.mtls.googleapis.com'
    sandbox_endpoint = 'example.sandbox.googleapis.com'
    sandbox_mtls_endpoint = 'example.mtls.sandbox.googleapis.com'
    non_googleapi = 'api.example.com'
    assert CompanyServiceClient._get_default_mtls_endpoint(None) is None
    assert CompanyServiceClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert CompanyServiceClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert CompanyServiceClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert CompanyServiceClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert CompanyServiceClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(CompanyServiceClient, 'grpc'), (CompanyServiceAsyncClient, 'grpc_asyncio'), (CompanyServiceClient, 'rest')])
def test_company_service_client_from_service_account_info(client_class, transport_name):
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
        assert client.transport._host == ('jobs.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://jobs.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.CompanyServiceGrpcTransport, 'grpc'), (transports.CompanyServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (transports.CompanyServiceRestTransport, 'rest')])
def test_company_service_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(CompanyServiceClient, 'grpc'), (CompanyServiceAsyncClient, 'grpc_asyncio'), (CompanyServiceClient, 'rest')])
def test_company_service_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('jobs.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://jobs.googleapis.com')

def test_company_service_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = CompanyServiceClient.get_transport_class()
    available_transports = [transports.CompanyServiceGrpcTransport, transports.CompanyServiceRestTransport]
    assert transport in available_transports
    transport = CompanyServiceClient.get_transport_class('grpc')
    assert transport == transports.CompanyServiceGrpcTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CompanyServiceClient, transports.CompanyServiceGrpcTransport, 'grpc'), (CompanyServiceAsyncClient, transports.CompanyServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (CompanyServiceClient, transports.CompanyServiceRestTransport, 'rest')])
@mock.patch.object(CompanyServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CompanyServiceClient))
@mock.patch.object(CompanyServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CompanyServiceAsyncClient))
def test_company_service_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(CompanyServiceClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(CompanyServiceClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(CompanyServiceClient, transports.CompanyServiceGrpcTransport, 'grpc', 'true'), (CompanyServiceAsyncClient, transports.CompanyServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'true'), (CompanyServiceClient, transports.CompanyServiceGrpcTransport, 'grpc', 'false'), (CompanyServiceAsyncClient, transports.CompanyServiceGrpcAsyncIOTransport, 'grpc_asyncio', 'false'), (CompanyServiceClient, transports.CompanyServiceRestTransport, 'rest', 'true'), (CompanyServiceClient, transports.CompanyServiceRestTransport, 'rest', 'false')])
@mock.patch.object(CompanyServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CompanyServiceClient))
@mock.patch.object(CompanyServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CompanyServiceAsyncClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_company_service_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [CompanyServiceClient, CompanyServiceAsyncClient])
@mock.patch.object(CompanyServiceClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CompanyServiceClient))
@mock.patch.object(CompanyServiceAsyncClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(CompanyServiceAsyncClient))
def test_company_service_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(CompanyServiceClient, transports.CompanyServiceGrpcTransport, 'grpc'), (CompanyServiceAsyncClient, transports.CompanyServiceGrpcAsyncIOTransport, 'grpc_asyncio'), (CompanyServiceClient, transports.CompanyServiceRestTransport, 'rest')])
def test_company_service_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CompanyServiceClient, transports.CompanyServiceGrpcTransport, 'grpc', grpc_helpers), (CompanyServiceAsyncClient, transports.CompanyServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async), (CompanyServiceClient, transports.CompanyServiceRestTransport, 'rest', None)])
def test_company_service_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

def test_company_service_client_client_options_from_dict():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.talent_v4beta1.services.company_service.transports.CompanyServiceGrpcTransport.__init__') as grpc_transport:
        grpc_transport.return_value = None
        client = CompanyServiceClient(client_options={'api_endpoint': 'squid.clam.whelk'})
        grpc_transport.assert_called_once_with(credentials=None, credentials_file=None, host='squid.clam.whelk', scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(CompanyServiceClient, transports.CompanyServiceGrpcTransport, 'grpc', grpc_helpers), (CompanyServiceAsyncClient, transports.CompanyServiceGrpcAsyncIOTransport, 'grpc_asyncio', grpc_helpers_async)])
def test_company_service_client_create_channel_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
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
        create_channel.assert_called_with('jobs.googleapis.com:443', credentials=file_creds, credentials_file=None, quota_project_id=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/jobs'), scopes=None, default_host='jobs.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('request_type', [company_service.CreateCompanyRequest, dict])
def test_create_company(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_company), '__call__') as call:
        call.return_value = gct_company.Company(name='name_value', display_name='display_name_value', external_id='external_id_value', size=common.CompanySize.MINI, headquarters_address='headquarters_address_value', hiring_agency=True, eeo_text='eeo_text_value', website_uri='website_uri_value', career_site_uri='career_site_uri_value', image_uri='image_uri_value', keyword_searchable_job_custom_attributes=['keyword_searchable_job_custom_attributes_value'], suspended=True)
        response = client.create_company(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == company_service.CreateCompanyRequest()
    assert isinstance(response, gct_company.Company)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.external_id == 'external_id_value'
    assert response.size == common.CompanySize.MINI
    assert response.headquarters_address == 'headquarters_address_value'
    assert response.hiring_agency is True
    assert response.eeo_text == 'eeo_text_value'
    assert response.website_uri == 'website_uri_value'
    assert response.career_site_uri == 'career_site_uri_value'
    assert response.image_uri == 'image_uri_value'
    assert response.keyword_searchable_job_custom_attributes == ['keyword_searchable_job_custom_attributes_value']
    assert response.suspended is True

def test_create_company_empty_call():
    if False:
        print('Hello World!')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.create_company), '__call__') as call:
        client.create_company()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == company_service.CreateCompanyRequest()

@pytest.mark.asyncio
async def test_create_company_async(transport: str='grpc_asyncio', request_type=company_service.CreateCompanyRequest):
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.create_company), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_company.Company(name='name_value', display_name='display_name_value', external_id='external_id_value', size=common.CompanySize.MINI, headquarters_address='headquarters_address_value', hiring_agency=True, eeo_text='eeo_text_value', website_uri='website_uri_value', career_site_uri='career_site_uri_value', image_uri='image_uri_value', keyword_searchable_job_custom_attributes=['keyword_searchable_job_custom_attributes_value'], suspended=True))
        response = await client.create_company(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == company_service.CreateCompanyRequest()
    assert isinstance(response, gct_company.Company)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.external_id == 'external_id_value'
    assert response.size == common.CompanySize.MINI
    assert response.headquarters_address == 'headquarters_address_value'
    assert response.hiring_agency is True
    assert response.eeo_text == 'eeo_text_value'
    assert response.website_uri == 'website_uri_value'
    assert response.career_site_uri == 'career_site_uri_value'
    assert response.image_uri == 'image_uri_value'
    assert response.keyword_searchable_job_custom_attributes == ['keyword_searchable_job_custom_attributes_value']
    assert response.suspended is True

@pytest.mark.asyncio
async def test_create_company_async_from_dict():
    await test_create_company_async(request_type=dict)

def test_create_company_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = company_service.CreateCompanyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_company), '__call__') as call:
        call.return_value = gct_company.Company()
        client.create_company(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_create_company_field_headers_async():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = company_service.CreateCompanyRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.create_company), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_company.Company())
        await client.create_company(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_create_company_flattened():
    if False:
        print('Hello World!')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_company), '__call__') as call:
        call.return_value = gct_company.Company()
        client.create_company(parent='parent_value', company=gct_company.Company(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].company
        mock_val = gct_company.Company(name='name_value')
        assert arg == mock_val

def test_create_company_flattened_error():
    if False:
        for i in range(10):
            print('nop')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.create_company(company_service.CreateCompanyRequest(), parent='parent_value', company=gct_company.Company(name='name_value'))

@pytest.mark.asyncio
async def test_create_company_flattened_async():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.create_company), '__call__') as call:
        call.return_value = gct_company.Company()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_company.Company())
        response = await client.create_company(parent='parent_value', company=gct_company.Company(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val
        arg = args[0].company
        mock_val = gct_company.Company(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_create_company_flattened_error_async():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.create_company(company_service.CreateCompanyRequest(), parent='parent_value', company=gct_company.Company(name='name_value'))

@pytest.mark.parametrize('request_type', [company_service.GetCompanyRequest, dict])
def test_get_company(request_type, transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_company), '__call__') as call:
        call.return_value = company.Company(name='name_value', display_name='display_name_value', external_id='external_id_value', size=common.CompanySize.MINI, headquarters_address='headquarters_address_value', hiring_agency=True, eeo_text='eeo_text_value', website_uri='website_uri_value', career_site_uri='career_site_uri_value', image_uri='image_uri_value', keyword_searchable_job_custom_attributes=['keyword_searchable_job_custom_attributes_value'], suspended=True)
        response = client.get_company(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == company_service.GetCompanyRequest()
    assert isinstance(response, company.Company)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.external_id == 'external_id_value'
    assert response.size == common.CompanySize.MINI
    assert response.headquarters_address == 'headquarters_address_value'
    assert response.hiring_agency is True
    assert response.eeo_text == 'eeo_text_value'
    assert response.website_uri == 'website_uri_value'
    assert response.career_site_uri == 'career_site_uri_value'
    assert response.image_uri == 'image_uri_value'
    assert response.keyword_searchable_job_custom_attributes == ['keyword_searchable_job_custom_attributes_value']
    assert response.suspended is True

def test_get_company_empty_call():
    if False:
        i = 10
        return i + 15
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.get_company), '__call__') as call:
        client.get_company()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == company_service.GetCompanyRequest()

@pytest.mark.asyncio
async def test_get_company_async(transport: str='grpc_asyncio', request_type=company_service.GetCompanyRequest):
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.get_company), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(company.Company(name='name_value', display_name='display_name_value', external_id='external_id_value', size=common.CompanySize.MINI, headquarters_address='headquarters_address_value', hiring_agency=True, eeo_text='eeo_text_value', website_uri='website_uri_value', career_site_uri='career_site_uri_value', image_uri='image_uri_value', keyword_searchable_job_custom_attributes=['keyword_searchable_job_custom_attributes_value'], suspended=True))
        response = await client.get_company(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == company_service.GetCompanyRequest()
    assert isinstance(response, company.Company)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.external_id == 'external_id_value'
    assert response.size == common.CompanySize.MINI
    assert response.headquarters_address == 'headquarters_address_value'
    assert response.hiring_agency is True
    assert response.eeo_text == 'eeo_text_value'
    assert response.website_uri == 'website_uri_value'
    assert response.career_site_uri == 'career_site_uri_value'
    assert response.image_uri == 'image_uri_value'
    assert response.keyword_searchable_job_custom_attributes == ['keyword_searchable_job_custom_attributes_value']
    assert response.suspended is True

@pytest.mark.asyncio
async def test_get_company_async_from_dict():
    await test_get_company_async(request_type=dict)

def test_get_company_field_headers():
    if False:
        print('Hello World!')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = company_service.GetCompanyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_company), '__call__') as call:
        call.return_value = company.Company()
        client.get_company(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_get_company_field_headers_async():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = company_service.GetCompanyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.get_company), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(company.Company())
        await client.get_company(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_get_company_flattened():
    if False:
        return 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_company), '__call__') as call:
        call.return_value = company.Company()
        client.get_company(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_get_company_flattened_error():
    if False:
        print('Hello World!')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.get_company(company_service.GetCompanyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_get_company_flattened_async():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_company), '__call__') as call:
        call.return_value = company.Company()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(company.Company())
        response = await client.get_company(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_get_company_flattened_error_async():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.get_company(company_service.GetCompanyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [company_service.UpdateCompanyRequest, dict])
def test_update_company(request_type, transport: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_company), '__call__') as call:
        call.return_value = gct_company.Company(name='name_value', display_name='display_name_value', external_id='external_id_value', size=common.CompanySize.MINI, headquarters_address='headquarters_address_value', hiring_agency=True, eeo_text='eeo_text_value', website_uri='website_uri_value', career_site_uri='career_site_uri_value', image_uri='image_uri_value', keyword_searchable_job_custom_attributes=['keyword_searchable_job_custom_attributes_value'], suspended=True)
        response = client.update_company(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == company_service.UpdateCompanyRequest()
    assert isinstance(response, gct_company.Company)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.external_id == 'external_id_value'
    assert response.size == common.CompanySize.MINI
    assert response.headquarters_address == 'headquarters_address_value'
    assert response.hiring_agency is True
    assert response.eeo_text == 'eeo_text_value'
    assert response.website_uri == 'website_uri_value'
    assert response.career_site_uri == 'career_site_uri_value'
    assert response.image_uri == 'image_uri_value'
    assert response.keyword_searchable_job_custom_attributes == ['keyword_searchable_job_custom_attributes_value']
    assert response.suspended is True

def test_update_company_empty_call():
    if False:
        print('Hello World!')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.update_company), '__call__') as call:
        client.update_company()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == company_service.UpdateCompanyRequest()

@pytest.mark.asyncio
async def test_update_company_async(transport: str='grpc_asyncio', request_type=company_service.UpdateCompanyRequest):
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.update_company), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_company.Company(name='name_value', display_name='display_name_value', external_id='external_id_value', size=common.CompanySize.MINI, headquarters_address='headquarters_address_value', hiring_agency=True, eeo_text='eeo_text_value', website_uri='website_uri_value', career_site_uri='career_site_uri_value', image_uri='image_uri_value', keyword_searchable_job_custom_attributes=['keyword_searchable_job_custom_attributes_value'], suspended=True))
        response = await client.update_company(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == company_service.UpdateCompanyRequest()
    assert isinstance(response, gct_company.Company)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.external_id == 'external_id_value'
    assert response.size == common.CompanySize.MINI
    assert response.headquarters_address == 'headquarters_address_value'
    assert response.hiring_agency is True
    assert response.eeo_text == 'eeo_text_value'
    assert response.website_uri == 'website_uri_value'
    assert response.career_site_uri == 'career_site_uri_value'
    assert response.image_uri == 'image_uri_value'
    assert response.keyword_searchable_job_custom_attributes == ['keyword_searchable_job_custom_attributes_value']
    assert response.suspended is True

@pytest.mark.asyncio
async def test_update_company_async_from_dict():
    await test_update_company_async(request_type=dict)

def test_update_company_field_headers():
    if False:
        print('Hello World!')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = company_service.UpdateCompanyRequest()
    request.company.name = 'name_value'
    with mock.patch.object(type(client.transport.update_company), '__call__') as call:
        call.return_value = gct_company.Company()
        client.update_company(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'company.name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_update_company_field_headers_async():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = company_service.UpdateCompanyRequest()
    request.company.name = 'name_value'
    with mock.patch.object(type(client.transport.update_company), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_company.Company())
        await client.update_company(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'company.name=name_value') in kw['metadata']

def test_update_company_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_company), '__call__') as call:
        call.return_value = gct_company.Company()
        client.update_company(company=gct_company.Company(name='name_value'))
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].company
        mock_val = gct_company.Company(name='name_value')
        assert arg == mock_val

def test_update_company_flattened_error():
    if False:
        print('Hello World!')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.update_company(company_service.UpdateCompanyRequest(), company=gct_company.Company(name='name_value'))

@pytest.mark.asyncio
async def test_update_company_flattened_async():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.update_company), '__call__') as call:
        call.return_value = gct_company.Company()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(gct_company.Company())
        response = await client.update_company(company=gct_company.Company(name='name_value'))
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].company
        mock_val = gct_company.Company(name='name_value')
        assert arg == mock_val

@pytest.mark.asyncio
async def test_update_company_flattened_error_async():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.update_company(company_service.UpdateCompanyRequest(), company=gct_company.Company(name='name_value'))

@pytest.mark.parametrize('request_type', [company_service.DeleteCompanyRequest, dict])
def test_delete_company(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_company), '__call__') as call:
        call.return_value = None
        response = client.delete_company(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == company_service.DeleteCompanyRequest()
    assert response is None

def test_delete_company_empty_call():
    if False:
        while True:
            i = 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.delete_company), '__call__') as call:
        client.delete_company()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == company_service.DeleteCompanyRequest()

@pytest.mark.asyncio
async def test_delete_company_async(transport: str='grpc_asyncio', request_type=company_service.DeleteCompanyRequest):
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.delete_company), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_company(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == company_service.DeleteCompanyRequest()
    assert response is None

@pytest.mark.asyncio
async def test_delete_company_async_from_dict():
    await test_delete_company_async(request_type=dict)

def test_delete_company_field_headers():
    if False:
        print('Hello World!')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = company_service.DeleteCompanyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_company), '__call__') as call:
        call.return_value = None
        client.delete_company(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

@pytest.mark.asyncio
async def test_delete_company_field_headers_async():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = company_service.DeleteCompanyRequest()
    request.name = 'name_value'
    with mock.patch.object(type(client.transport.delete_company), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        await client.delete_company(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'name=name_value') in kw['metadata']

def test_delete_company_flattened():
    if False:
        while True:
            i = 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_company), '__call__') as call:
        call.return_value = None
        client.delete_company(name='name_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

def test_delete_company_flattened_error():
    if False:
        print('Hello World!')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.delete_company(company_service.DeleteCompanyRequest(), name='name_value')

@pytest.mark.asyncio
async def test_delete_company_flattened_async():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.delete_company), '__call__') as call:
        call.return_value = None
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(None)
        response = await client.delete_company(name='name_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].name
        mock_val = 'name_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_delete_company_flattened_error_async():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.delete_company(company_service.DeleteCompanyRequest(), name='name_value')

@pytest.mark.parametrize('request_type', [company_service.ListCompaniesRequest, dict])
def test_list_companies(request_type, transport: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_companies), '__call__') as call:
        call.return_value = company_service.ListCompaniesResponse(next_page_token='next_page_token_value')
        response = client.list_companies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == company_service.ListCompaniesRequest()
    assert isinstance(response, pagers.ListCompaniesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_companies_empty_call():
    if False:
        for i in range(10):
            print('nop')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc')
    with mock.patch.object(type(client.transport.list_companies), '__call__') as call:
        client.list_companies()
        call.assert_called()
        (_, args, _) = call.mock_calls[0]
        assert args[0] == company_service.ListCompaniesRequest()

@pytest.mark.asyncio
async def test_list_companies_async(transport: str='grpc_asyncio', request_type=company_service.ListCompaniesRequest):
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request = request_type()
    with mock.patch.object(type(client.transport.list_companies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(company_service.ListCompaniesResponse(next_page_token='next_page_token_value'))
        response = await client.list_companies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == company_service.ListCompaniesRequest()
    assert isinstance(response, pagers.ListCompaniesAsyncPager)
    assert response.next_page_token == 'next_page_token_value'

@pytest.mark.asyncio
async def test_list_companies_async_from_dict():
    await test_list_companies_async(request_type=dict)

def test_list_companies_field_headers():
    if False:
        for i in range(10):
            print('nop')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    request = company_service.ListCompaniesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_companies), '__call__') as call:
        call.return_value = company_service.ListCompaniesResponse()
        client.list_companies(request)
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

@pytest.mark.asyncio
async def test_list_companies_field_headers_async():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    request = company_service.ListCompaniesRequest()
    request.parent = 'parent_value'
    with mock.patch.object(type(client.transport.list_companies), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(company_service.ListCompaniesResponse())
        await client.list_companies(request)
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        assert args[0] == request
    (_, _, kw) = call.mock_calls[0]
    assert ('x-goog-request-params', 'parent=parent_value') in kw['metadata']

def test_list_companies_flattened():
    if False:
        return 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_companies), '__call__') as call:
        call.return_value = company_service.ListCompaniesResponse()
        client.list_companies(parent='parent_value')
        assert len(call.mock_calls) == 1
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

def test_list_companies_flattened_error():
    if False:
        return 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client.list_companies(company_service.ListCompaniesRequest(), parent='parent_value')

@pytest.mark.asyncio
async def test_list_companies_flattened_async():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.list_companies), '__call__') as call:
        call.return_value = company_service.ListCompaniesResponse()
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(company_service.ListCompaniesResponse())
        response = await client.list_companies(parent='parent_value')
        assert len(call.mock_calls)
        (_, args, _) = call.mock_calls[0]
        arg = args[0].parent
        mock_val = 'parent_value'
        assert arg == mock_val

@pytest.mark.asyncio
async def test_list_companies_flattened_error_async():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        await client.list_companies(company_service.ListCompaniesRequest(), parent='parent_value')

def test_list_companies_pager(transport_name: str='grpc'):
    if False:
        for i in range(10):
            print('nop')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_companies), '__call__') as call:
        call.side_effect = (company_service.ListCompaniesResponse(companies=[company.Company(), company.Company(), company.Company()], next_page_token='abc'), company_service.ListCompaniesResponse(companies=[], next_page_token='def'), company_service.ListCompaniesResponse(companies=[company.Company()], next_page_token='ghi'), company_service.ListCompaniesResponse(companies=[company.Company(), company.Company()]), RuntimeError)
        metadata = ()
        metadata = tuple(metadata) + (gapic_v1.routing_header.to_grpc_metadata((('parent', ''),)),)
        pager = client.list_companies(request={})
        assert pager._metadata == metadata
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, company.Company) for i in results))

def test_list_companies_pages(transport_name: str='grpc'):
    if False:
        i = 10
        return i + 15
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials, transport=transport_name)
    with mock.patch.object(type(client.transport.list_companies), '__call__') as call:
        call.side_effect = (company_service.ListCompaniesResponse(companies=[company.Company(), company.Company(), company.Company()], next_page_token='abc'), company_service.ListCompaniesResponse(companies=[], next_page_token='def'), company_service.ListCompaniesResponse(companies=[company.Company()], next_page_token='ghi'), company_service.ListCompaniesResponse(companies=[company.Company(), company.Company()]), RuntimeError)
        pages = list(client.list_companies(request={}).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.asyncio
async def test_list_companies_async_pager():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_companies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (company_service.ListCompaniesResponse(companies=[company.Company(), company.Company(), company.Company()], next_page_token='abc'), company_service.ListCompaniesResponse(companies=[], next_page_token='def'), company_service.ListCompaniesResponse(companies=[company.Company()], next_page_token='ghi'), company_service.ListCompaniesResponse(companies=[company.Company(), company.Company()]), RuntimeError)
        async_pager = await client.list_companies(request={})
        assert async_pager.next_page_token == 'abc'
        responses = []
        async for response in async_pager:
            responses.append(response)
        assert len(responses) == 6
        assert all((isinstance(i, company.Company) for i in responses))

@pytest.mark.asyncio
async def test_list_companies_async_pages():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials)
    with mock.patch.object(type(client.transport.list_companies), '__call__', new_callable=mock.AsyncMock) as call:
        call.side_effect = (company_service.ListCompaniesResponse(companies=[company.Company(), company.Company(), company.Company()], next_page_token='abc'), company_service.ListCompaniesResponse(companies=[], next_page_token='def'), company_service.ListCompaniesResponse(companies=[company.Company()], next_page_token='ghi'), company_service.ListCompaniesResponse(companies=[company.Company(), company.Company()]), RuntimeError)
        pages = []
        async for page_ in (await client.list_companies(request={})).pages:
            pages.append(page_)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [company_service.CreateCompanyRequest, dict])
def test_create_company_rest(request_type):
    if False:
        while True:
            i = 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/tenants/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gct_company.Company(name='name_value', display_name='display_name_value', external_id='external_id_value', size=common.CompanySize.MINI, headquarters_address='headquarters_address_value', hiring_agency=True, eeo_text='eeo_text_value', website_uri='website_uri_value', career_site_uri='career_site_uri_value', image_uri='image_uri_value', keyword_searchable_job_custom_attributes=['keyword_searchable_job_custom_attributes_value'], suspended=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = gct_company.Company.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_company(request)
    assert isinstance(response, gct_company.Company)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.external_id == 'external_id_value'
    assert response.size == common.CompanySize.MINI
    assert response.headquarters_address == 'headquarters_address_value'
    assert response.hiring_agency is True
    assert response.eeo_text == 'eeo_text_value'
    assert response.website_uri == 'website_uri_value'
    assert response.career_site_uri == 'career_site_uri_value'
    assert response.image_uri == 'image_uri_value'
    assert response.keyword_searchable_job_custom_attributes == ['keyword_searchable_job_custom_attributes_value']
    assert response.suspended is True

def test_create_company_rest_required_fields(request_type=company_service.CreateCompanyRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CompanyServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_company._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_company._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gct_company.Company()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gct_company.Company.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.create_company(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_company_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CompanyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_company._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('parent', 'company'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_company_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CompanyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CompanyServiceRestInterceptor())
    client = CompanyServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CompanyServiceRestInterceptor, 'post_create_company') as post, mock.patch.object(transports.CompanyServiceRestInterceptor, 'pre_create_company') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = company_service.CreateCompanyRequest.pb(company_service.CreateCompanyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gct_company.Company.to_json(gct_company.Company())
        request = company_service.CreateCompanyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gct_company.Company()
        client.create_company(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_company_rest_bad_request(transport: str='rest', request_type=company_service.CreateCompanyRequest):
    if False:
        while True:
            i = 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/tenants/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_company(request)

def test_create_company_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gct_company.Company()
        sample_request = {'parent': 'projects/sample1/tenants/sample2'}
        mock_args = dict(parent='parent_value', company=gct_company.Company(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gct_company.Company.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_company(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v4beta1/{parent=projects/*/tenants/*}/companies' % client.transport._host, args[1])

def test_create_company_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_company(company_service.CreateCompanyRequest(), parent='parent_value', company=gct_company.Company(name='name_value'))

def test_create_company_rest_error():
    if False:
        while True:
            i = 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [company_service.GetCompanyRequest, dict])
def test_get_company_rest(request_type):
    if False:
        print('Hello World!')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/tenants/sample2/companies/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = company.Company(name='name_value', display_name='display_name_value', external_id='external_id_value', size=common.CompanySize.MINI, headquarters_address='headquarters_address_value', hiring_agency=True, eeo_text='eeo_text_value', website_uri='website_uri_value', career_site_uri='career_site_uri_value', image_uri='image_uri_value', keyword_searchable_job_custom_attributes=['keyword_searchable_job_custom_attributes_value'], suspended=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = company.Company.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_company(request)
    assert isinstance(response, company.Company)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.external_id == 'external_id_value'
    assert response.size == common.CompanySize.MINI
    assert response.headquarters_address == 'headquarters_address_value'
    assert response.hiring_agency is True
    assert response.eeo_text == 'eeo_text_value'
    assert response.website_uri == 'website_uri_value'
    assert response.career_site_uri == 'career_site_uri_value'
    assert response.image_uri == 'image_uri_value'
    assert response.keyword_searchable_job_custom_attributes == ['keyword_searchable_job_custom_attributes_value']
    assert response.suspended is True

def test_get_company_rest_required_fields(request_type=company_service.GetCompanyRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.CompanyServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_company._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_company._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = company.Company()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = company.Company.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_company(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_company_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CompanyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_company._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_company_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CompanyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CompanyServiceRestInterceptor())
    client = CompanyServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CompanyServiceRestInterceptor, 'post_get_company') as post, mock.patch.object(transports.CompanyServiceRestInterceptor, 'pre_get_company') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = company_service.GetCompanyRequest.pb(company_service.GetCompanyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = company.Company.to_json(company.Company())
        request = company_service.GetCompanyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = company.Company()
        client.get_company(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_company_rest_bad_request(transport: str='rest', request_type=company_service.GetCompanyRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/tenants/sample2/companies/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_company(request)

def test_get_company_rest_flattened():
    if False:
        return 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = company.Company()
        sample_request = {'name': 'projects/sample1/tenants/sample2/companies/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = company.Company.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_company(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v4beta1/{name=projects/*/tenants/*/companies/*}' % client.transport._host, args[1])

def test_get_company_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_company(company_service.GetCompanyRequest(), name='name_value')

def test_get_company_rest_error():
    if False:
        i = 10
        return i + 15
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [company_service.UpdateCompanyRequest, dict])
def test_update_company_rest(request_type):
    if False:
        print('Hello World!')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'company': {'name': 'projects/sample1/tenants/sample2/companies/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gct_company.Company(name='name_value', display_name='display_name_value', external_id='external_id_value', size=common.CompanySize.MINI, headquarters_address='headquarters_address_value', hiring_agency=True, eeo_text='eeo_text_value', website_uri='website_uri_value', career_site_uri='career_site_uri_value', image_uri='image_uri_value', keyword_searchable_job_custom_attributes=['keyword_searchable_job_custom_attributes_value'], suspended=True)
        response_value = Response()
        response_value.status_code = 200
        return_value = gct_company.Company.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_company(request)
    assert isinstance(response, gct_company.Company)
    assert response.name == 'name_value'
    assert response.display_name == 'display_name_value'
    assert response.external_id == 'external_id_value'
    assert response.size == common.CompanySize.MINI
    assert response.headquarters_address == 'headquarters_address_value'
    assert response.hiring_agency is True
    assert response.eeo_text == 'eeo_text_value'
    assert response.website_uri == 'website_uri_value'
    assert response.career_site_uri == 'career_site_uri_value'
    assert response.image_uri == 'image_uri_value'
    assert response.keyword_searchable_job_custom_attributes == ['keyword_searchable_job_custom_attributes_value']
    assert response.suspended is True

def test_update_company_rest_required_fields(request_type=company_service.UpdateCompanyRequest):
    if False:
        return 10
    transport_class = transports.CompanyServiceRestTransport
    request_init = {}
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_company._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_company._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = gct_company.Company()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = gct_company.Company.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_company(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_company_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CompanyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_company._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('company',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_company_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.CompanyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CompanyServiceRestInterceptor())
    client = CompanyServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CompanyServiceRestInterceptor, 'post_update_company') as post, mock.patch.object(transports.CompanyServiceRestInterceptor, 'pre_update_company') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = company_service.UpdateCompanyRequest.pb(company_service.UpdateCompanyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = gct_company.Company.to_json(gct_company.Company())
        request = company_service.UpdateCompanyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = gct_company.Company()
        client.update_company(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_company_rest_bad_request(transport: str='rest', request_type=company_service.UpdateCompanyRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'company': {'name': 'projects/sample1/tenants/sample2/companies/sample3'}}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_company(request)

def test_update_company_rest_flattened():
    if False:
        return 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = gct_company.Company()
        sample_request = {'company': {'name': 'projects/sample1/tenants/sample2/companies/sample3'}}
        mock_args = dict(company=gct_company.Company(name='name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = gct_company.Company.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_company(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v4beta1/{company.name=projects/*/tenants/*/companies/*}' % client.transport._host, args[1])

def test_update_company_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_company(company_service.UpdateCompanyRequest(), company=gct_company.Company(name='name_value'))

def test_update_company_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [company_service.DeleteCompanyRequest, dict])
def test_delete_company_rest(request_type):
    if False:
        print('Hello World!')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'name': 'projects/sample1/tenants/sample2/companies/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_company(request)
    assert response is None

def test_delete_company_rest_required_fields(request_type=company_service.DeleteCompanyRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.CompanyServiceRestTransport
    request_init = {}
    request_init['name'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_company._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['name'] = 'name_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_company._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'name' in jsonified_request
    assert jsonified_request['name'] == 'name_value'
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_company(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_company_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.CompanyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_company._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('name',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_company_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.CompanyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CompanyServiceRestInterceptor())
    client = CompanyServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CompanyServiceRestInterceptor, 'pre_delete_company') as pre:
        pre.assert_not_called()
        pb_message = company_service.DeleteCompanyRequest.pb(company_service.DeleteCompanyRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        request = company_service.DeleteCompanyRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        client.delete_company(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()

def test_delete_company_rest_bad_request(transport: str='rest', request_type=company_service.DeleteCompanyRequest):
    if False:
        return 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'name': 'projects/sample1/tenants/sample2/companies/sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_company(request)

def test_delete_company_rest_flattened():
    if False:
        return 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = None
        sample_request = {'name': 'projects/sample1/tenants/sample2/companies/sample3'}
        mock_args = dict(name='name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        json_return_value = ''
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_company(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v4beta1/{name=projects/*/tenants/*/companies/*}' % client.transport._host, args[1])

def test_delete_company_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_company(company_service.DeleteCompanyRequest(), name='name_value')

def test_delete_company_rest_error():
    if False:
        while True:
            i = 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [company_service.ListCompaniesRequest, dict])
def test_list_companies_rest(request_type):
    if False:
        while True:
            i = 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'parent': 'projects/sample1/tenants/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = company_service.ListCompaniesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = company_service.ListCompaniesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_companies(request)
    assert isinstance(response, pagers.ListCompaniesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_companies_rest_required_fields(request_type=company_service.ListCompaniesRequest):
    if False:
        return 10
    transport_class = transports.CompanyServiceRestTransport
    request_init = {}
    request_init['parent'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_companies._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['parent'] = 'parent_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_companies._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('page_size', 'page_token', 'require_open_jobs'))
    jsonified_request.update(unset_fields)
    assert 'parent' in jsonified_request
    assert jsonified_request['parent'] == 'parent_value'
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = company_service.ListCompaniesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = company_service.ListCompaniesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_companies(request)
            expected_params = [('$alt', 'json;enum-encoding=int')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_companies_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.CompanyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_companies._get_unset_required_fields({})
    assert set(unset_fields) == set(('pageSize', 'pageToken', 'requireOpenJobs')) & set(('parent',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_companies_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.CompanyServiceRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.CompanyServiceRestInterceptor())
    client = CompanyServiceClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.CompanyServiceRestInterceptor, 'post_list_companies') as post, mock.patch.object(transports.CompanyServiceRestInterceptor, 'pre_list_companies') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = company_service.ListCompaniesRequest.pb(company_service.ListCompaniesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = company_service.ListCompaniesResponse.to_json(company_service.ListCompaniesResponse())
        request = company_service.ListCompaniesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = company_service.ListCompaniesResponse()
        client.list_companies(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_companies_rest_bad_request(transport: str='rest', request_type=company_service.ListCompaniesRequest):
    if False:
        for i in range(10):
            print('nop')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'parent': 'projects/sample1/tenants/sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_companies(request)

def test_list_companies_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = company_service.ListCompaniesResponse()
        sample_request = {'parent': 'projects/sample1/tenants/sample2'}
        mock_args = dict(parent='parent_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = company_service.ListCompaniesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_companies(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/v4beta1/{parent=projects/*/tenants/*}/companies' % client.transport._host, args[1])

def test_list_companies_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_companies(company_service.ListCompaniesRequest(), parent='parent_value')

def test_list_companies_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (company_service.ListCompaniesResponse(companies=[company.Company(), company.Company(), company.Company()], next_page_token='abc'), company_service.ListCompaniesResponse(companies=[], next_page_token='def'), company_service.ListCompaniesResponse(companies=[company.Company()], next_page_token='ghi'), company_service.ListCompaniesResponse(companies=[company.Company(), company.Company()]))
        response = response + response
        response = tuple((company_service.ListCompaniesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'parent': 'projects/sample1/tenants/sample2'}
        pager = client.list_companies(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, company.Company) for i in results))
        pages = list(client.list_companies(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        i = 10
        return i + 15
    transport = transports.CompanyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.CompanyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CompanyServiceClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.CompanyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CompanyServiceClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = CompanyServiceClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.CompanyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = CompanyServiceClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.CompanyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    client = CompanyServiceClient(transport=transport)
    assert client.transport is transport

def test_transport_get_channel():
    if False:
        print('Hello World!')
    transport = transports.CompanyServiceGrpcTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel
    transport = transports.CompanyServiceGrpcAsyncIOTransport(credentials=ga_credentials.AnonymousCredentials())
    channel = transport.grpc_channel
    assert channel

@pytest.mark.parametrize('transport_class', [transports.CompanyServiceGrpcTransport, transports.CompanyServiceGrpcAsyncIOTransport, transports.CompanyServiceRestTransport])
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
        i = 10
        return i + 15
    transport = CompanyServiceClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_transport_grpc_default():
    if False:
        while True:
            i = 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    assert isinstance(client.transport, transports.CompanyServiceGrpcTransport)

def test_company_service_base_transport_error():
    if False:
        print('Hello World!')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.CompanyServiceTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_company_service_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.talent_v4beta1.services.company_service.transports.CompanyServiceTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.CompanyServiceTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('create_company', 'get_company', 'update_company', 'delete_company', 'list_companies', 'get_operation')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_company_service_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.talent_v4beta1.services.company_service.transports.CompanyServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CompanyServiceTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/jobs'), quota_project_id='octopus')

def test_company_service_base_transport_with_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.talent_v4beta1.services.company_service.transports.CompanyServiceTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.CompanyServiceTransport()
        adc.assert_called_once()

def test_company_service_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        CompanyServiceClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/jobs'), quota_project_id=None)

@pytest.mark.parametrize('transport_class', [transports.CompanyServiceGrpcTransport, transports.CompanyServiceGrpcAsyncIOTransport])
def test_company_service_transport_auth_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        adc.assert_called_once_with(scopes=['1', '2'], default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/jobs'), quota_project_id='octopus')

@pytest.mark.parametrize('transport_class', [transports.CompanyServiceGrpcTransport, transports.CompanyServiceGrpcAsyncIOTransport, transports.CompanyServiceRestTransport])
def test_company_service_transport_auth_gdch_credentials(transport_class):
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

@pytest.mark.parametrize('transport_class,grpc_helpers', [(transports.CompanyServiceGrpcTransport, grpc_helpers), (transports.CompanyServiceGrpcAsyncIOTransport, grpc_helpers_async)])
def test_company_service_transport_create_channel(transport_class, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch.object(grpc_helpers, 'create_channel', autospec=True) as create_channel:
        creds = ga_credentials.AnonymousCredentials()
        adc.return_value = (creds, None)
        transport_class(quota_project_id='octopus', scopes=['1', '2'])
        create_channel.assert_called_with('jobs.googleapis.com:443', credentials=creds, credentials_file=None, quota_project_id='octopus', default_scopes=('https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/jobs'), scopes=['1', '2'], default_host='jobs.googleapis.com', ssl_credentials=None, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])

@pytest.mark.parametrize('transport_class', [transports.CompanyServiceGrpcTransport, transports.CompanyServiceGrpcAsyncIOTransport])
def test_company_service_grpc_transport_client_cert_source_for_mtls(transport_class):
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

def test_company_service_http_transport_client_cert_source_for_mtls():
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.CompanyServiceRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_company_service_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='jobs.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('jobs.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://jobs.googleapis.com')

@pytest.mark.parametrize('transport_name', ['grpc', 'grpc_asyncio', 'rest'])
def test_company_service_host_with_port(transport_name):
    if False:
        return 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='jobs.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('jobs.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://jobs.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_company_service_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = CompanyServiceClient(credentials=creds1, transport=transport_name)
    client2 = CompanyServiceClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.create_company._session
    session2 = client2.transport.create_company._session
    assert session1 != session2
    session1 = client1.transport.get_company._session
    session2 = client2.transport.get_company._session
    assert session1 != session2
    session1 = client1.transport.update_company._session
    session2 = client2.transport.update_company._session
    assert session1 != session2
    session1 = client1.transport.delete_company._session
    session2 = client2.transport.delete_company._session
    assert session1 != session2
    session1 = client1.transport.list_companies._session
    session2 = client2.transport.list_companies._session
    assert session1 != session2

def test_company_service_grpc_transport_channel():
    if False:
        i = 10
        return i + 15
    channel = grpc.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CompanyServiceGrpcTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

def test_company_service_grpc_asyncio_transport_channel():
    if False:
        for i in range(10):
            print('nop')
    channel = aio.secure_channel('http://localhost/', grpc.local_channel_credentials())
    transport = transports.CompanyServiceGrpcAsyncIOTransport(host='squid.clam.whelk', channel=channel)
    assert transport.grpc_channel == channel
    assert transport._host == 'squid.clam.whelk:443'
    assert transport._ssl_channel_credentials == None

@pytest.mark.parametrize('transport_class', [transports.CompanyServiceGrpcTransport, transports.CompanyServiceGrpcAsyncIOTransport])
def test_company_service_transport_channel_mtls_with_client_cert_source(transport_class):
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

@pytest.mark.parametrize('transport_class', [transports.CompanyServiceGrpcTransport, transports.CompanyServiceGrpcAsyncIOTransport])
def test_company_service_transport_channel_mtls_with_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
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

def test_company_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    tenant = 'clam'
    company = 'whelk'
    expected = 'projects/{project}/tenants/{tenant}/companies/{company}'.format(project=project, tenant=tenant, company=company)
    actual = CompanyServiceClient.company_path(project, tenant, company)
    assert expected == actual

def test_parse_company_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'octopus', 'tenant': 'oyster', 'company': 'nudibranch'}
    path = CompanyServiceClient.company_path(**expected)
    actual = CompanyServiceClient.parse_company_path(path)
    assert expected == actual

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'cuttlefish'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = CompanyServiceClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        print('Hello World!')
    expected = {'billing_account': 'mussel'}
    path = CompanyServiceClient.common_billing_account_path(**expected)
    actual = CompanyServiceClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'winkle'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = CompanyServiceClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'nautilus'}
    path = CompanyServiceClient.common_folder_path(**expected)
    actual = CompanyServiceClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'scallop'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = CompanyServiceClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        return 10
    expected = {'organization': 'abalone'}
    path = CompanyServiceClient.common_organization_path(**expected)
    actual = CompanyServiceClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        i = 10
        return i + 15
    project = 'squid'
    expected = 'projects/{project}'.format(project=project)
    actual = CompanyServiceClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        print('Hello World!')
    expected = {'project': 'clam'}
    path = CompanyServiceClient.common_project_path(**expected)
    actual = CompanyServiceClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        i = 10
        return i + 15
    project = 'whelk'
    location = 'octopus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = CompanyServiceClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'oyster', 'location': 'nudibranch'}
    path = CompanyServiceClient.common_location_path(**expected)
    actual = CompanyServiceClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.CompanyServiceTransport, '_prep_wrapped_messages') as prep:
        client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.CompanyServiceTransport, '_prep_wrapped_messages') as prep:
        transport_class = CompanyServiceClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

@pytest.mark.asyncio
async def test_transport_close_async():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport='grpc_asyncio')
    with mock.patch.object(type(getattr(client.transport, 'grpc_channel')), 'close') as close:
        async with client:
            close.assert_not_called()
        close.assert_called_once()

def test_get_operation_rest_bad_request(transport: str='rest', request_type=operations_pb2.GetOperationRequest):
    if False:
        return 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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

def test_get_operation(transport: str='grpc'):
    if False:
        print('Hello World!')
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials())
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
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
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
        while True:
            i = 10
    client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = operations_pb2.Operation()
        response = client.get_operation(request={'name': 'locations'})
        call.assert_called()

@pytest.mark.asyncio
async def test_get_operation_from_dict_async():
    client = CompanyServiceAsyncClient(credentials=ga_credentials.AnonymousCredentials())
    with mock.patch.object(type(client.transport.get_operation), '__call__') as call:
        call.return_value = grpc_helpers_async.FakeUnaryUnaryCall(operations_pb2.Operation())
        response = await client.get_operation(request={'name': 'locations'})
        call.assert_called()

def test_transport_close():
    if False:
        return 10
    transports = {'rest': '_session', 'grpc': '_grpc_channel'}
    for (transport, close_name) in transports.items():
        client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        print('Hello World!')
    transports = ['rest', 'grpc']
    for transport in transports:
        client = CompanyServiceClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(CompanyServiceClient, transports.CompanyServiceGrpcTransport), (CompanyServiceAsyncClient, transports.CompanyServiceGrpcAsyncIOTransport)])
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