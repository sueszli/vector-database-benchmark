import os
try:
    from unittest import mock
    from unittest.mock import AsyncMock
except ImportError:
    import mock
from collections.abc import Iterable
import json
import math
from google.api_core import future, gapic_v1, grpc_helpers, grpc_helpers_async, path_template
from google.api_core import client_options
from google.api_core import exceptions as core_exceptions
from google.api_core import extended_operation
import google.auth
from google.auth import credentials as ga_credentials
from google.auth.exceptions import MutualTLSChannelError
from google.oauth2 import service_account
from google.protobuf import json_format
import grpc
from grpc.experimental import aio
from proto.marshal.rules import wrappers
from proto.marshal.rules.dates import DurationRule, TimestampRule
import pytest
from requests import PreparedRequest, Request, Response
from requests.sessions import Session
from google.cloud.compute_v1.services.backend_services import BackendServicesClient, pagers, transports
from google.cloud.compute_v1.types import compute

def client_cert_source_callback():
    if False:
        i = 10
        return i + 15
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
    assert BackendServicesClient._get_default_mtls_endpoint(None) is None
    assert BackendServicesClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert BackendServicesClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert BackendServicesClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert BackendServicesClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert BackendServicesClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(BackendServicesClient, 'rest')])
def test_backend_services_client_from_service_account_info(client_class, transport_name):
    if False:
        return 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('compute.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.BackendServicesRestTransport, 'rest')])
def test_backend_services_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(BackendServicesClient, 'rest')])
def test_backend_services_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('compute.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com')

def test_backend_services_client_get_transport_class():
    if False:
        i = 10
        return i + 15
    transport = BackendServicesClient.get_transport_class()
    available_transports = [transports.BackendServicesRestTransport]
    assert transport in available_transports
    transport = BackendServicesClient.get_transport_class('rest')
    assert transport == transports.BackendServicesRestTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(BackendServicesClient, transports.BackendServicesRestTransport, 'rest')])
@mock.patch.object(BackendServicesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(BackendServicesClient))
def test_backend_services_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(BackendServicesClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(BackendServicesClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(BackendServicesClient, transports.BackendServicesRestTransport, 'rest', 'true'), (BackendServicesClient, transports.BackendServicesRestTransport, 'rest', 'false')])
@mock.patch.object(BackendServicesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(BackendServicesClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_backend_services_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [BackendServicesClient])
@mock.patch.object(BackendServicesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(BackendServicesClient))
def test_backend_services_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(BackendServicesClient, transports.BackendServicesRestTransport, 'rest')])
def test_backend_services_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(BackendServicesClient, transports.BackendServicesRestTransport, 'rest', None)])
def test_backend_services_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('request_type', [compute.AddSignedUrlKeyBackendServiceRequest, dict])
def test_add_signed_url_key_rest(request_type):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request_init['signed_url_key_resource'] = {'key_name': 'key_name_value', 'key_value': 'key_value_value'}
    test_field = compute.AddSignedUrlKeyBackendServiceRequest.meta.fields['signed_url_key_resource']

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
    for (field, value) in request_init['signed_url_key_resource'].items():
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
                for i in range(0, len(request_init['signed_url_key_resource'][field])):
                    del request_init['signed_url_key_resource'][field][i][subfield]
            else:
                del request_init['signed_url_key_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.add_signed_url_key(request)
    assert isinstance(response, extended_operation.ExtendedOperation)
    assert response.client_operation_id == 'client_operation_id_value'
    assert response.creation_timestamp == 'creation_timestamp_value'
    assert response.description == 'description_value'
    assert response.end_time == 'end_time_value'
    assert response.http_error_message == 'http_error_message_value'
    assert response.http_error_status_code == 2374
    assert response.id == 205
    assert response.insert_time == 'insert_time_value'
    assert response.kind == 'kind_value'
    assert response.name == 'name_value'
    assert response.operation_group_id == 'operation_group_id_value'
    assert response.operation_type == 'operation_type_value'
    assert response.progress == 885
    assert response.region == 'region_value'
    assert response.self_link == 'self_link_value'
    assert response.start_time == 'start_time_value'
    assert response.status == compute.Operation.Status.DONE
    assert response.status_message == 'status_message_value'
    assert response.target_id == 947
    assert response.target_link == 'target_link_value'
    assert response.user == 'user_value'
    assert response.zone == 'zone_value'

def test_add_signed_url_key_rest_required_fields(request_type=compute.AddSignedUrlKeyBackendServiceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['backend_service'] = ''
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_signed_url_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['backendService'] = 'backend_service_value'
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_signed_url_key._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'backendService' in jsonified_request
    assert jsonified_request['backendService'] == 'backend_service_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Operation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Operation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.add_signed_url_key(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_add_signed_url_key_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.add_signed_url_key._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('backendService', 'project', 'signedUrlKeyResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_add_signed_url_key_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_add_signed_url_key') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_add_signed_url_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.AddSignedUrlKeyBackendServiceRequest.pb(compute.AddSignedUrlKeyBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.AddSignedUrlKeyBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.add_signed_url_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_add_signed_url_key_rest_bad_request(transport: str='rest', request_type=compute.AddSignedUrlKeyBackendServiceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.add_signed_url_key(request)

def test_add_signed_url_key_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'backend_service': 'sample2'}
        mock_args = dict(project='project_value', backend_service='backend_service_value', signed_url_key_resource=compute.SignedUrlKey(key_name='key_name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.add_signed_url_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices/{backend_service}/addSignedUrlKey' % client.transport._host, args[1])

def test_add_signed_url_key_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.add_signed_url_key(compute.AddSignedUrlKeyBackendServiceRequest(), project='project_value', backend_service='backend_service_value', signed_url_key_resource=compute.SignedUrlKey(key_name='key_name_value'))

def test_add_signed_url_key_rest_error():
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.AddSignedUrlKeyBackendServiceRequest, dict])
def test_add_signed_url_key_unary_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request_init['signed_url_key_resource'] = {'key_name': 'key_name_value', 'key_value': 'key_value_value'}
    test_field = compute.AddSignedUrlKeyBackendServiceRequest.meta.fields['signed_url_key_resource']

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
    for (field, value) in request_init['signed_url_key_resource'].items():
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
                for i in range(0, len(request_init['signed_url_key_resource'][field])):
                    del request_init['signed_url_key_resource'][field][i][subfield]
            else:
                del request_init['signed_url_key_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.add_signed_url_key_unary(request)
    assert isinstance(response, compute.Operation)

def test_add_signed_url_key_unary_rest_required_fields(request_type=compute.AddSignedUrlKeyBackendServiceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['backend_service'] = ''
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_signed_url_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['backendService'] = 'backend_service_value'
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_signed_url_key._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'backendService' in jsonified_request
    assert jsonified_request['backendService'] == 'backend_service_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Operation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Operation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.add_signed_url_key_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_add_signed_url_key_unary_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.add_signed_url_key._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('backendService', 'project', 'signedUrlKeyResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_add_signed_url_key_unary_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_add_signed_url_key') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_add_signed_url_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.AddSignedUrlKeyBackendServiceRequest.pb(compute.AddSignedUrlKeyBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.AddSignedUrlKeyBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.add_signed_url_key_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_add_signed_url_key_unary_rest_bad_request(transport: str='rest', request_type=compute.AddSignedUrlKeyBackendServiceRequest):
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.add_signed_url_key_unary(request)

def test_add_signed_url_key_unary_rest_flattened():
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'backend_service': 'sample2'}
        mock_args = dict(project='project_value', backend_service='backend_service_value', signed_url_key_resource=compute.SignedUrlKey(key_name='key_name_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.add_signed_url_key_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices/{backend_service}/addSignedUrlKey' % client.transport._host, args[1])

def test_add_signed_url_key_unary_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.add_signed_url_key_unary(compute.AddSignedUrlKeyBackendServiceRequest(), project='project_value', backend_service='backend_service_value', signed_url_key_resource=compute.SignedUrlKey(key_name='key_name_value'))

def test_add_signed_url_key_unary_rest_error():
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.AggregatedListBackendServicesRequest, dict])
def test_aggregated_list_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.BackendServiceAggregatedList(id='id_value', kind='kind_value', next_page_token='next_page_token_value', self_link='self_link_value', unreachables=['unreachables_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.BackendServiceAggregatedList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.aggregated_list(request)
    assert isinstance(response, pagers.AggregatedListPager)
    assert response.id == 'id_value'
    assert response.kind == 'kind_value'
    assert response.next_page_token == 'next_page_token_value'
    assert response.self_link == 'self_link_value'
    assert response.unreachables == ['unreachables_value']

def test_aggregated_list_rest_required_fields(request_type=compute.AggregatedListBackendServicesRequest):
    if False:
        return 10
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).aggregated_list._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).aggregated_list._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'include_all_scopes', 'max_results', 'order_by', 'page_token', 'return_partial_success'))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.BackendServiceAggregatedList()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.BackendServiceAggregatedList.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.aggregated_list(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_aggregated_list_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.aggregated_list._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess')) & set(('project',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_aggregated_list_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_aggregated_list') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_aggregated_list') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.AggregatedListBackendServicesRequest.pb(compute.AggregatedListBackendServicesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.BackendServiceAggregatedList.to_json(compute.BackendServiceAggregatedList())
        request = compute.AggregatedListBackendServicesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.BackendServiceAggregatedList()
        client.aggregated_list(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_aggregated_list_rest_bad_request(transport: str='rest', request_type=compute.AggregatedListBackendServicesRequest):
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.aggregated_list(request)

def test_aggregated_list_rest_flattened():
    if False:
        return 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.BackendServiceAggregatedList()
        sample_request = {'project': 'sample1'}
        mock_args = dict(project='project_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.BackendServiceAggregatedList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.aggregated_list(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/aggregated/backendServices' % client.transport._host, args[1])

def test_aggregated_list_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.aggregated_list(compute.AggregatedListBackendServicesRequest(), project='project_value')

def test_aggregated_list_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (compute.BackendServiceAggregatedList(items={'a': compute.BackendServicesScopedList(), 'b': compute.BackendServicesScopedList(), 'c': compute.BackendServicesScopedList()}, next_page_token='abc'), compute.BackendServiceAggregatedList(items={}, next_page_token='def'), compute.BackendServiceAggregatedList(items={'g': compute.BackendServicesScopedList()}, next_page_token='ghi'), compute.BackendServiceAggregatedList(items={'h': compute.BackendServicesScopedList(), 'i': compute.BackendServicesScopedList()}))
        response = response + response
        response = tuple((compute.BackendServiceAggregatedList.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'project': 'sample1'}
        pager = client.aggregated_list(request=sample_request)
        assert isinstance(pager.get('a'), compute.BackendServicesScopedList)
        assert pager.get('h') is None
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tuple) for i in results))
        for result in results:
            assert isinstance(result, tuple)
            assert tuple((type(t) for t in result)) == (str, compute.BackendServicesScopedList)
        assert pager.get('a') is None
        assert isinstance(pager.get('h'), compute.BackendServicesScopedList)
        pages = list(client.aggregated_list(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [compute.DeleteBackendServiceRequest, dict])
def test_delete_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete(request)
    assert isinstance(response, extended_operation.ExtendedOperation)
    assert response.client_operation_id == 'client_operation_id_value'
    assert response.creation_timestamp == 'creation_timestamp_value'
    assert response.description == 'description_value'
    assert response.end_time == 'end_time_value'
    assert response.http_error_message == 'http_error_message_value'
    assert response.http_error_status_code == 2374
    assert response.id == 205
    assert response.insert_time == 'insert_time_value'
    assert response.kind == 'kind_value'
    assert response.name == 'name_value'
    assert response.operation_group_id == 'operation_group_id_value'
    assert response.operation_type == 'operation_type_value'
    assert response.progress == 885
    assert response.region == 'region_value'
    assert response.self_link == 'self_link_value'
    assert response.start_time == 'start_time_value'
    assert response.status == compute.Operation.Status.DONE
    assert response.status_message == 'status_message_value'
    assert response.target_id == 947
    assert response.target_link == 'target_link_value'
    assert response.user == 'user_value'
    assert response.zone == 'zone_value'

def test_delete_rest_required_fields(request_type=compute.DeleteBackendServiceRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['backend_service'] = ''
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['backendService'] = 'backend_service_value'
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'backendService' in jsonified_request
    assert jsonified_request['backendService'] == 'backend_service_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Operation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'delete', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Operation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('backendService', 'project'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_delete') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_delete') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteBackendServiceRequest.pb(compute.DeleteBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_rest_bad_request(transport: str='rest', request_type=compute.DeleteBackendServiceRequest):
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete(request)

def test_delete_rest_flattened():
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'backend_service': 'sample2'}
        mock_args = dict(project='project_value', backend_service='backend_service_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices/{backend_service}' % client.transport._host, args[1])

def test_delete_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete(compute.DeleteBackendServiceRequest(), project='project_value', backend_service='backend_service_value')

def test_delete_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.DeleteBackendServiceRequest, dict])
def test_delete_unary_rest(request_type):
    if False:
        return 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_unary(request)
    assert isinstance(response, compute.Operation)

def test_delete_unary_rest_required_fields(request_type=compute.DeleteBackendServiceRequest):
    if False:
        print('Hello World!')
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['backend_service'] = ''
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['backendService'] = 'backend_service_value'
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'backendService' in jsonified_request
    assert jsonified_request['backendService'] == 'backend_service_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Operation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'delete', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Operation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_unary_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('backendService', 'project'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_unary_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_delete') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_delete') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteBackendServiceRequest.pb(compute.DeleteBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_unary_rest_bad_request(transport: str='rest', request_type=compute.DeleteBackendServiceRequest):
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_unary(request)

def test_delete_unary_rest_flattened():
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'backend_service': 'sample2'}
        mock_args = dict(project='project_value', backend_service='backend_service_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices/{backend_service}' % client.transport._host, args[1])

def test_delete_unary_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_unary(compute.DeleteBackendServiceRequest(), project='project_value', backend_service='backend_service_value')

def test_delete_unary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.DeleteSignedUrlKeyBackendServiceRequest, dict])
def test_delete_signed_url_key_rest(request_type):
    if False:
        return 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_signed_url_key(request)
    assert isinstance(response, extended_operation.ExtendedOperation)
    assert response.client_operation_id == 'client_operation_id_value'
    assert response.creation_timestamp == 'creation_timestamp_value'
    assert response.description == 'description_value'
    assert response.end_time == 'end_time_value'
    assert response.http_error_message == 'http_error_message_value'
    assert response.http_error_status_code == 2374
    assert response.id == 205
    assert response.insert_time == 'insert_time_value'
    assert response.kind == 'kind_value'
    assert response.name == 'name_value'
    assert response.operation_group_id == 'operation_group_id_value'
    assert response.operation_type == 'operation_type_value'
    assert response.progress == 885
    assert response.region == 'region_value'
    assert response.self_link == 'self_link_value'
    assert response.start_time == 'start_time_value'
    assert response.status == compute.Operation.Status.DONE
    assert response.status_message == 'status_message_value'
    assert response.target_id == 947
    assert response.target_link == 'target_link_value'
    assert response.user == 'user_value'
    assert response.zone == 'zone_value'

def test_delete_signed_url_key_rest_required_fields(request_type=compute.DeleteSignedUrlKeyBackendServiceRequest):
    if False:
        return 10
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['backend_service'] = ''
    request_init['key_name'] = ''
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'keyName' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_signed_url_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'keyName' in jsonified_request
    assert jsonified_request['keyName'] == request_init['key_name']
    jsonified_request['backendService'] = 'backend_service_value'
    jsonified_request['keyName'] = 'key_name_value'
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_signed_url_key._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('key_name', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'backendService' in jsonified_request
    assert jsonified_request['backendService'] == 'backend_service_value'
    assert 'keyName' in jsonified_request
    assert jsonified_request['keyName'] == 'key_name_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Operation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Operation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete_signed_url_key(request)
            expected_params = [('keyName', '')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_signed_url_key_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_signed_url_key._get_unset_required_fields({})
    assert set(unset_fields) == set(('keyName', 'requestId')) & set(('backendService', 'keyName', 'project'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_signed_url_key_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_delete_signed_url_key') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_delete_signed_url_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteSignedUrlKeyBackendServiceRequest.pb(compute.DeleteSignedUrlKeyBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteSignedUrlKeyBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete_signed_url_key(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_signed_url_key_rest_bad_request(transport: str='rest', request_type=compute.DeleteSignedUrlKeyBackendServiceRequest):
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_signed_url_key(request)

def test_delete_signed_url_key_rest_flattened():
    if False:
        return 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'backend_service': 'sample2'}
        mock_args = dict(project='project_value', backend_service='backend_service_value', key_name='key_name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_signed_url_key(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices/{backend_service}/deleteSignedUrlKey' % client.transport._host, args[1])

def test_delete_signed_url_key_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_signed_url_key(compute.DeleteSignedUrlKeyBackendServiceRequest(), project='project_value', backend_service='backend_service_value', key_name='key_name_value')

def test_delete_signed_url_key_rest_error():
    if False:
        return 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.DeleteSignedUrlKeyBackendServiceRequest, dict])
def test_delete_signed_url_key_unary_rest(request_type):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_signed_url_key_unary(request)
    assert isinstance(response, compute.Operation)

def test_delete_signed_url_key_unary_rest_required_fields(request_type=compute.DeleteSignedUrlKeyBackendServiceRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['backend_service'] = ''
    request_init['key_name'] = ''
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'keyName' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_signed_url_key._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'keyName' in jsonified_request
    assert jsonified_request['keyName'] == request_init['key_name']
    jsonified_request['backendService'] = 'backend_service_value'
    jsonified_request['keyName'] = 'key_name_value'
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_signed_url_key._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('key_name', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'backendService' in jsonified_request
    assert jsonified_request['backendService'] == 'backend_service_value'
    assert 'keyName' in jsonified_request
    assert jsonified_request['keyName'] == 'key_name_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Operation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Operation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.delete_signed_url_key_unary(request)
            expected_params = [('keyName', '')]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_signed_url_key_unary_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_signed_url_key._get_unset_required_fields({})
    assert set(unset_fields) == set(('keyName', 'requestId')) & set(('backendService', 'keyName', 'project'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_signed_url_key_unary_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_delete_signed_url_key') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_delete_signed_url_key') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteSignedUrlKeyBackendServiceRequest.pb(compute.DeleteSignedUrlKeyBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteSignedUrlKeyBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete_signed_url_key_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_signed_url_key_unary_rest_bad_request(transport: str='rest', request_type=compute.DeleteSignedUrlKeyBackendServiceRequest):
    if False:
        return 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_signed_url_key_unary(request)

def test_delete_signed_url_key_unary_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'backend_service': 'sample2'}
        mock_args = dict(project='project_value', backend_service='backend_service_value', key_name='key_name_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_signed_url_key_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices/{backend_service}/deleteSignedUrlKey' % client.transport._host, args[1])

def test_delete_signed_url_key_unary_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_signed_url_key_unary(compute.DeleteSignedUrlKeyBackendServiceRequest(), project='project_value', backend_service='backend_service_value', key_name='key_name_value')

def test_delete_signed_url_key_unary_rest_error():
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.GetBackendServiceRequest, dict])
def test_get_rest(request_type):
    if False:
        return 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.BackendService(affinity_cookie_ttl_sec=2432, compression_mode='compression_mode_value', creation_timestamp='creation_timestamp_value', custom_request_headers=['custom_request_headers_value'], custom_response_headers=['custom_response_headers_value'], description='description_value', edge_security_policy='edge_security_policy_value', enable_c_d_n=True, fingerprint='fingerprint_value', health_checks=['health_checks_value'], id=205, kind='kind_value', load_balancing_scheme='load_balancing_scheme_value', locality_lb_policy='locality_lb_policy_value', name='name_value', network='network_value', port=453, port_name='port_name_value', protocol='protocol_value', region='region_value', security_policy='security_policy_value', self_link='self_link_value', service_bindings=['service_bindings_value'], session_affinity='session_affinity_value', timeout_sec=1185)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.BackendService.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get(request)
    assert isinstance(response, compute.BackendService)
    assert response.affinity_cookie_ttl_sec == 2432
    assert response.compression_mode == 'compression_mode_value'
    assert response.creation_timestamp == 'creation_timestamp_value'
    assert response.custom_request_headers == ['custom_request_headers_value']
    assert response.custom_response_headers == ['custom_response_headers_value']
    assert response.description == 'description_value'
    assert response.edge_security_policy == 'edge_security_policy_value'
    assert response.enable_c_d_n is True
    assert response.fingerprint == 'fingerprint_value'
    assert response.health_checks == ['health_checks_value']
    assert response.id == 205
    assert response.kind == 'kind_value'
    assert response.load_balancing_scheme == 'load_balancing_scheme_value'
    assert response.locality_lb_policy == 'locality_lb_policy_value'
    assert response.name == 'name_value'
    assert response.network == 'network_value'
    assert response.port == 453
    assert response.port_name == 'port_name_value'
    assert response.protocol == 'protocol_value'
    assert response.region == 'region_value'
    assert response.security_policy == 'security_policy_value'
    assert response.self_link == 'self_link_value'
    assert response.service_bindings == ['service_bindings_value']
    assert response.session_affinity == 'session_affinity_value'
    assert response.timeout_sec == 1185

def test_get_rest_required_fields(request_type=compute.GetBackendServiceRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['backend_service'] = ''
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['backendService'] = 'backend_service_value'
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'backendService' in jsonified_request
    assert jsonified_request['backendService'] == 'backend_service_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.BackendService()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.BackendService.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('backendService', 'project'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_get') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_get') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.GetBackendServiceRequest.pb(compute.GetBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.BackendService.to_json(compute.BackendService())
        request = compute.GetBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.BackendService()
        client.get(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_rest_bad_request(transport: str='rest', request_type=compute.GetBackendServiceRequest):
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get(request)

def test_get_rest_flattened():
    if False:
        return 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.BackendService()
        sample_request = {'project': 'sample1', 'backend_service': 'sample2'}
        mock_args = dict(project='project_value', backend_service='backend_service_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.BackendService.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices/{backend_service}' % client.transport._host, args[1])

def test_get_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get(compute.GetBackendServiceRequest(), project='project_value', backend_service='backend_service_value')

def test_get_rest_error():
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.GetHealthBackendServiceRequest, dict])
def test_get_health_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request_init['resource_group_reference_resource'] = {'group': 'group_value'}
    test_field = compute.GetHealthBackendServiceRequest.meta.fields['resource_group_reference_resource']

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
    for (field, value) in request_init['resource_group_reference_resource'].items():
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
                for i in range(0, len(request_init['resource_group_reference_resource'][field])):
                    del request_init['resource_group_reference_resource'][field][i][subfield]
            else:
                del request_init['resource_group_reference_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.BackendServiceGroupHealth(kind='kind_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.BackendServiceGroupHealth.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_health(request)
    assert isinstance(response, compute.BackendServiceGroupHealth)
    assert response.kind == 'kind_value'

def test_get_health_rest_required_fields(request_type=compute.GetHealthBackendServiceRequest):
    if False:
        print('Hello World!')
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['backend_service'] = ''
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_health._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['backendService'] = 'backend_service_value'
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_health._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'backendService' in jsonified_request
    assert jsonified_request['backendService'] == 'backend_service_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.BackendServiceGroupHealth()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.BackendServiceGroupHealth.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_health(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_health_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_health._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('backendService', 'project', 'resourceGroupReferenceResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_health_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_get_health') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_get_health') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.GetHealthBackendServiceRequest.pb(compute.GetHealthBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.BackendServiceGroupHealth.to_json(compute.BackendServiceGroupHealth())
        request = compute.GetHealthBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.BackendServiceGroupHealth()
        client.get_health(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_health_rest_bad_request(transport: str='rest', request_type=compute.GetHealthBackendServiceRequest):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_health(request)

def test_get_health_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.BackendServiceGroupHealth()
        sample_request = {'project': 'sample1', 'backend_service': 'sample2'}
        mock_args = dict(project='project_value', backend_service='backend_service_value', resource_group_reference_resource=compute.ResourceGroupReference(group='group_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.BackendServiceGroupHealth.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_health(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices/{backend_service}/getHealth' % client.transport._host, args[1])

def test_get_health_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_health(compute.GetHealthBackendServiceRequest(), project='project_value', backend_service='backend_service_value', resource_group_reference_resource=compute.ResourceGroupReference(group='group_value'))

def test_get_health_rest_error():
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.GetIamPolicyBackendServiceRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'resource': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Policy(etag='etag_value', iam_owned=True, version=774)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Policy.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get_iam_policy(request)
    assert isinstance(response, compute.Policy)
    assert response.etag == 'etag_value'
    assert response.iam_owned is True
    assert response.version == 774

def test_get_iam_policy_rest_required_fields(request_type=compute.GetIamPolicyBackendServiceRequest):
    if False:
        print('Hello World!')
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('options_requested_policy_version',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Policy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Policy.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get_iam_policy(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_iam_policy_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(('optionsRequestedPolicyVersion',)) & set(('project', 'resource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_iam_policy_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_get_iam_policy') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_get_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.GetIamPolicyBackendServiceRequest.pb(compute.GetIamPolicyBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Policy.to_json(compute.Policy())
        request = compute.GetIamPolicyBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Policy()
        client.get_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=compute.GetIamPolicyBackendServiceRequest):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'resource': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get_iam_policy(request)

def test_get_iam_policy_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Policy()
        sample_request = {'project': 'sample1', 'resource': 'sample2'}
        mock_args = dict(project='project_value', resource='resource_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Policy.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get_iam_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices/{resource}/getIamPolicy' % client.transport._host, args[1])

def test_get_iam_policy_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_iam_policy(compute.GetIamPolicyBackendServiceRequest(), project='project_value', resource='resource_value')

def test_get_iam_policy_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.InsertBackendServiceRequest, dict])
def test_insert_rest(request_type):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1'}
    request_init['backend_service_resource'] = {'affinity_cookie_ttl_sec': 2432, 'backends': [{'balancing_mode': 'balancing_mode_value', 'capacity_scaler': 0.1575, 'description': 'description_value', 'failover': True, 'group': 'group_value', 'max_connections': 1608, 'max_connections_per_endpoint': 2990, 'max_connections_per_instance': 2978, 'max_rate': 849, 'max_rate_per_endpoint': 0.22310000000000002, 'max_rate_per_instance': 0.22190000000000001, 'max_utilization': 0.1633}], 'cdn_policy': {'bypass_cache_on_request_headers': [{'header_name': 'header_name_value'}], 'cache_key_policy': {'include_host': True, 'include_http_headers': ['include_http_headers_value1', 'include_http_headers_value2'], 'include_named_cookies': ['include_named_cookies_value1', 'include_named_cookies_value2'], 'include_protocol': True, 'include_query_string': True, 'query_string_blacklist': ['query_string_blacklist_value1', 'query_string_blacklist_value2'], 'query_string_whitelist': ['query_string_whitelist_value1', 'query_string_whitelist_value2']}, 'cache_mode': 'cache_mode_value', 'client_ttl': 1074, 'default_ttl': 1176, 'max_ttl': 761, 'negative_caching': True, 'negative_caching_policy': [{'code': 411, 'ttl': 340}], 'request_coalescing': True, 'serve_while_stale': 1813, 'signed_url_cache_max_age_sec': 2890, 'signed_url_key_names': ['signed_url_key_names_value1', 'signed_url_key_names_value2']}, 'circuit_breakers': {'max_connections': 1608, 'max_pending_requests': 2149, 'max_requests': 1313, 'max_requests_per_connection': 2902, 'max_retries': 1187}, 'compression_mode': 'compression_mode_value', 'connection_draining': {'draining_timeout_sec': 2124}, 'connection_tracking_policy': {'connection_persistence_on_unhealthy_backends': 'connection_persistence_on_unhealthy_backends_value', 'enable_strong_affinity': True, 'idle_timeout_sec': 1694, 'tracking_mode': 'tracking_mode_value'}, 'consistent_hash': {'http_cookie': {'name': 'name_value', 'path': 'path_value', 'ttl': {'nanos': 543, 'seconds': 751}}, 'http_header_name': 'http_header_name_value', 'minimum_ring_size': 1829}, 'creation_timestamp': 'creation_timestamp_value', 'custom_request_headers': ['custom_request_headers_value1', 'custom_request_headers_value2'], 'custom_response_headers': ['custom_response_headers_value1', 'custom_response_headers_value2'], 'description': 'description_value', 'edge_security_policy': 'edge_security_policy_value', 'enable_c_d_n': True, 'failover_policy': {'disable_connection_drain_on_failover': True, 'drop_traffic_if_unhealthy': True, 'failover_ratio': 0.1494}, 'fingerprint': 'fingerprint_value', 'health_checks': ['health_checks_value1', 'health_checks_value2'], 'iap': {'enabled': True, 'oauth2_client_id': 'oauth2_client_id_value', 'oauth2_client_secret': 'oauth2_client_secret_value', 'oauth2_client_secret_sha256': 'oauth2_client_secret_sha256_value'}, 'id': 205, 'kind': 'kind_value', 'load_balancing_scheme': 'load_balancing_scheme_value', 'locality_lb_policies': [{'custom_policy': {'data': 'data_value', 'name': 'name_value'}, 'policy': {'name': 'name_value'}}], 'locality_lb_policy': 'locality_lb_policy_value', 'log_config': {'enable': True, 'optional_fields': ['optional_fields_value1', 'optional_fields_value2'], 'optional_mode': 'optional_mode_value', 'sample_rate': 0.1165}, 'max_stream_duration': {}, 'metadatas': {}, 'name': 'name_value', 'network': 'network_value', 'outlier_detection': {'base_ejection_time': {}, 'consecutive_errors': 1956, 'consecutive_gateway_failure': 2880, 'enforcing_consecutive_errors': 3006, 'enforcing_consecutive_gateway_failure': 3930, 'enforcing_success_rate': 2334, 'interval': {}, 'max_ejection_percent': 2118, 'success_rate_minimum_hosts': 2799, 'success_rate_request_volume': 2915, 'success_rate_stdev_factor': 2663}, 'port': 453, 'port_name': 'port_name_value', 'protocol': 'protocol_value', 'region': 'region_value', 'security_policy': 'security_policy_value', 'security_settings': {'client_tls_policy': 'client_tls_policy_value', 'subject_alt_names': ['subject_alt_names_value1', 'subject_alt_names_value2']}, 'self_link': 'self_link_value', 'service_bindings': ['service_bindings_value1', 'service_bindings_value2'], 'session_affinity': 'session_affinity_value', 'subsetting': {'policy': 'policy_value'}, 'timeout_sec': 1185}
    test_field = compute.InsertBackendServiceRequest.meta.fields['backend_service_resource']

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
    for (field, value) in request_init['backend_service_resource'].items():
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
                for i in range(0, len(request_init['backend_service_resource'][field])):
                    del request_init['backend_service_resource'][field][i][subfield]
            else:
                del request_init['backend_service_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.insert(request)
    assert isinstance(response, extended_operation.ExtendedOperation)
    assert response.client_operation_id == 'client_operation_id_value'
    assert response.creation_timestamp == 'creation_timestamp_value'
    assert response.description == 'description_value'
    assert response.end_time == 'end_time_value'
    assert response.http_error_message == 'http_error_message_value'
    assert response.http_error_status_code == 2374
    assert response.id == 205
    assert response.insert_time == 'insert_time_value'
    assert response.kind == 'kind_value'
    assert response.name == 'name_value'
    assert response.operation_group_id == 'operation_group_id_value'
    assert response.operation_type == 'operation_type_value'
    assert response.progress == 885
    assert response.region == 'region_value'
    assert response.self_link == 'self_link_value'
    assert response.start_time == 'start_time_value'
    assert response.status == compute.Operation.Status.DONE
    assert response.status_message == 'status_message_value'
    assert response.target_id == 947
    assert response.target_link == 'target_link_value'
    assert response.user == 'user_value'
    assert response.zone == 'zone_value'

def test_insert_rest_required_fields(request_type=compute.InsertBackendServiceRequest):
    if False:
        print('Hello World!')
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).insert._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).insert._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Operation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Operation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.insert(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_insert_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.insert._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('backendServiceResource', 'project'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_insert_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_insert') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_insert') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.InsertBackendServiceRequest.pb(compute.InsertBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.InsertBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.insert(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_insert_rest_bad_request(transport: str='rest', request_type=compute.InsertBackendServiceRequest):
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.insert(request)

def test_insert_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1'}
        mock_args = dict(project='project_value', backend_service_resource=compute.BackendService(affinity_cookie_ttl_sec=2432))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.insert(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices' % client.transport._host, args[1])

def test_insert_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.insert(compute.InsertBackendServiceRequest(), project='project_value', backend_service_resource=compute.BackendService(affinity_cookie_ttl_sec=2432))

def test_insert_rest_error():
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.InsertBackendServiceRequest, dict])
def test_insert_unary_rest(request_type):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1'}
    request_init['backend_service_resource'] = {'affinity_cookie_ttl_sec': 2432, 'backends': [{'balancing_mode': 'balancing_mode_value', 'capacity_scaler': 0.1575, 'description': 'description_value', 'failover': True, 'group': 'group_value', 'max_connections': 1608, 'max_connections_per_endpoint': 2990, 'max_connections_per_instance': 2978, 'max_rate': 849, 'max_rate_per_endpoint': 0.22310000000000002, 'max_rate_per_instance': 0.22190000000000001, 'max_utilization': 0.1633}], 'cdn_policy': {'bypass_cache_on_request_headers': [{'header_name': 'header_name_value'}], 'cache_key_policy': {'include_host': True, 'include_http_headers': ['include_http_headers_value1', 'include_http_headers_value2'], 'include_named_cookies': ['include_named_cookies_value1', 'include_named_cookies_value2'], 'include_protocol': True, 'include_query_string': True, 'query_string_blacklist': ['query_string_blacklist_value1', 'query_string_blacklist_value2'], 'query_string_whitelist': ['query_string_whitelist_value1', 'query_string_whitelist_value2']}, 'cache_mode': 'cache_mode_value', 'client_ttl': 1074, 'default_ttl': 1176, 'max_ttl': 761, 'negative_caching': True, 'negative_caching_policy': [{'code': 411, 'ttl': 340}], 'request_coalescing': True, 'serve_while_stale': 1813, 'signed_url_cache_max_age_sec': 2890, 'signed_url_key_names': ['signed_url_key_names_value1', 'signed_url_key_names_value2']}, 'circuit_breakers': {'max_connections': 1608, 'max_pending_requests': 2149, 'max_requests': 1313, 'max_requests_per_connection': 2902, 'max_retries': 1187}, 'compression_mode': 'compression_mode_value', 'connection_draining': {'draining_timeout_sec': 2124}, 'connection_tracking_policy': {'connection_persistence_on_unhealthy_backends': 'connection_persistence_on_unhealthy_backends_value', 'enable_strong_affinity': True, 'idle_timeout_sec': 1694, 'tracking_mode': 'tracking_mode_value'}, 'consistent_hash': {'http_cookie': {'name': 'name_value', 'path': 'path_value', 'ttl': {'nanos': 543, 'seconds': 751}}, 'http_header_name': 'http_header_name_value', 'minimum_ring_size': 1829}, 'creation_timestamp': 'creation_timestamp_value', 'custom_request_headers': ['custom_request_headers_value1', 'custom_request_headers_value2'], 'custom_response_headers': ['custom_response_headers_value1', 'custom_response_headers_value2'], 'description': 'description_value', 'edge_security_policy': 'edge_security_policy_value', 'enable_c_d_n': True, 'failover_policy': {'disable_connection_drain_on_failover': True, 'drop_traffic_if_unhealthy': True, 'failover_ratio': 0.1494}, 'fingerprint': 'fingerprint_value', 'health_checks': ['health_checks_value1', 'health_checks_value2'], 'iap': {'enabled': True, 'oauth2_client_id': 'oauth2_client_id_value', 'oauth2_client_secret': 'oauth2_client_secret_value', 'oauth2_client_secret_sha256': 'oauth2_client_secret_sha256_value'}, 'id': 205, 'kind': 'kind_value', 'load_balancing_scheme': 'load_balancing_scheme_value', 'locality_lb_policies': [{'custom_policy': {'data': 'data_value', 'name': 'name_value'}, 'policy': {'name': 'name_value'}}], 'locality_lb_policy': 'locality_lb_policy_value', 'log_config': {'enable': True, 'optional_fields': ['optional_fields_value1', 'optional_fields_value2'], 'optional_mode': 'optional_mode_value', 'sample_rate': 0.1165}, 'max_stream_duration': {}, 'metadatas': {}, 'name': 'name_value', 'network': 'network_value', 'outlier_detection': {'base_ejection_time': {}, 'consecutive_errors': 1956, 'consecutive_gateway_failure': 2880, 'enforcing_consecutive_errors': 3006, 'enforcing_consecutive_gateway_failure': 3930, 'enforcing_success_rate': 2334, 'interval': {}, 'max_ejection_percent': 2118, 'success_rate_minimum_hosts': 2799, 'success_rate_request_volume': 2915, 'success_rate_stdev_factor': 2663}, 'port': 453, 'port_name': 'port_name_value', 'protocol': 'protocol_value', 'region': 'region_value', 'security_policy': 'security_policy_value', 'security_settings': {'client_tls_policy': 'client_tls_policy_value', 'subject_alt_names': ['subject_alt_names_value1', 'subject_alt_names_value2']}, 'self_link': 'self_link_value', 'service_bindings': ['service_bindings_value1', 'service_bindings_value2'], 'session_affinity': 'session_affinity_value', 'subsetting': {'policy': 'policy_value'}, 'timeout_sec': 1185}
    test_field = compute.InsertBackendServiceRequest.meta.fields['backend_service_resource']

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
    for (field, value) in request_init['backend_service_resource'].items():
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
                for i in range(0, len(request_init['backend_service_resource'][field])):
                    del request_init['backend_service_resource'][field][i][subfield]
            else:
                del request_init['backend_service_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.insert_unary(request)
    assert isinstance(response, compute.Operation)

def test_insert_unary_rest_required_fields(request_type=compute.InsertBackendServiceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).insert._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).insert._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Operation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Operation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.insert_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_insert_unary_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.insert._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('backendServiceResource', 'project'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_insert_unary_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_insert') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_insert') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.InsertBackendServiceRequest.pb(compute.InsertBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.InsertBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.insert_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_insert_unary_rest_bad_request(transport: str='rest', request_type=compute.InsertBackendServiceRequest):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.insert_unary(request)

def test_insert_unary_rest_flattened():
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1'}
        mock_args = dict(project='project_value', backend_service_resource=compute.BackendService(affinity_cookie_ttl_sec=2432))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.insert_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices' % client.transport._host, args[1])

def test_insert_unary_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.insert_unary(compute.InsertBackendServiceRequest(), project='project_value', backend_service_resource=compute.BackendService(affinity_cookie_ttl_sec=2432))

def test_insert_unary_rest_error():
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.ListBackendServicesRequest, dict])
def test_list_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.BackendServiceList(id='id_value', kind='kind_value', next_page_token='next_page_token_value', self_link='self_link_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.BackendServiceList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list(request)
    assert isinstance(response, pagers.ListPager)
    assert response.id == 'id_value'
    assert response.kind == 'kind_value'
    assert response.next_page_token == 'next_page_token_value'
    assert response.self_link == 'self_link_value'

def test_list_rest_required_fields(request_type=compute.ListBackendServicesRequest):
    if False:
        return 10
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'max_results', 'order_by', 'page_token', 'return_partial_success'))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.BackendServiceList()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.BackendServiceList.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess')) & set(('project',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_list') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_list') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.ListBackendServicesRequest.pb(compute.ListBackendServicesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.BackendServiceList.to_json(compute.BackendServiceList())
        request = compute.ListBackendServicesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.BackendServiceList()
        client.list(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_rest_bad_request(transport: str='rest', request_type=compute.ListBackendServicesRequest):
    if False:
        return 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list(request)

def test_list_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.BackendServiceList()
        sample_request = {'project': 'sample1'}
        mock_args = dict(project='project_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.BackendServiceList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices' % client.transport._host, args[1])

def test_list_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list(compute.ListBackendServicesRequest(), project='project_value')

def test_list_rest_pager(transport: str='rest'):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (compute.BackendServiceList(items=[compute.BackendService(), compute.BackendService(), compute.BackendService()], next_page_token='abc'), compute.BackendServiceList(items=[], next_page_token='def'), compute.BackendServiceList(items=[compute.BackendService()], next_page_token='ghi'), compute.BackendServiceList(items=[compute.BackendService(), compute.BackendService()]))
        response = response + response
        response = tuple((compute.BackendServiceList.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'project': 'sample1'}
        pager = client.list(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, compute.BackendService) for i in results))
        pages = list(client.list(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [compute.PatchBackendServiceRequest, dict])
def test_patch_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request_init['backend_service_resource'] = {'affinity_cookie_ttl_sec': 2432, 'backends': [{'balancing_mode': 'balancing_mode_value', 'capacity_scaler': 0.1575, 'description': 'description_value', 'failover': True, 'group': 'group_value', 'max_connections': 1608, 'max_connections_per_endpoint': 2990, 'max_connections_per_instance': 2978, 'max_rate': 849, 'max_rate_per_endpoint': 0.22310000000000002, 'max_rate_per_instance': 0.22190000000000001, 'max_utilization': 0.1633}], 'cdn_policy': {'bypass_cache_on_request_headers': [{'header_name': 'header_name_value'}], 'cache_key_policy': {'include_host': True, 'include_http_headers': ['include_http_headers_value1', 'include_http_headers_value2'], 'include_named_cookies': ['include_named_cookies_value1', 'include_named_cookies_value2'], 'include_protocol': True, 'include_query_string': True, 'query_string_blacklist': ['query_string_blacklist_value1', 'query_string_blacklist_value2'], 'query_string_whitelist': ['query_string_whitelist_value1', 'query_string_whitelist_value2']}, 'cache_mode': 'cache_mode_value', 'client_ttl': 1074, 'default_ttl': 1176, 'max_ttl': 761, 'negative_caching': True, 'negative_caching_policy': [{'code': 411, 'ttl': 340}], 'request_coalescing': True, 'serve_while_stale': 1813, 'signed_url_cache_max_age_sec': 2890, 'signed_url_key_names': ['signed_url_key_names_value1', 'signed_url_key_names_value2']}, 'circuit_breakers': {'max_connections': 1608, 'max_pending_requests': 2149, 'max_requests': 1313, 'max_requests_per_connection': 2902, 'max_retries': 1187}, 'compression_mode': 'compression_mode_value', 'connection_draining': {'draining_timeout_sec': 2124}, 'connection_tracking_policy': {'connection_persistence_on_unhealthy_backends': 'connection_persistence_on_unhealthy_backends_value', 'enable_strong_affinity': True, 'idle_timeout_sec': 1694, 'tracking_mode': 'tracking_mode_value'}, 'consistent_hash': {'http_cookie': {'name': 'name_value', 'path': 'path_value', 'ttl': {'nanos': 543, 'seconds': 751}}, 'http_header_name': 'http_header_name_value', 'minimum_ring_size': 1829}, 'creation_timestamp': 'creation_timestamp_value', 'custom_request_headers': ['custom_request_headers_value1', 'custom_request_headers_value2'], 'custom_response_headers': ['custom_response_headers_value1', 'custom_response_headers_value2'], 'description': 'description_value', 'edge_security_policy': 'edge_security_policy_value', 'enable_c_d_n': True, 'failover_policy': {'disable_connection_drain_on_failover': True, 'drop_traffic_if_unhealthy': True, 'failover_ratio': 0.1494}, 'fingerprint': 'fingerprint_value', 'health_checks': ['health_checks_value1', 'health_checks_value2'], 'iap': {'enabled': True, 'oauth2_client_id': 'oauth2_client_id_value', 'oauth2_client_secret': 'oauth2_client_secret_value', 'oauth2_client_secret_sha256': 'oauth2_client_secret_sha256_value'}, 'id': 205, 'kind': 'kind_value', 'load_balancing_scheme': 'load_balancing_scheme_value', 'locality_lb_policies': [{'custom_policy': {'data': 'data_value', 'name': 'name_value'}, 'policy': {'name': 'name_value'}}], 'locality_lb_policy': 'locality_lb_policy_value', 'log_config': {'enable': True, 'optional_fields': ['optional_fields_value1', 'optional_fields_value2'], 'optional_mode': 'optional_mode_value', 'sample_rate': 0.1165}, 'max_stream_duration': {}, 'metadatas': {}, 'name': 'name_value', 'network': 'network_value', 'outlier_detection': {'base_ejection_time': {}, 'consecutive_errors': 1956, 'consecutive_gateway_failure': 2880, 'enforcing_consecutive_errors': 3006, 'enforcing_consecutive_gateway_failure': 3930, 'enforcing_success_rate': 2334, 'interval': {}, 'max_ejection_percent': 2118, 'success_rate_minimum_hosts': 2799, 'success_rate_request_volume': 2915, 'success_rate_stdev_factor': 2663}, 'port': 453, 'port_name': 'port_name_value', 'protocol': 'protocol_value', 'region': 'region_value', 'security_policy': 'security_policy_value', 'security_settings': {'client_tls_policy': 'client_tls_policy_value', 'subject_alt_names': ['subject_alt_names_value1', 'subject_alt_names_value2']}, 'self_link': 'self_link_value', 'service_bindings': ['service_bindings_value1', 'service_bindings_value2'], 'session_affinity': 'session_affinity_value', 'subsetting': {'policy': 'policy_value'}, 'timeout_sec': 1185}
    test_field = compute.PatchBackendServiceRequest.meta.fields['backend_service_resource']

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
    for (field, value) in request_init['backend_service_resource'].items():
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
                for i in range(0, len(request_init['backend_service_resource'][field])):
                    del request_init['backend_service_resource'][field][i][subfield]
            else:
                del request_init['backend_service_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.patch(request)
    assert isinstance(response, extended_operation.ExtendedOperation)
    assert response.client_operation_id == 'client_operation_id_value'
    assert response.creation_timestamp == 'creation_timestamp_value'
    assert response.description == 'description_value'
    assert response.end_time == 'end_time_value'
    assert response.http_error_message == 'http_error_message_value'
    assert response.http_error_status_code == 2374
    assert response.id == 205
    assert response.insert_time == 'insert_time_value'
    assert response.kind == 'kind_value'
    assert response.name == 'name_value'
    assert response.operation_group_id == 'operation_group_id_value'
    assert response.operation_type == 'operation_type_value'
    assert response.progress == 885
    assert response.region == 'region_value'
    assert response.self_link == 'self_link_value'
    assert response.start_time == 'start_time_value'
    assert response.status == compute.Operation.Status.DONE
    assert response.status_message == 'status_message_value'
    assert response.target_id == 947
    assert response.target_link == 'target_link_value'
    assert response.user == 'user_value'
    assert response.zone == 'zone_value'

def test_patch_rest_required_fields(request_type=compute.PatchBackendServiceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['backend_service'] = ''
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['backendService'] = 'backend_service_value'
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'backendService' in jsonified_request
    assert jsonified_request['backendService'] == 'backend_service_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Operation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Operation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.patch(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_patch_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.patch._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('backendService', 'backendServiceResource', 'project'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_patch_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_patch') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_patch') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.PatchBackendServiceRequest.pb(compute.PatchBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.PatchBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.patch(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_patch_rest_bad_request(transport: str='rest', request_type=compute.PatchBackendServiceRequest):
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.patch(request)

def test_patch_rest_flattened():
    if False:
        return 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'backend_service': 'sample2'}
        mock_args = dict(project='project_value', backend_service='backend_service_value', backend_service_resource=compute.BackendService(affinity_cookie_ttl_sec=2432))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.patch(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices/{backend_service}' % client.transport._host, args[1])

def test_patch_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.patch(compute.PatchBackendServiceRequest(), project='project_value', backend_service='backend_service_value', backend_service_resource=compute.BackendService(affinity_cookie_ttl_sec=2432))

def test_patch_rest_error():
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.PatchBackendServiceRequest, dict])
def test_patch_unary_rest(request_type):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request_init['backend_service_resource'] = {'affinity_cookie_ttl_sec': 2432, 'backends': [{'balancing_mode': 'balancing_mode_value', 'capacity_scaler': 0.1575, 'description': 'description_value', 'failover': True, 'group': 'group_value', 'max_connections': 1608, 'max_connections_per_endpoint': 2990, 'max_connections_per_instance': 2978, 'max_rate': 849, 'max_rate_per_endpoint': 0.22310000000000002, 'max_rate_per_instance': 0.22190000000000001, 'max_utilization': 0.1633}], 'cdn_policy': {'bypass_cache_on_request_headers': [{'header_name': 'header_name_value'}], 'cache_key_policy': {'include_host': True, 'include_http_headers': ['include_http_headers_value1', 'include_http_headers_value2'], 'include_named_cookies': ['include_named_cookies_value1', 'include_named_cookies_value2'], 'include_protocol': True, 'include_query_string': True, 'query_string_blacklist': ['query_string_blacklist_value1', 'query_string_blacklist_value2'], 'query_string_whitelist': ['query_string_whitelist_value1', 'query_string_whitelist_value2']}, 'cache_mode': 'cache_mode_value', 'client_ttl': 1074, 'default_ttl': 1176, 'max_ttl': 761, 'negative_caching': True, 'negative_caching_policy': [{'code': 411, 'ttl': 340}], 'request_coalescing': True, 'serve_while_stale': 1813, 'signed_url_cache_max_age_sec': 2890, 'signed_url_key_names': ['signed_url_key_names_value1', 'signed_url_key_names_value2']}, 'circuit_breakers': {'max_connections': 1608, 'max_pending_requests': 2149, 'max_requests': 1313, 'max_requests_per_connection': 2902, 'max_retries': 1187}, 'compression_mode': 'compression_mode_value', 'connection_draining': {'draining_timeout_sec': 2124}, 'connection_tracking_policy': {'connection_persistence_on_unhealthy_backends': 'connection_persistence_on_unhealthy_backends_value', 'enable_strong_affinity': True, 'idle_timeout_sec': 1694, 'tracking_mode': 'tracking_mode_value'}, 'consistent_hash': {'http_cookie': {'name': 'name_value', 'path': 'path_value', 'ttl': {'nanos': 543, 'seconds': 751}}, 'http_header_name': 'http_header_name_value', 'minimum_ring_size': 1829}, 'creation_timestamp': 'creation_timestamp_value', 'custom_request_headers': ['custom_request_headers_value1', 'custom_request_headers_value2'], 'custom_response_headers': ['custom_response_headers_value1', 'custom_response_headers_value2'], 'description': 'description_value', 'edge_security_policy': 'edge_security_policy_value', 'enable_c_d_n': True, 'failover_policy': {'disable_connection_drain_on_failover': True, 'drop_traffic_if_unhealthy': True, 'failover_ratio': 0.1494}, 'fingerprint': 'fingerprint_value', 'health_checks': ['health_checks_value1', 'health_checks_value2'], 'iap': {'enabled': True, 'oauth2_client_id': 'oauth2_client_id_value', 'oauth2_client_secret': 'oauth2_client_secret_value', 'oauth2_client_secret_sha256': 'oauth2_client_secret_sha256_value'}, 'id': 205, 'kind': 'kind_value', 'load_balancing_scheme': 'load_balancing_scheme_value', 'locality_lb_policies': [{'custom_policy': {'data': 'data_value', 'name': 'name_value'}, 'policy': {'name': 'name_value'}}], 'locality_lb_policy': 'locality_lb_policy_value', 'log_config': {'enable': True, 'optional_fields': ['optional_fields_value1', 'optional_fields_value2'], 'optional_mode': 'optional_mode_value', 'sample_rate': 0.1165}, 'max_stream_duration': {}, 'metadatas': {}, 'name': 'name_value', 'network': 'network_value', 'outlier_detection': {'base_ejection_time': {}, 'consecutive_errors': 1956, 'consecutive_gateway_failure': 2880, 'enforcing_consecutive_errors': 3006, 'enforcing_consecutive_gateway_failure': 3930, 'enforcing_success_rate': 2334, 'interval': {}, 'max_ejection_percent': 2118, 'success_rate_minimum_hosts': 2799, 'success_rate_request_volume': 2915, 'success_rate_stdev_factor': 2663}, 'port': 453, 'port_name': 'port_name_value', 'protocol': 'protocol_value', 'region': 'region_value', 'security_policy': 'security_policy_value', 'security_settings': {'client_tls_policy': 'client_tls_policy_value', 'subject_alt_names': ['subject_alt_names_value1', 'subject_alt_names_value2']}, 'self_link': 'self_link_value', 'service_bindings': ['service_bindings_value1', 'service_bindings_value2'], 'session_affinity': 'session_affinity_value', 'subsetting': {'policy': 'policy_value'}, 'timeout_sec': 1185}
    test_field = compute.PatchBackendServiceRequest.meta.fields['backend_service_resource']

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
    for (field, value) in request_init['backend_service_resource'].items():
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
                for i in range(0, len(request_init['backend_service_resource'][field])):
                    del request_init['backend_service_resource'][field][i][subfield]
            else:
                del request_init['backend_service_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.patch_unary(request)
    assert isinstance(response, compute.Operation)

def test_patch_unary_rest_required_fields(request_type=compute.PatchBackendServiceRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['backend_service'] = ''
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['backendService'] = 'backend_service_value'
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'backendService' in jsonified_request
    assert jsonified_request['backendService'] == 'backend_service_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Operation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'patch', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Operation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.patch_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_patch_unary_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.patch._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('backendService', 'backendServiceResource', 'project'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_patch_unary_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_patch') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_patch') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.PatchBackendServiceRequest.pb(compute.PatchBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.PatchBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.patch_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_patch_unary_rest_bad_request(transport: str='rest', request_type=compute.PatchBackendServiceRequest):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.patch_unary(request)

def test_patch_unary_rest_flattened():
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'backend_service': 'sample2'}
        mock_args = dict(project='project_value', backend_service='backend_service_value', backend_service_resource=compute.BackendService(affinity_cookie_ttl_sec=2432))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.patch_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices/{backend_service}' % client.transport._host, args[1])

def test_patch_unary_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.patch_unary(compute.PatchBackendServiceRequest(), project='project_value', backend_service='backend_service_value', backend_service_resource=compute.BackendService(affinity_cookie_ttl_sec=2432))

def test_patch_unary_rest_error():
    if False:
        return 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetEdgeSecurityPolicyBackendServiceRequest, dict])
def test_set_edge_security_policy_rest(request_type):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request_init['security_policy_reference_resource'] = {'security_policy': 'security_policy_value'}
    test_field = compute.SetEdgeSecurityPolicyBackendServiceRequest.meta.fields['security_policy_reference_resource']

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
    for (field, value) in request_init['security_policy_reference_resource'].items():
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
                for i in range(0, len(request_init['security_policy_reference_resource'][field])):
                    del request_init['security_policy_reference_resource'][field][i][subfield]
            else:
                del request_init['security_policy_reference_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_edge_security_policy(request)
    assert isinstance(response, extended_operation.ExtendedOperation)
    assert response.client_operation_id == 'client_operation_id_value'
    assert response.creation_timestamp == 'creation_timestamp_value'
    assert response.description == 'description_value'
    assert response.end_time == 'end_time_value'
    assert response.http_error_message == 'http_error_message_value'
    assert response.http_error_status_code == 2374
    assert response.id == 205
    assert response.insert_time == 'insert_time_value'
    assert response.kind == 'kind_value'
    assert response.name == 'name_value'
    assert response.operation_group_id == 'operation_group_id_value'
    assert response.operation_type == 'operation_type_value'
    assert response.progress == 885
    assert response.region == 'region_value'
    assert response.self_link == 'self_link_value'
    assert response.start_time == 'start_time_value'
    assert response.status == compute.Operation.Status.DONE
    assert response.status_message == 'status_message_value'
    assert response.target_id == 947
    assert response.target_link == 'target_link_value'
    assert response.user == 'user_value'
    assert response.zone == 'zone_value'

def test_set_edge_security_policy_rest_required_fields(request_type=compute.SetEdgeSecurityPolicyBackendServiceRequest):
    if False:
        print('Hello World!')
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['backend_service'] = ''
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_edge_security_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['backendService'] = 'backend_service_value'
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_edge_security_policy._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'backendService' in jsonified_request
    assert jsonified_request['backendService'] == 'backend_service_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Operation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Operation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.set_edge_security_policy(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_edge_security_policy_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_edge_security_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('backendService', 'project', 'securityPolicyReferenceResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_edge_security_policy_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_set_edge_security_policy') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_set_edge_security_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetEdgeSecurityPolicyBackendServiceRequest.pb(compute.SetEdgeSecurityPolicyBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SetEdgeSecurityPolicyBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.set_edge_security_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_edge_security_policy_rest_bad_request(transport: str='rest', request_type=compute.SetEdgeSecurityPolicyBackendServiceRequest):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_edge_security_policy(request)

def test_set_edge_security_policy_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'backend_service': 'sample2'}
        mock_args = dict(project='project_value', backend_service='backend_service_value', security_policy_reference_resource=compute.SecurityPolicyReference(security_policy='security_policy_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_edge_security_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices/{backend_service}/setEdgeSecurityPolicy' % client.transport._host, args[1])

def test_set_edge_security_policy_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_edge_security_policy(compute.SetEdgeSecurityPolicyBackendServiceRequest(), project='project_value', backend_service='backend_service_value', security_policy_reference_resource=compute.SecurityPolicyReference(security_policy='security_policy_value'))

def test_set_edge_security_policy_rest_error():
    if False:
        return 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetEdgeSecurityPolicyBackendServiceRequest, dict])
def test_set_edge_security_policy_unary_rest(request_type):
    if False:
        return 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request_init['security_policy_reference_resource'] = {'security_policy': 'security_policy_value'}
    test_field = compute.SetEdgeSecurityPolicyBackendServiceRequest.meta.fields['security_policy_reference_resource']

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
    for (field, value) in request_init['security_policy_reference_resource'].items():
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
                for i in range(0, len(request_init['security_policy_reference_resource'][field])):
                    del request_init['security_policy_reference_resource'][field][i][subfield]
            else:
                del request_init['security_policy_reference_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_edge_security_policy_unary(request)
    assert isinstance(response, compute.Operation)

def test_set_edge_security_policy_unary_rest_required_fields(request_type=compute.SetEdgeSecurityPolicyBackendServiceRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['backend_service'] = ''
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_edge_security_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['backendService'] = 'backend_service_value'
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_edge_security_policy._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'backendService' in jsonified_request
    assert jsonified_request['backendService'] == 'backend_service_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Operation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Operation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.set_edge_security_policy_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_edge_security_policy_unary_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_edge_security_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('backendService', 'project', 'securityPolicyReferenceResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_edge_security_policy_unary_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_set_edge_security_policy') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_set_edge_security_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetEdgeSecurityPolicyBackendServiceRequest.pb(compute.SetEdgeSecurityPolicyBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SetEdgeSecurityPolicyBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.set_edge_security_policy_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_edge_security_policy_unary_rest_bad_request(transport: str='rest', request_type=compute.SetEdgeSecurityPolicyBackendServiceRequest):
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_edge_security_policy_unary(request)

def test_set_edge_security_policy_unary_rest_flattened():
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'backend_service': 'sample2'}
        mock_args = dict(project='project_value', backend_service='backend_service_value', security_policy_reference_resource=compute.SecurityPolicyReference(security_policy='security_policy_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_edge_security_policy_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices/{backend_service}/setEdgeSecurityPolicy' % client.transport._host, args[1])

def test_set_edge_security_policy_unary_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_edge_security_policy_unary(compute.SetEdgeSecurityPolicyBackendServiceRequest(), project='project_value', backend_service='backend_service_value', security_policy_reference_resource=compute.SecurityPolicyReference(security_policy='security_policy_value'))

def test_set_edge_security_policy_unary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetIamPolicyBackendServiceRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'resource': 'sample2'}
    request_init['global_set_policy_request_resource'] = {'bindings': [{'binding_id': 'binding_id_value', 'condition': {'description': 'description_value', 'expression': 'expression_value', 'location': 'location_value', 'title': 'title_value'}, 'members': ['members_value1', 'members_value2'], 'role': 'role_value'}], 'etag': 'etag_value', 'policy': {'audit_configs': [{'audit_log_configs': [{'exempted_members': ['exempted_members_value1', 'exempted_members_value2'], 'ignore_child_exemptions': True, 'log_type': 'log_type_value'}], 'exempted_members': ['exempted_members_value1', 'exempted_members_value2'], 'service': 'service_value'}], 'bindings': {}, 'etag': 'etag_value', 'iam_owned': True, 'rules': [{'action': 'action_value', 'conditions': [{'iam': 'iam_value', 'op': 'op_value', 'svc': 'svc_value', 'sys': 'sys_value', 'values': ['values_value1', 'values_value2']}], 'description': 'description_value', 'ins': ['ins_value1', 'ins_value2'], 'log_configs': [{'cloud_audit': {'authorization_logging_options': {'permission_type': 'permission_type_value'}, 'log_name': 'log_name_value'}, 'counter': {'custom_fields': [{'name': 'name_value', 'value': 'value_value'}], 'field': 'field_value', 'metric': 'metric_value'}, 'data_access': {'log_mode': 'log_mode_value'}}], 'not_ins': ['not_ins_value1', 'not_ins_value2'], 'permissions': ['permissions_value1', 'permissions_value2']}], 'version': 774}}
    test_field = compute.SetIamPolicyBackendServiceRequest.meta.fields['global_set_policy_request_resource']

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
    for (field, value) in request_init['global_set_policy_request_resource'].items():
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
                for i in range(0, len(request_init['global_set_policy_request_resource'][field])):
                    del request_init['global_set_policy_request_resource'][field][i][subfield]
            else:
                del request_init['global_set_policy_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Policy(etag='etag_value', iam_owned=True, version=774)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Policy.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_iam_policy(request)
    assert isinstance(response, compute.Policy)
    assert response.etag == 'etag_value'
    assert response.iam_owned is True
    assert response.version == 774

def test_set_iam_policy_rest_required_fields(request_type=compute.SetIamPolicyBackendServiceRequest):
    if False:
        return 10
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Policy()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Policy.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.set_iam_policy(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_iam_policy_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('globalSetPolicyRequestResource', 'project', 'resource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_iam_policy_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_set_iam_policy') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_set_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetIamPolicyBackendServiceRequest.pb(compute.SetIamPolicyBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Policy.to_json(compute.Policy())
        request = compute.SetIamPolicyBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Policy()
        client.set_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_iam_policy_rest_bad_request(transport: str='rest', request_type=compute.SetIamPolicyBackendServiceRequest):
    if False:
        return 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'resource': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

def test_set_iam_policy_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Policy()
        sample_request = {'project': 'sample1', 'resource': 'sample2'}
        mock_args = dict(project='project_value', resource='resource_value', global_set_policy_request_resource=compute.GlobalSetPolicyRequest(bindings=[compute.Binding(binding_id='binding_id_value')]))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Policy.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_iam_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices/{resource}/setIamPolicy' % client.transport._host, args[1])

def test_set_iam_policy_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_iam_policy(compute.SetIamPolicyBackendServiceRequest(), project='project_value', resource='resource_value', global_set_policy_request_resource=compute.GlobalSetPolicyRequest(bindings=[compute.Binding(binding_id='binding_id_value')]))

def test_set_iam_policy_rest_error():
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetSecurityPolicyBackendServiceRequest, dict])
def test_set_security_policy_rest(request_type):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request_init['security_policy_reference_resource'] = {'security_policy': 'security_policy_value'}
    test_field = compute.SetSecurityPolicyBackendServiceRequest.meta.fields['security_policy_reference_resource']

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
    for (field, value) in request_init['security_policy_reference_resource'].items():
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
                for i in range(0, len(request_init['security_policy_reference_resource'][field])):
                    del request_init['security_policy_reference_resource'][field][i][subfield]
            else:
                del request_init['security_policy_reference_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_security_policy(request)
    assert isinstance(response, extended_operation.ExtendedOperation)
    assert response.client_operation_id == 'client_operation_id_value'
    assert response.creation_timestamp == 'creation_timestamp_value'
    assert response.description == 'description_value'
    assert response.end_time == 'end_time_value'
    assert response.http_error_message == 'http_error_message_value'
    assert response.http_error_status_code == 2374
    assert response.id == 205
    assert response.insert_time == 'insert_time_value'
    assert response.kind == 'kind_value'
    assert response.name == 'name_value'
    assert response.operation_group_id == 'operation_group_id_value'
    assert response.operation_type == 'operation_type_value'
    assert response.progress == 885
    assert response.region == 'region_value'
    assert response.self_link == 'self_link_value'
    assert response.start_time == 'start_time_value'
    assert response.status == compute.Operation.Status.DONE
    assert response.status_message == 'status_message_value'
    assert response.target_id == 947
    assert response.target_link == 'target_link_value'
    assert response.user == 'user_value'
    assert response.zone == 'zone_value'

def test_set_security_policy_rest_required_fields(request_type=compute.SetSecurityPolicyBackendServiceRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['backend_service'] = ''
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_security_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['backendService'] = 'backend_service_value'
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_security_policy._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'backendService' in jsonified_request
    assert jsonified_request['backendService'] == 'backend_service_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Operation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Operation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.set_security_policy(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_security_policy_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_security_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('backendService', 'project', 'securityPolicyReferenceResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_security_policy_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_set_security_policy') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_set_security_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetSecurityPolicyBackendServiceRequest.pb(compute.SetSecurityPolicyBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SetSecurityPolicyBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.set_security_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_security_policy_rest_bad_request(transport: str='rest', request_type=compute.SetSecurityPolicyBackendServiceRequest):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_security_policy(request)

def test_set_security_policy_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'backend_service': 'sample2'}
        mock_args = dict(project='project_value', backend_service='backend_service_value', security_policy_reference_resource=compute.SecurityPolicyReference(security_policy='security_policy_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_security_policy(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices/{backend_service}/setSecurityPolicy' % client.transport._host, args[1])

def test_set_security_policy_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_security_policy(compute.SetSecurityPolicyBackendServiceRequest(), project='project_value', backend_service='backend_service_value', security_policy_reference_resource=compute.SecurityPolicyReference(security_policy='security_policy_value'))

def test_set_security_policy_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetSecurityPolicyBackendServiceRequest, dict])
def test_set_security_policy_unary_rest(request_type):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request_init['security_policy_reference_resource'] = {'security_policy': 'security_policy_value'}
    test_field = compute.SetSecurityPolicyBackendServiceRequest.meta.fields['security_policy_reference_resource']

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
    for (field, value) in request_init['security_policy_reference_resource'].items():
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
                for i in range(0, len(request_init['security_policy_reference_resource'][field])):
                    del request_init['security_policy_reference_resource'][field][i][subfield]
            else:
                del request_init['security_policy_reference_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_security_policy_unary(request)
    assert isinstance(response, compute.Operation)

def test_set_security_policy_unary_rest_required_fields(request_type=compute.SetSecurityPolicyBackendServiceRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['backend_service'] = ''
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_security_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['backendService'] = 'backend_service_value'
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_security_policy._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'backendService' in jsonified_request
    assert jsonified_request['backendService'] == 'backend_service_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Operation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Operation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.set_security_policy_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_security_policy_unary_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_security_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('backendService', 'project', 'securityPolicyReferenceResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_security_policy_unary_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_set_security_policy') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_set_security_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetSecurityPolicyBackendServiceRequest.pb(compute.SetSecurityPolicyBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SetSecurityPolicyBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.set_security_policy_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_security_policy_unary_rest_bad_request(transport: str='rest', request_type=compute.SetSecurityPolicyBackendServiceRequest):
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_security_policy_unary(request)

def test_set_security_policy_unary_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'backend_service': 'sample2'}
        mock_args = dict(project='project_value', backend_service='backend_service_value', security_policy_reference_resource=compute.SecurityPolicyReference(security_policy='security_policy_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_security_policy_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices/{backend_service}/setSecurityPolicy' % client.transport._host, args[1])

def test_set_security_policy_unary_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_security_policy_unary(compute.SetSecurityPolicyBackendServiceRequest(), project='project_value', backend_service='backend_service_value', security_policy_reference_resource=compute.SecurityPolicyReference(security_policy='security_policy_value'))

def test_set_security_policy_unary_rest_error():
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.UpdateBackendServiceRequest, dict])
def test_update_rest(request_type):
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request_init['backend_service_resource'] = {'affinity_cookie_ttl_sec': 2432, 'backends': [{'balancing_mode': 'balancing_mode_value', 'capacity_scaler': 0.1575, 'description': 'description_value', 'failover': True, 'group': 'group_value', 'max_connections': 1608, 'max_connections_per_endpoint': 2990, 'max_connections_per_instance': 2978, 'max_rate': 849, 'max_rate_per_endpoint': 0.22310000000000002, 'max_rate_per_instance': 0.22190000000000001, 'max_utilization': 0.1633}], 'cdn_policy': {'bypass_cache_on_request_headers': [{'header_name': 'header_name_value'}], 'cache_key_policy': {'include_host': True, 'include_http_headers': ['include_http_headers_value1', 'include_http_headers_value2'], 'include_named_cookies': ['include_named_cookies_value1', 'include_named_cookies_value2'], 'include_protocol': True, 'include_query_string': True, 'query_string_blacklist': ['query_string_blacklist_value1', 'query_string_blacklist_value2'], 'query_string_whitelist': ['query_string_whitelist_value1', 'query_string_whitelist_value2']}, 'cache_mode': 'cache_mode_value', 'client_ttl': 1074, 'default_ttl': 1176, 'max_ttl': 761, 'negative_caching': True, 'negative_caching_policy': [{'code': 411, 'ttl': 340}], 'request_coalescing': True, 'serve_while_stale': 1813, 'signed_url_cache_max_age_sec': 2890, 'signed_url_key_names': ['signed_url_key_names_value1', 'signed_url_key_names_value2']}, 'circuit_breakers': {'max_connections': 1608, 'max_pending_requests': 2149, 'max_requests': 1313, 'max_requests_per_connection': 2902, 'max_retries': 1187}, 'compression_mode': 'compression_mode_value', 'connection_draining': {'draining_timeout_sec': 2124}, 'connection_tracking_policy': {'connection_persistence_on_unhealthy_backends': 'connection_persistence_on_unhealthy_backends_value', 'enable_strong_affinity': True, 'idle_timeout_sec': 1694, 'tracking_mode': 'tracking_mode_value'}, 'consistent_hash': {'http_cookie': {'name': 'name_value', 'path': 'path_value', 'ttl': {'nanos': 543, 'seconds': 751}}, 'http_header_name': 'http_header_name_value', 'minimum_ring_size': 1829}, 'creation_timestamp': 'creation_timestamp_value', 'custom_request_headers': ['custom_request_headers_value1', 'custom_request_headers_value2'], 'custom_response_headers': ['custom_response_headers_value1', 'custom_response_headers_value2'], 'description': 'description_value', 'edge_security_policy': 'edge_security_policy_value', 'enable_c_d_n': True, 'failover_policy': {'disable_connection_drain_on_failover': True, 'drop_traffic_if_unhealthy': True, 'failover_ratio': 0.1494}, 'fingerprint': 'fingerprint_value', 'health_checks': ['health_checks_value1', 'health_checks_value2'], 'iap': {'enabled': True, 'oauth2_client_id': 'oauth2_client_id_value', 'oauth2_client_secret': 'oauth2_client_secret_value', 'oauth2_client_secret_sha256': 'oauth2_client_secret_sha256_value'}, 'id': 205, 'kind': 'kind_value', 'load_balancing_scheme': 'load_balancing_scheme_value', 'locality_lb_policies': [{'custom_policy': {'data': 'data_value', 'name': 'name_value'}, 'policy': {'name': 'name_value'}}], 'locality_lb_policy': 'locality_lb_policy_value', 'log_config': {'enable': True, 'optional_fields': ['optional_fields_value1', 'optional_fields_value2'], 'optional_mode': 'optional_mode_value', 'sample_rate': 0.1165}, 'max_stream_duration': {}, 'metadatas': {}, 'name': 'name_value', 'network': 'network_value', 'outlier_detection': {'base_ejection_time': {}, 'consecutive_errors': 1956, 'consecutive_gateway_failure': 2880, 'enforcing_consecutive_errors': 3006, 'enforcing_consecutive_gateway_failure': 3930, 'enforcing_success_rate': 2334, 'interval': {}, 'max_ejection_percent': 2118, 'success_rate_minimum_hosts': 2799, 'success_rate_request_volume': 2915, 'success_rate_stdev_factor': 2663}, 'port': 453, 'port_name': 'port_name_value', 'protocol': 'protocol_value', 'region': 'region_value', 'security_policy': 'security_policy_value', 'security_settings': {'client_tls_policy': 'client_tls_policy_value', 'subject_alt_names': ['subject_alt_names_value1', 'subject_alt_names_value2']}, 'self_link': 'self_link_value', 'service_bindings': ['service_bindings_value1', 'service_bindings_value2'], 'session_affinity': 'session_affinity_value', 'subsetting': {'policy': 'policy_value'}, 'timeout_sec': 1185}
    test_field = compute.UpdateBackendServiceRequest.meta.fields['backend_service_resource']

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
    for (field, value) in request_init['backend_service_resource'].items():
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
                for i in range(0, len(request_init['backend_service_resource'][field])):
                    del request_init['backend_service_resource'][field][i][subfield]
            else:
                del request_init['backend_service_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update(request)
    assert isinstance(response, extended_operation.ExtendedOperation)
    assert response.client_operation_id == 'client_operation_id_value'
    assert response.creation_timestamp == 'creation_timestamp_value'
    assert response.description == 'description_value'
    assert response.end_time == 'end_time_value'
    assert response.http_error_message == 'http_error_message_value'
    assert response.http_error_status_code == 2374
    assert response.id == 205
    assert response.insert_time == 'insert_time_value'
    assert response.kind == 'kind_value'
    assert response.name == 'name_value'
    assert response.operation_group_id == 'operation_group_id_value'
    assert response.operation_type == 'operation_type_value'
    assert response.progress == 885
    assert response.region == 'region_value'
    assert response.self_link == 'self_link_value'
    assert response.start_time == 'start_time_value'
    assert response.status == compute.Operation.Status.DONE
    assert response.status_message == 'status_message_value'
    assert response.target_id == 947
    assert response.target_link == 'target_link_value'
    assert response.user == 'user_value'
    assert response.zone == 'zone_value'

def test_update_rest_required_fields(request_type=compute.UpdateBackendServiceRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['backend_service'] = ''
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['backendService'] = 'backend_service_value'
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'backendService' in jsonified_request
    assert jsonified_request['backendService'] == 'backend_service_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Operation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'put', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Operation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('backendService', 'backendServiceResource', 'project'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_update') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_update') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.UpdateBackendServiceRequest.pb(compute.UpdateBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.UpdateBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.update(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_rest_bad_request(transport: str='rest', request_type=compute.UpdateBackendServiceRequest):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update(request)

def test_update_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'backend_service': 'sample2'}
        mock_args = dict(project='project_value', backend_service='backend_service_value', backend_service_resource=compute.BackendService(affinity_cookie_ttl_sec=2432))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices/{backend_service}' % client.transport._host, args[1])

def test_update_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update(compute.UpdateBackendServiceRequest(), project='project_value', backend_service='backend_service_value', backend_service_resource=compute.BackendService(affinity_cookie_ttl_sec=2432))

def test_update_rest_error():
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.UpdateBackendServiceRequest, dict])
def test_update_unary_rest(request_type):
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request_init['backend_service_resource'] = {'affinity_cookie_ttl_sec': 2432, 'backends': [{'balancing_mode': 'balancing_mode_value', 'capacity_scaler': 0.1575, 'description': 'description_value', 'failover': True, 'group': 'group_value', 'max_connections': 1608, 'max_connections_per_endpoint': 2990, 'max_connections_per_instance': 2978, 'max_rate': 849, 'max_rate_per_endpoint': 0.22310000000000002, 'max_rate_per_instance': 0.22190000000000001, 'max_utilization': 0.1633}], 'cdn_policy': {'bypass_cache_on_request_headers': [{'header_name': 'header_name_value'}], 'cache_key_policy': {'include_host': True, 'include_http_headers': ['include_http_headers_value1', 'include_http_headers_value2'], 'include_named_cookies': ['include_named_cookies_value1', 'include_named_cookies_value2'], 'include_protocol': True, 'include_query_string': True, 'query_string_blacklist': ['query_string_blacklist_value1', 'query_string_blacklist_value2'], 'query_string_whitelist': ['query_string_whitelist_value1', 'query_string_whitelist_value2']}, 'cache_mode': 'cache_mode_value', 'client_ttl': 1074, 'default_ttl': 1176, 'max_ttl': 761, 'negative_caching': True, 'negative_caching_policy': [{'code': 411, 'ttl': 340}], 'request_coalescing': True, 'serve_while_stale': 1813, 'signed_url_cache_max_age_sec': 2890, 'signed_url_key_names': ['signed_url_key_names_value1', 'signed_url_key_names_value2']}, 'circuit_breakers': {'max_connections': 1608, 'max_pending_requests': 2149, 'max_requests': 1313, 'max_requests_per_connection': 2902, 'max_retries': 1187}, 'compression_mode': 'compression_mode_value', 'connection_draining': {'draining_timeout_sec': 2124}, 'connection_tracking_policy': {'connection_persistence_on_unhealthy_backends': 'connection_persistence_on_unhealthy_backends_value', 'enable_strong_affinity': True, 'idle_timeout_sec': 1694, 'tracking_mode': 'tracking_mode_value'}, 'consistent_hash': {'http_cookie': {'name': 'name_value', 'path': 'path_value', 'ttl': {'nanos': 543, 'seconds': 751}}, 'http_header_name': 'http_header_name_value', 'minimum_ring_size': 1829}, 'creation_timestamp': 'creation_timestamp_value', 'custom_request_headers': ['custom_request_headers_value1', 'custom_request_headers_value2'], 'custom_response_headers': ['custom_response_headers_value1', 'custom_response_headers_value2'], 'description': 'description_value', 'edge_security_policy': 'edge_security_policy_value', 'enable_c_d_n': True, 'failover_policy': {'disable_connection_drain_on_failover': True, 'drop_traffic_if_unhealthy': True, 'failover_ratio': 0.1494}, 'fingerprint': 'fingerprint_value', 'health_checks': ['health_checks_value1', 'health_checks_value2'], 'iap': {'enabled': True, 'oauth2_client_id': 'oauth2_client_id_value', 'oauth2_client_secret': 'oauth2_client_secret_value', 'oauth2_client_secret_sha256': 'oauth2_client_secret_sha256_value'}, 'id': 205, 'kind': 'kind_value', 'load_balancing_scheme': 'load_balancing_scheme_value', 'locality_lb_policies': [{'custom_policy': {'data': 'data_value', 'name': 'name_value'}, 'policy': {'name': 'name_value'}}], 'locality_lb_policy': 'locality_lb_policy_value', 'log_config': {'enable': True, 'optional_fields': ['optional_fields_value1', 'optional_fields_value2'], 'optional_mode': 'optional_mode_value', 'sample_rate': 0.1165}, 'max_stream_duration': {}, 'metadatas': {}, 'name': 'name_value', 'network': 'network_value', 'outlier_detection': {'base_ejection_time': {}, 'consecutive_errors': 1956, 'consecutive_gateway_failure': 2880, 'enforcing_consecutive_errors': 3006, 'enforcing_consecutive_gateway_failure': 3930, 'enforcing_success_rate': 2334, 'interval': {}, 'max_ejection_percent': 2118, 'success_rate_minimum_hosts': 2799, 'success_rate_request_volume': 2915, 'success_rate_stdev_factor': 2663}, 'port': 453, 'port_name': 'port_name_value', 'protocol': 'protocol_value', 'region': 'region_value', 'security_policy': 'security_policy_value', 'security_settings': {'client_tls_policy': 'client_tls_policy_value', 'subject_alt_names': ['subject_alt_names_value1', 'subject_alt_names_value2']}, 'self_link': 'self_link_value', 'service_bindings': ['service_bindings_value1', 'service_bindings_value2'], 'session_affinity': 'session_affinity_value', 'subsetting': {'policy': 'policy_value'}, 'timeout_sec': 1185}
    test_field = compute.UpdateBackendServiceRequest.meta.fields['backend_service_resource']

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
    for (field, value) in request_init['backend_service_resource'].items():
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
                for i in range(0, len(request_init['backend_service_resource'][field])):
                    del request_init['backend_service_resource'][field][i][subfield]
            else:
                del request_init['backend_service_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_unary(request)
    assert isinstance(response, compute.Operation)

def test_update_unary_rest_required_fields(request_type=compute.UpdateBackendServiceRequest):
    if False:
        print('Hello World!')
    transport_class = transports.BackendServicesRestTransport
    request_init = {}
    request_init['backend_service'] = ''
    request_init['project'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['backendService'] = 'backend_service_value'
    jsonified_request['project'] = 'project_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'backendService' in jsonified_request
    assert jsonified_request['backendService'] == 'backend_service_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Operation()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'put', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Operation.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.update_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_unary_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('backendService', 'backendServiceResource', 'project'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_unary_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.BackendServicesRestInterceptor())
    client = BackendServicesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.BackendServicesRestInterceptor, 'post_update') as post, mock.patch.object(transports.BackendServicesRestInterceptor, 'pre_update') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.UpdateBackendServiceRequest.pb(compute.UpdateBackendServiceRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.UpdateBackendServiceRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.update_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_unary_rest_bad_request(transport: str='rest', request_type=compute.UpdateBackendServiceRequest):
    if False:
        return 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'backend_service': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_unary(request)

def test_update_unary_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'backend_service': 'sample2'}
        mock_args = dict(project='project_value', backend_service='backend_service_value', backend_service_resource=compute.BackendService(affinity_cookie_ttl_sec=2432))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/global/backendServices/{backend_service}' % client.transport._host, args[1])

def test_update_unary_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_unary(compute.UpdateBackendServiceRequest(), project='project_value', backend_service='backend_service_value', backend_service_resource=compute.BackendService(affinity_cookie_ttl_sec=2432))

def test_update_unary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = BackendServicesClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = BackendServicesClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = BackendServicesClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = BackendServicesClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.BackendServicesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    client = BackendServicesClient(transport=transport)
    assert client.transport is transport

@pytest.mark.parametrize('transport_class', [transports.BackendServicesRestTransport])
def test_transport_adc(transport_class):
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['rest'])
def test_transport_kind(transport_name):
    if False:
        for i in range(10):
            print('nop')
    transport = BackendServicesClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_backend_services_base_transport_error():
    if False:
        return 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.BackendServicesTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_backend_services_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.compute_v1.services.backend_services.transports.BackendServicesTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.BackendServicesTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('add_signed_url_key', 'aggregated_list', 'delete', 'delete_signed_url_key', 'get', 'get_health', 'get_iam_policy', 'insert', 'list', 'patch', 'set_edge_security_policy', 'set_iam_policy', 'set_security_policy', 'update')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_backend_services_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.compute_v1.services.backend_services.transports.BackendServicesTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.BackendServicesTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id='octopus')

def test_backend_services_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.compute_v1.services.backend_services.transports.BackendServicesTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.BackendServicesTransport()
        adc.assert_called_once()

def test_backend_services_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        BackendServicesClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id=None)

def test_backend_services_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.BackendServicesRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['rest'])
def test_backend_services_host_no_port(transport_name):
    if False:
        print('Hello World!')
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='compute.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('compute.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_backend_services_host_with_port(transport_name):
    if False:
        i = 10
        return i + 15
    client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='compute.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('compute.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_backend_services_client_transport_session_collision(transport_name):
    if False:
        i = 10
        return i + 15
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = BackendServicesClient(credentials=creds1, transport=transport_name)
    client2 = BackendServicesClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.add_signed_url_key._session
    session2 = client2.transport.add_signed_url_key._session
    assert session1 != session2
    session1 = client1.transport.aggregated_list._session
    session2 = client2.transport.aggregated_list._session
    assert session1 != session2
    session1 = client1.transport.delete._session
    session2 = client2.transport.delete._session
    assert session1 != session2
    session1 = client1.transport.delete_signed_url_key._session
    session2 = client2.transport.delete_signed_url_key._session
    assert session1 != session2
    session1 = client1.transport.get._session
    session2 = client2.transport.get._session
    assert session1 != session2
    session1 = client1.transport.get_health._session
    session2 = client2.transport.get_health._session
    assert session1 != session2
    session1 = client1.transport.get_iam_policy._session
    session2 = client2.transport.get_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.insert._session
    session2 = client2.transport.insert._session
    assert session1 != session2
    session1 = client1.transport.list._session
    session2 = client2.transport.list._session
    assert session1 != session2
    session1 = client1.transport.patch._session
    session2 = client2.transport.patch._session
    assert session1 != session2
    session1 = client1.transport.set_edge_security_policy._session
    session2 = client2.transport.set_edge_security_policy._session
    assert session1 != session2
    session1 = client1.transport.set_iam_policy._session
    session2 = client2.transport.set_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.set_security_policy._session
    session2 = client2.transport.set_security_policy._session
    assert session1 != session2
    session1 = client1.transport.update._session
    session2 = client2.transport.update._session
    assert session1 != session2

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = BackendServicesClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        return 10
    expected = {'billing_account': 'clam'}
    path = BackendServicesClient.common_billing_account_path(**expected)
    actual = BackendServicesClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        i = 10
        return i + 15
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = BackendServicesClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'octopus'}
    path = BackendServicesClient.common_folder_path(**expected)
    actual = BackendServicesClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = BackendServicesClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'organization': 'nudibranch'}
    path = BackendServicesClient.common_organization_path(**expected)
    actual = BackendServicesClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = BackendServicesClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'mussel'}
    path = BackendServicesClient.common_project_path(**expected)
    actual = BackendServicesClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = BackendServicesClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        print('Hello World!')
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = BackendServicesClient.common_location_path(**expected)
    actual = BackendServicesClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.BackendServicesTransport, '_prep_wrapped_messages') as prep:
        client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.BackendServicesTransport, '_prep_wrapped_messages') as prep:
        transport_class = BackendServicesClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

def test_transport_close():
    if False:
        for i in range(10):
            print('nop')
    transports = {'rest': '_session'}
    for (transport, close_name) in transports.items():
        client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        for i in range(10):
            print('nop')
    transports = ['rest']
    for transport in transports:
        client = BackendServicesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(BackendServicesClient, transports.BackendServicesRestTransport)])
def test_api_key_credentials(client_class, transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth._default, 'get_api_key_credentials', create=True) as get_api_key_credentials:
        mock_cred = mock.Mock()
        get_api_key_credentials.return_value = mock_cred
        options = client_options.ClientOptions()
        options.api_key = 'api_key'
        with mock.patch.object(transport_class, '__init__') as patched:
            patched.return_value = None
            client = client_class(client_options=options)
            patched.assert_called_once_with(credentials=mock_cred, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)