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
from google.cloud.compute_v1.services.region_instance_templates import RegionInstanceTemplatesClient, pagers, transports
from google.cloud.compute_v1.types import compute

def client_cert_source_callback():
    if False:
        for i in range(10):
            print('nop')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        return 10
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
    assert RegionInstanceTemplatesClient._get_default_mtls_endpoint(None) is None
    assert RegionInstanceTemplatesClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert RegionInstanceTemplatesClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert RegionInstanceTemplatesClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert RegionInstanceTemplatesClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert RegionInstanceTemplatesClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(RegionInstanceTemplatesClient, 'rest')])
def test_region_instance_templates_client_from_service_account_info(client_class, transport_name):
    if False:
        while True:
            i = 10
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('compute.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.RegionInstanceTemplatesRestTransport, 'rest')])
def test_region_instance_templates_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(RegionInstanceTemplatesClient, 'rest')])
def test_region_instance_templates_client_from_service_account_file(client_class, transport_name):
    if False:
        for i in range(10):
            print('nop')
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

def test_region_instance_templates_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = RegionInstanceTemplatesClient.get_transport_class()
    available_transports = [transports.RegionInstanceTemplatesRestTransport]
    assert transport in available_transports
    transport = RegionInstanceTemplatesClient.get_transport_class('rest')
    assert transport == transports.RegionInstanceTemplatesRestTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(RegionInstanceTemplatesClient, transports.RegionInstanceTemplatesRestTransport, 'rest')])
@mock.patch.object(RegionInstanceTemplatesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RegionInstanceTemplatesClient))
def test_region_instance_templates_client_client_options(client_class, transport_class, transport_name):
    if False:
        i = 10
        return i + 15
    with mock.patch.object(RegionInstanceTemplatesClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(RegionInstanceTemplatesClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(RegionInstanceTemplatesClient, transports.RegionInstanceTemplatesRestTransport, 'rest', 'true'), (RegionInstanceTemplatesClient, transports.RegionInstanceTemplatesRestTransport, 'rest', 'false')])
@mock.patch.object(RegionInstanceTemplatesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RegionInstanceTemplatesClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_region_instance_templates_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [RegionInstanceTemplatesClient])
@mock.patch.object(RegionInstanceTemplatesClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RegionInstanceTemplatesClient))
def test_region_instance_templates_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(RegionInstanceTemplatesClient, transports.RegionInstanceTemplatesRestTransport, 'rest')])
def test_region_instance_templates_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(RegionInstanceTemplatesClient, transports.RegionInstanceTemplatesRestTransport, 'rest', None)])
def test_region_instance_templates_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        i = 10
        return i + 15
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('request_type', [compute.DeleteRegionInstanceTemplateRequest, dict])
def test_delete_rest(request_type):
    if False:
        print('Hello World!')
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_template': 'sample3'}
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

def test_delete_rest_required_fields(request_type=compute.DeleteRegionInstanceTemplateRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegionInstanceTemplatesRestTransport
    request_init = {}
    request_init['instance_template'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceTemplate'] = 'instance_template_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceTemplate' in jsonified_request
    assert jsonified_request['instanceTemplate'] == 'instance_template_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    transport = transports.RegionInstanceTemplatesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceTemplate', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RegionInstanceTemplatesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceTemplatesRestInterceptor())
    client = RegionInstanceTemplatesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceTemplatesRestInterceptor, 'post_delete') as post, mock.patch.object(transports.RegionInstanceTemplatesRestInterceptor, 'pre_delete') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteRegionInstanceTemplateRequest.pb(compute.DeleteRegionInstanceTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteRegionInstanceTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_rest_bad_request(transport: str='rest', request_type=compute.DeleteRegionInstanceTemplateRequest):
    if False:
        print('Hello World!')
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_template': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete(request)

def test_delete_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_template': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_template='instance_template_value')
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceTemplates/{instance_template}' % client.transport._host, args[1])

def test_delete_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete(compute.DeleteRegionInstanceTemplateRequest(), project='project_value', region='region_value', instance_template='instance_template_value')

def test_delete_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.DeleteRegionInstanceTemplateRequest, dict])
def test_delete_unary_rest(request_type):
    if False:
        return 10
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_template': 'sample3'}
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

def test_delete_unary_rest_required_fields(request_type=compute.DeleteRegionInstanceTemplateRequest):
    if False:
        print('Hello World!')
    transport_class = transports.RegionInstanceTemplatesRestTransport
    request_init = {}
    request_init['instance_template'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceTemplate'] = 'instance_template_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceTemplate' in jsonified_request
    assert jsonified_request['instanceTemplate'] == 'instance_template_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    transport = transports.RegionInstanceTemplatesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceTemplate', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_unary_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionInstanceTemplatesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceTemplatesRestInterceptor())
    client = RegionInstanceTemplatesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceTemplatesRestInterceptor, 'post_delete') as post, mock.patch.object(transports.RegionInstanceTemplatesRestInterceptor, 'pre_delete') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteRegionInstanceTemplateRequest.pb(compute.DeleteRegionInstanceTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteRegionInstanceTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_unary_rest_bad_request(transport: str='rest', request_type=compute.DeleteRegionInstanceTemplateRequest):
    if False:
        return 10
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_template': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_unary(request)

def test_delete_unary_rest_flattened():
    if False:
        while True:
            i = 10
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_template': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_template='instance_template_value')
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceTemplates/{instance_template}' % client.transport._host, args[1])

def test_delete_unary_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_unary(compute.DeleteRegionInstanceTemplateRequest(), project='project_value', region='region_value', instance_template='instance_template_value')

def test_delete_unary_rest_error():
    if False:
        return 10
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.GetRegionInstanceTemplateRequest, dict])
def test_get_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_template': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.InstanceTemplate(creation_timestamp='creation_timestamp_value', description='description_value', id=205, kind='kind_value', name='name_value', region='region_value', self_link='self_link_value', source_instance='source_instance_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.InstanceTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get(request)
    assert isinstance(response, compute.InstanceTemplate)
    assert response.creation_timestamp == 'creation_timestamp_value'
    assert response.description == 'description_value'
    assert response.id == 205
    assert response.kind == 'kind_value'
    assert response.name == 'name_value'
    assert response.region == 'region_value'
    assert response.self_link == 'self_link_value'
    assert response.source_instance == 'source_instance_value'

def test_get_rest_required_fields(request_type=compute.GetRegionInstanceTemplateRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.RegionInstanceTemplatesRestTransport
    request_init = {}
    request_init['instance_template'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceTemplate'] = 'instance_template_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'instanceTemplate' in jsonified_request
    assert jsonified_request['instanceTemplate'] == 'instance_template_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.InstanceTemplate()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.InstanceTemplate.pb(return_value)
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
    transport = transports.RegionInstanceTemplatesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('instanceTemplate', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionInstanceTemplatesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceTemplatesRestInterceptor())
    client = RegionInstanceTemplatesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceTemplatesRestInterceptor, 'post_get') as post, mock.patch.object(transports.RegionInstanceTemplatesRestInterceptor, 'pre_get') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.GetRegionInstanceTemplateRequest.pb(compute.GetRegionInstanceTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.InstanceTemplate.to_json(compute.InstanceTemplate())
        request = compute.GetRegionInstanceTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.InstanceTemplate()
        client.get(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_rest_bad_request(transport: str='rest', request_type=compute.GetRegionInstanceTemplateRequest):
    if False:
        while True:
            i = 10
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_template': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get(request)

def test_get_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.InstanceTemplate()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_template': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_template='instance_template_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.InstanceTemplate.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceTemplates/{instance_template}' % client.transport._host, args[1])

def test_get_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get(compute.GetRegionInstanceTemplateRequest(), project='project_value', region='region_value', instance_template='instance_template_value')

def test_get_rest_error():
    if False:
        return 10
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.InsertRegionInstanceTemplateRequest, dict])
def test_insert_rest(request_type):
    if False:
        print('Hello World!')
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request_init['instance_template_resource'] = {'creation_timestamp': 'creation_timestamp_value', 'description': 'description_value', 'id': 205, 'kind': 'kind_value', 'name': 'name_value', 'properties': {'advanced_machine_features': {'enable_nested_virtualization': True, 'enable_uefi_networking': True, 'threads_per_core': 1689, 'visible_core_count': 1918}, 'can_ip_forward': True, 'confidential_instance_config': {'enable_confidential_compute': True}, 'description': 'description_value', 'disks': [{'architecture': 'architecture_value', 'auto_delete': True, 'boot': True, 'device_name': 'device_name_value', 'disk_encryption_key': {'kms_key_name': 'kms_key_name_value', 'kms_key_service_account': 'kms_key_service_account_value', 'raw_key': 'raw_key_value', 'rsa_encrypted_key': 'rsa_encrypted_key_value', 'sha256': 'sha256_value'}, 'disk_size_gb': 1261, 'force_attach': True, 'guest_os_features': [{'type_': 'type__value'}], 'index': 536, 'initialize_params': {'architecture': 'architecture_value', 'description': 'description_value', 'disk_name': 'disk_name_value', 'disk_size_gb': 1261, 'disk_type': 'disk_type_value', 'labels': {}, 'licenses': ['licenses_value1', 'licenses_value2'], 'on_update_action': 'on_update_action_value', 'provisioned_iops': 1740, 'provisioned_throughput': 2411, 'replica_zones': ['replica_zones_value1', 'replica_zones_value2'], 'resource_manager_tags': {}, 'resource_policies': ['resource_policies_value1', 'resource_policies_value2'], 'source_image': 'source_image_value', 'source_image_encryption_key': {}, 'source_snapshot': 'source_snapshot_value', 'source_snapshot_encryption_key': {}}, 'interface': 'interface_value', 'kind': 'kind_value', 'licenses': ['licenses_value1', 'licenses_value2'], 'mode': 'mode_value', 'saved_state': 'saved_state_value', 'shielded_instance_initial_state': {'dbs': [{'content': 'content_value', 'file_type': 'file_type_value'}], 'dbxs': {}, 'keks': {}, 'pk': {}}, 'source': 'source_value', 'type_': 'type__value'}], 'guest_accelerators': [{'accelerator_count': 1805, 'accelerator_type': 'accelerator_type_value'}], 'key_revocation_action_type': 'key_revocation_action_type_value', 'labels': {}, 'machine_type': 'machine_type_value', 'metadata': {'fingerprint': 'fingerprint_value', 'items': [{'key': 'key_value', 'value': 'value_value'}], 'kind': 'kind_value'}, 'min_cpu_platform': 'min_cpu_platform_value', 'network_interfaces': [{'access_configs': [{'external_ipv6': 'external_ipv6_value', 'external_ipv6_prefix_length': 2837, 'kind': 'kind_value', 'name': 'name_value', 'nat_i_p': 'nat_i_p_value', 'network_tier': 'network_tier_value', 'public_ptr_domain_name': 'public_ptr_domain_name_value', 'set_public_ptr': True, 'type_': 'type__value'}], 'alias_ip_ranges': [{'ip_cidr_range': 'ip_cidr_range_value', 'subnetwork_range_name': 'subnetwork_range_name_value'}], 'fingerprint': 'fingerprint_value', 'internal_ipv6_prefix_length': 2831, 'ipv6_access_configs': {}, 'ipv6_access_type': 'ipv6_access_type_value', 'ipv6_address': 'ipv6_address_value', 'kind': 'kind_value', 'name': 'name_value', 'network': 'network_value', 'network_attachment': 'network_attachment_value', 'network_i_p': 'network_i_p_value', 'nic_type': 'nic_type_value', 'queue_count': 1197, 'stack_type': 'stack_type_value', 'subnetwork': 'subnetwork_value'}], 'network_performance_config': {'total_egress_bandwidth_tier': 'total_egress_bandwidth_tier_value'}, 'private_ipv6_google_access': 'private_ipv6_google_access_value', 'reservation_affinity': {'consume_reservation_type': 'consume_reservation_type_value', 'key': 'key_value', 'values': ['values_value1', 'values_value2']}, 'resource_manager_tags': {}, 'resource_policies': ['resource_policies_value1', 'resource_policies_value2'], 'scheduling': {'automatic_restart': True, 'instance_termination_action': 'instance_termination_action_value', 'local_ssd_recovery_timeout': {'nanos': 543, 'seconds': 751}, 'location_hint': 'location_hint_value', 'min_node_cpus': 1379, 'node_affinities': [{'key': 'key_value', 'operator': 'operator_value', 'values': ['values_value1', 'values_value2']}], 'on_host_maintenance': 'on_host_maintenance_value', 'preemptible': True, 'provisioning_model': 'provisioning_model_value'}, 'service_accounts': [{'email': 'email_value', 'scopes': ['scopes_value1', 'scopes_value2']}], 'shielded_instance_config': {'enable_integrity_monitoring': True, 'enable_secure_boot': True, 'enable_vtpm': True}, 'tags': {'fingerprint': 'fingerprint_value', 'items': ['items_value1', 'items_value2']}}, 'region': 'region_value', 'self_link': 'self_link_value', 'source_instance': 'source_instance_value', 'source_instance_params': {'disk_configs': [{'auto_delete': True, 'custom_image': 'custom_image_value', 'device_name': 'device_name_value', 'instantiate_from': 'instantiate_from_value'}]}}
    test_field = compute.InsertRegionInstanceTemplateRequest.meta.fields['instance_template_resource']

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
    for (field, value) in request_init['instance_template_resource'].items():
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
                for i in range(0, len(request_init['instance_template_resource'][field])):
                    del request_init['instance_template_resource'][field][i][subfield]
            else:
                del request_init['instance_template_resource'][field][subfield]
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

def test_insert_rest_required_fields(request_type=compute.InsertRegionInstanceTemplateRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.RegionInstanceTemplatesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).insert._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).insert._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    transport = transports.RegionInstanceTemplatesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.insert._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceTemplateResource', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_insert_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RegionInstanceTemplatesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceTemplatesRestInterceptor())
    client = RegionInstanceTemplatesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceTemplatesRestInterceptor, 'post_insert') as post, mock.patch.object(transports.RegionInstanceTemplatesRestInterceptor, 'pre_insert') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.InsertRegionInstanceTemplateRequest.pb(compute.InsertRegionInstanceTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.InsertRegionInstanceTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.insert(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_insert_rest_bad_request(transport: str='rest', request_type=compute.InsertRegionInstanceTemplateRequest):
    if False:
        print('Hello World!')
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.insert(request)

def test_insert_rest_flattened():
    if False:
        return 10
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        mock_args = dict(project='project_value', region='region_value', instance_template_resource=compute.InstanceTemplate(creation_timestamp='creation_timestamp_value'))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceTemplates' % client.transport._host, args[1])

def test_insert_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.insert(compute.InsertRegionInstanceTemplateRequest(), project='project_value', region='region_value', instance_template_resource=compute.InstanceTemplate(creation_timestamp='creation_timestamp_value'))

def test_insert_rest_error():
    if False:
        return 10
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.InsertRegionInstanceTemplateRequest, dict])
def test_insert_unary_rest(request_type):
    if False:
        return 10
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request_init['instance_template_resource'] = {'creation_timestamp': 'creation_timestamp_value', 'description': 'description_value', 'id': 205, 'kind': 'kind_value', 'name': 'name_value', 'properties': {'advanced_machine_features': {'enable_nested_virtualization': True, 'enable_uefi_networking': True, 'threads_per_core': 1689, 'visible_core_count': 1918}, 'can_ip_forward': True, 'confidential_instance_config': {'enable_confidential_compute': True}, 'description': 'description_value', 'disks': [{'architecture': 'architecture_value', 'auto_delete': True, 'boot': True, 'device_name': 'device_name_value', 'disk_encryption_key': {'kms_key_name': 'kms_key_name_value', 'kms_key_service_account': 'kms_key_service_account_value', 'raw_key': 'raw_key_value', 'rsa_encrypted_key': 'rsa_encrypted_key_value', 'sha256': 'sha256_value'}, 'disk_size_gb': 1261, 'force_attach': True, 'guest_os_features': [{'type_': 'type__value'}], 'index': 536, 'initialize_params': {'architecture': 'architecture_value', 'description': 'description_value', 'disk_name': 'disk_name_value', 'disk_size_gb': 1261, 'disk_type': 'disk_type_value', 'labels': {}, 'licenses': ['licenses_value1', 'licenses_value2'], 'on_update_action': 'on_update_action_value', 'provisioned_iops': 1740, 'provisioned_throughput': 2411, 'replica_zones': ['replica_zones_value1', 'replica_zones_value2'], 'resource_manager_tags': {}, 'resource_policies': ['resource_policies_value1', 'resource_policies_value2'], 'source_image': 'source_image_value', 'source_image_encryption_key': {}, 'source_snapshot': 'source_snapshot_value', 'source_snapshot_encryption_key': {}}, 'interface': 'interface_value', 'kind': 'kind_value', 'licenses': ['licenses_value1', 'licenses_value2'], 'mode': 'mode_value', 'saved_state': 'saved_state_value', 'shielded_instance_initial_state': {'dbs': [{'content': 'content_value', 'file_type': 'file_type_value'}], 'dbxs': {}, 'keks': {}, 'pk': {}}, 'source': 'source_value', 'type_': 'type__value'}], 'guest_accelerators': [{'accelerator_count': 1805, 'accelerator_type': 'accelerator_type_value'}], 'key_revocation_action_type': 'key_revocation_action_type_value', 'labels': {}, 'machine_type': 'machine_type_value', 'metadata': {'fingerprint': 'fingerprint_value', 'items': [{'key': 'key_value', 'value': 'value_value'}], 'kind': 'kind_value'}, 'min_cpu_platform': 'min_cpu_platform_value', 'network_interfaces': [{'access_configs': [{'external_ipv6': 'external_ipv6_value', 'external_ipv6_prefix_length': 2837, 'kind': 'kind_value', 'name': 'name_value', 'nat_i_p': 'nat_i_p_value', 'network_tier': 'network_tier_value', 'public_ptr_domain_name': 'public_ptr_domain_name_value', 'set_public_ptr': True, 'type_': 'type__value'}], 'alias_ip_ranges': [{'ip_cidr_range': 'ip_cidr_range_value', 'subnetwork_range_name': 'subnetwork_range_name_value'}], 'fingerprint': 'fingerprint_value', 'internal_ipv6_prefix_length': 2831, 'ipv6_access_configs': {}, 'ipv6_access_type': 'ipv6_access_type_value', 'ipv6_address': 'ipv6_address_value', 'kind': 'kind_value', 'name': 'name_value', 'network': 'network_value', 'network_attachment': 'network_attachment_value', 'network_i_p': 'network_i_p_value', 'nic_type': 'nic_type_value', 'queue_count': 1197, 'stack_type': 'stack_type_value', 'subnetwork': 'subnetwork_value'}], 'network_performance_config': {'total_egress_bandwidth_tier': 'total_egress_bandwidth_tier_value'}, 'private_ipv6_google_access': 'private_ipv6_google_access_value', 'reservation_affinity': {'consume_reservation_type': 'consume_reservation_type_value', 'key': 'key_value', 'values': ['values_value1', 'values_value2']}, 'resource_manager_tags': {}, 'resource_policies': ['resource_policies_value1', 'resource_policies_value2'], 'scheduling': {'automatic_restart': True, 'instance_termination_action': 'instance_termination_action_value', 'local_ssd_recovery_timeout': {'nanos': 543, 'seconds': 751}, 'location_hint': 'location_hint_value', 'min_node_cpus': 1379, 'node_affinities': [{'key': 'key_value', 'operator': 'operator_value', 'values': ['values_value1', 'values_value2']}], 'on_host_maintenance': 'on_host_maintenance_value', 'preemptible': True, 'provisioning_model': 'provisioning_model_value'}, 'service_accounts': [{'email': 'email_value', 'scopes': ['scopes_value1', 'scopes_value2']}], 'shielded_instance_config': {'enable_integrity_monitoring': True, 'enable_secure_boot': True, 'enable_vtpm': True}, 'tags': {'fingerprint': 'fingerprint_value', 'items': ['items_value1', 'items_value2']}}, 'region': 'region_value', 'self_link': 'self_link_value', 'source_instance': 'source_instance_value', 'source_instance_params': {'disk_configs': [{'auto_delete': True, 'custom_image': 'custom_image_value', 'device_name': 'device_name_value', 'instantiate_from': 'instantiate_from_value'}]}}
    test_field = compute.InsertRegionInstanceTemplateRequest.meta.fields['instance_template_resource']

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
    for (field, value) in request_init['instance_template_resource'].items():
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
                for i in range(0, len(request_init['instance_template_resource'][field])):
                    del request_init['instance_template_resource'][field][i][subfield]
            else:
                del request_init['instance_template_resource'][field][subfield]
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

def test_insert_unary_rest_required_fields(request_type=compute.InsertRegionInstanceTemplateRequest):
    if False:
        return 10
    transport_class = transports.RegionInstanceTemplatesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).insert._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).insert._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    transport = transports.RegionInstanceTemplatesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.insert._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceTemplateResource', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_insert_unary_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionInstanceTemplatesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceTemplatesRestInterceptor())
    client = RegionInstanceTemplatesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceTemplatesRestInterceptor, 'post_insert') as post, mock.patch.object(transports.RegionInstanceTemplatesRestInterceptor, 'pre_insert') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.InsertRegionInstanceTemplateRequest.pb(compute.InsertRegionInstanceTemplateRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.InsertRegionInstanceTemplateRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.insert_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_insert_unary_rest_bad_request(transport: str='rest', request_type=compute.InsertRegionInstanceTemplateRequest):
    if False:
        return 10
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2'}
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
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        mock_args = dict(project='project_value', region='region_value', instance_template_resource=compute.InstanceTemplate(creation_timestamp='creation_timestamp_value'))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceTemplates' % client.transport._host, args[1])

def test_insert_unary_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.insert_unary(compute.InsertRegionInstanceTemplateRequest(), project='project_value', region='region_value', instance_template_resource=compute.InstanceTemplate(creation_timestamp='creation_timestamp_value'))

def test_insert_unary_rest_error():
    if False:
        i = 10
        return i + 15
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.ListRegionInstanceTemplatesRequest, dict])
def test_list_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.InstanceTemplateList(id='id_value', kind='kind_value', next_page_token='next_page_token_value', self_link='self_link_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.InstanceTemplateList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list(request)
    assert isinstance(response, pagers.ListPager)
    assert response.id == 'id_value'
    assert response.kind == 'kind_value'
    assert response.next_page_token == 'next_page_token_value'
    assert response.self_link == 'self_link_value'

def test_list_rest_required_fields(request_type=compute.ListRegionInstanceTemplatesRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.RegionInstanceTemplatesRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'max_results', 'order_by', 'page_token', 'return_partial_success'))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.InstanceTemplateList()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.InstanceTemplateList.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.RegionInstanceTemplatesRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess')) & set(('project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RegionInstanceTemplatesRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceTemplatesRestInterceptor())
    client = RegionInstanceTemplatesClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceTemplatesRestInterceptor, 'post_list') as post, mock.patch.object(transports.RegionInstanceTemplatesRestInterceptor, 'pre_list') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.ListRegionInstanceTemplatesRequest.pb(compute.ListRegionInstanceTemplatesRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.InstanceTemplateList.to_json(compute.InstanceTemplateList())
        request = compute.ListRegionInstanceTemplatesRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.InstanceTemplateList()
        client.list(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_rest_bad_request(transport: str='rest', request_type=compute.ListRegionInstanceTemplatesRequest):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list(request)

def test_list_rest_flattened():
    if False:
        print('Hello World!')
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.InstanceTemplateList()
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        mock_args = dict(project='project_value', region='region_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.InstanceTemplateList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceTemplates' % client.transport._host, args[1])

def test_list_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list(compute.ListRegionInstanceTemplatesRequest(), project='project_value', region='region_value')

def test_list_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (compute.InstanceTemplateList(items=[compute.InstanceTemplate(), compute.InstanceTemplate(), compute.InstanceTemplate()], next_page_token='abc'), compute.InstanceTemplateList(items=[], next_page_token='def'), compute.InstanceTemplateList(items=[compute.InstanceTemplate()], next_page_token='ghi'), compute.InstanceTemplateList(items=[compute.InstanceTemplate(), compute.InstanceTemplate()]))
        response = response + response
        response = tuple((compute.InstanceTemplateList.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        pager = client.list(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, compute.InstanceTemplate) for i in results))
        pages = list(client.list(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

def test_credentials_transport_error():
    if False:
        return 10
    transport = transports.RegionInstanceTemplatesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.RegionInstanceTemplatesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = RegionInstanceTemplatesClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.RegionInstanceTemplatesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = RegionInstanceTemplatesClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = RegionInstanceTemplatesClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.RegionInstanceTemplatesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = RegionInstanceTemplatesClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionInstanceTemplatesRestTransport(credentials=ga_credentials.AnonymousCredentials())
    client = RegionInstanceTemplatesClient(transport=transport)
    assert client.transport is transport

@pytest.mark.parametrize('transport_class', [transports.RegionInstanceTemplatesRestTransport])
def test_transport_adc(transport_class):
    if False:
        return 10
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['rest'])
def test_transport_kind(transport_name):
    if False:
        while True:
            i = 10
    transport = RegionInstanceTemplatesClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_region_instance_templates_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.RegionInstanceTemplatesTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_region_instance_templates_base_transport():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch('google.cloud.compute_v1.services.region_instance_templates.transports.RegionInstanceTemplatesTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.RegionInstanceTemplatesTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('delete', 'get', 'insert', 'list')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_region_instance_templates_base_transport_with_credentials_file():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.compute_v1.services.region_instance_templates.transports.RegionInstanceTemplatesTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.RegionInstanceTemplatesTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id='octopus')

def test_region_instance_templates_base_transport_with_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.compute_v1.services.region_instance_templates.transports.RegionInstanceTemplatesTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.RegionInstanceTemplatesTransport()
        adc.assert_called_once()

def test_region_instance_templates_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        RegionInstanceTemplatesClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id=None)

def test_region_instance_templates_http_transport_client_cert_source_for_mtls():
    if False:
        return 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.RegionInstanceTemplatesRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['rest'])
def test_region_instance_templates_host_no_port(transport_name):
    if False:
        while True:
            i = 10
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='compute.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('compute.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_region_instance_templates_host_with_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='compute.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('compute.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_region_instance_templates_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = RegionInstanceTemplatesClient(credentials=creds1, transport=transport_name)
    client2 = RegionInstanceTemplatesClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.delete._session
    session2 = client2.transport.delete._session
    assert session1 != session2
    session1 = client1.transport.get._session
    session2 = client2.transport.get._session
    assert session1 != session2
    session1 = client1.transport.insert._session
    session2 = client2.transport.insert._session
    assert session1 != session2
    session1 = client1.transport.list._session
    session2 = client2.transport.list._session
    assert session1 != session2

def test_common_billing_account_path():
    if False:
        while True:
            i = 10
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = RegionInstanceTemplatesClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        i = 10
        return i + 15
    expected = {'billing_account': 'clam'}
    path = RegionInstanceTemplatesClient.common_billing_account_path(**expected)
    actual = RegionInstanceTemplatesClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = RegionInstanceTemplatesClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        print('Hello World!')
    expected = {'folder': 'octopus'}
    path = RegionInstanceTemplatesClient.common_folder_path(**expected)
    actual = RegionInstanceTemplatesClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        print('Hello World!')
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = RegionInstanceTemplatesClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'nudibranch'}
    path = RegionInstanceTemplatesClient.common_organization_path(**expected)
    actual = RegionInstanceTemplatesClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = RegionInstanceTemplatesClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'mussel'}
    path = RegionInstanceTemplatesClient.common_project_path(**expected)
    actual = RegionInstanceTemplatesClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = RegionInstanceTemplatesClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = RegionInstanceTemplatesClient.common_location_path(**expected)
    actual = RegionInstanceTemplatesClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.RegionInstanceTemplatesTransport, '_prep_wrapped_messages') as prep:
        client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.RegionInstanceTemplatesTransport, '_prep_wrapped_messages') as prep:
        transport_class = RegionInstanceTemplatesClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

def test_transport_close():
    if False:
        print('Hello World!')
    transports = {'rest': '_session'}
    for (transport, close_name) in transports.items():
        client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        i = 10
        return i + 15
    transports = ['rest']
    for transport in transports:
        client = RegionInstanceTemplatesClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(RegionInstanceTemplatesClient, transports.RegionInstanceTemplatesRestTransport)])
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