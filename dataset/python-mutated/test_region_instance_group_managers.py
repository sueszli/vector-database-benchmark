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
from google.cloud.compute_v1.services.region_instance_group_managers import RegionInstanceGroupManagersClient, pagers, transports
from google.cloud.compute_v1.types import compute

def client_cert_source_callback():
    if False:
        print('Hello World!')
    return (b'cert bytes', b'key bytes')

def modify_default_endpoint(client):
    if False:
        i = 10
        return i + 15
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
    assert RegionInstanceGroupManagersClient._get_default_mtls_endpoint(None) is None
    assert RegionInstanceGroupManagersClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert RegionInstanceGroupManagersClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert RegionInstanceGroupManagersClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert RegionInstanceGroupManagersClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert RegionInstanceGroupManagersClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(RegionInstanceGroupManagersClient, 'rest')])
def test_region_instance_group_managers_client_from_service_account_info(client_class, transport_name):
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

@pytest.mark.parametrize('transport_class,transport_name', [(transports.RegionInstanceGroupManagersRestTransport, 'rest')])
def test_region_instance_group_managers_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(RegionInstanceGroupManagersClient, 'rest')])
def test_region_instance_group_managers_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('compute.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com')

def test_region_instance_group_managers_client_get_transport_class():
    if False:
        print('Hello World!')
    transport = RegionInstanceGroupManagersClient.get_transport_class()
    available_transports = [transports.RegionInstanceGroupManagersRestTransport]
    assert transport in available_transports
    transport = RegionInstanceGroupManagersClient.get_transport_class('rest')
    assert transport == transports.RegionInstanceGroupManagersRestTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(RegionInstanceGroupManagersClient, transports.RegionInstanceGroupManagersRestTransport, 'rest')])
@mock.patch.object(RegionInstanceGroupManagersClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RegionInstanceGroupManagersClient))
def test_region_instance_group_managers_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(RegionInstanceGroupManagersClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(RegionInstanceGroupManagersClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(RegionInstanceGroupManagersClient, transports.RegionInstanceGroupManagersRestTransport, 'rest', 'true'), (RegionInstanceGroupManagersClient, transports.RegionInstanceGroupManagersRestTransport, 'rest', 'false')])
@mock.patch.object(RegionInstanceGroupManagersClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RegionInstanceGroupManagersClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_region_instance_group_managers_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [RegionInstanceGroupManagersClient])
@mock.patch.object(RegionInstanceGroupManagersClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RegionInstanceGroupManagersClient))
def test_region_instance_group_managers_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(RegionInstanceGroupManagersClient, transports.RegionInstanceGroupManagersRestTransport, 'rest')])
def test_region_instance_group_managers_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        return 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(RegionInstanceGroupManagersClient, transports.RegionInstanceGroupManagersRestTransport, 'rest', None)])
def test_region_instance_group_managers_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        for i in range(10):
            print('nop')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('request_type', [compute.AbandonInstancesRegionInstanceGroupManagerRequest, dict])
def test_abandon_instances_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_managers_abandon_instances_request_resource'] = {'instances': ['instances_value1', 'instances_value2']}
    test_field = compute.AbandonInstancesRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_managers_abandon_instances_request_resource']

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
    for (field, value) in request_init['region_instance_group_managers_abandon_instances_request_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_managers_abandon_instances_request_resource'][field])):
                    del request_init['region_instance_group_managers_abandon_instances_request_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_managers_abandon_instances_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.abandon_instances(request)
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

def test_abandon_instances_rest_required_fields(request_type=compute.AbandonInstancesRegionInstanceGroupManagerRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).abandon_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).abandon_instances._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.abandon_instances(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_abandon_instances_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.abandon_instances._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagersAbandonInstancesRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_abandon_instances_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_abandon_instances') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_abandon_instances') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.AbandonInstancesRegionInstanceGroupManagerRequest.pb(compute.AbandonInstancesRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.AbandonInstancesRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.abandon_instances(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_abandon_instances_rest_bad_request(transport: str='rest', request_type=compute.AbandonInstancesRegionInstanceGroupManagerRequest):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.abandon_instances(request)

def test_abandon_instances_rest_flattened():
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_abandon_instances_request_resource=compute.RegionInstanceGroupManagersAbandonInstancesRequest(instances=['instances_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.abandon_instances(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/abandonInstances' % client.transport._host, args[1])

def test_abandon_instances_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.abandon_instances(compute.AbandonInstancesRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_abandon_instances_request_resource=compute.RegionInstanceGroupManagersAbandonInstancesRequest(instances=['instances_value']))

def test_abandon_instances_rest_error():
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.AbandonInstancesRegionInstanceGroupManagerRequest, dict])
def test_abandon_instances_unary_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_managers_abandon_instances_request_resource'] = {'instances': ['instances_value1', 'instances_value2']}
    test_field = compute.AbandonInstancesRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_managers_abandon_instances_request_resource']

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
    for (field, value) in request_init['region_instance_group_managers_abandon_instances_request_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_managers_abandon_instances_request_resource'][field])):
                    del request_init['region_instance_group_managers_abandon_instances_request_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_managers_abandon_instances_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.abandon_instances_unary(request)
    assert isinstance(response, compute.Operation)

def test_abandon_instances_unary_rest_required_fields(request_type=compute.AbandonInstancesRegionInstanceGroupManagerRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).abandon_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).abandon_instances._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.abandon_instances_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_abandon_instances_unary_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.abandon_instances._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagersAbandonInstancesRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_abandon_instances_unary_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_abandon_instances') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_abandon_instances') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.AbandonInstancesRegionInstanceGroupManagerRequest.pb(compute.AbandonInstancesRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.AbandonInstancesRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.abandon_instances_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_abandon_instances_unary_rest_bad_request(transport: str='rest', request_type=compute.AbandonInstancesRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.abandon_instances_unary(request)

def test_abandon_instances_unary_rest_flattened():
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_abandon_instances_request_resource=compute.RegionInstanceGroupManagersAbandonInstancesRequest(instances=['instances_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.abandon_instances_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/abandonInstances' % client.transport._host, args[1])

def test_abandon_instances_unary_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.abandon_instances_unary(compute.AbandonInstancesRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_abandon_instances_request_resource=compute.RegionInstanceGroupManagersAbandonInstancesRequest(instances=['instances_value']))

def test_abandon_instances_unary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.ApplyUpdatesToInstancesRegionInstanceGroupManagerRequest, dict])
def test_apply_updates_to_instances_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_managers_apply_updates_request_resource'] = {'all_instances': True, 'instances': ['instances_value1', 'instances_value2'], 'minimal_action': 'minimal_action_value', 'most_disruptive_allowed_action': 'most_disruptive_allowed_action_value'}
    test_field = compute.ApplyUpdatesToInstancesRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_managers_apply_updates_request_resource']

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
    for (field, value) in request_init['region_instance_group_managers_apply_updates_request_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_managers_apply_updates_request_resource'][field])):
                    del request_init['region_instance_group_managers_apply_updates_request_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_managers_apply_updates_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.apply_updates_to_instances(request)
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

def test_apply_updates_to_instances_rest_required_fields(request_type=compute.ApplyUpdatesToInstancesRegionInstanceGroupManagerRequest):
    if False:
        return 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).apply_updates_to_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).apply_updates_to_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.apply_updates_to_instances(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_apply_updates_to_instances_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.apply_updates_to_instances._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagersApplyUpdatesRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_apply_updates_to_instances_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_apply_updates_to_instances') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_apply_updates_to_instances') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.ApplyUpdatesToInstancesRegionInstanceGroupManagerRequest.pb(compute.ApplyUpdatesToInstancesRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.ApplyUpdatesToInstancesRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.apply_updates_to_instances(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_apply_updates_to_instances_rest_bad_request(transport: str='rest', request_type=compute.ApplyUpdatesToInstancesRegionInstanceGroupManagerRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.apply_updates_to_instances(request)

def test_apply_updates_to_instances_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_apply_updates_request_resource=compute.RegionInstanceGroupManagersApplyUpdatesRequest(all_instances=True))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.apply_updates_to_instances(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/applyUpdatesToInstances' % client.transport._host, args[1])

def test_apply_updates_to_instances_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.apply_updates_to_instances(compute.ApplyUpdatesToInstancesRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_apply_updates_request_resource=compute.RegionInstanceGroupManagersApplyUpdatesRequest(all_instances=True))

def test_apply_updates_to_instances_rest_error():
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.ApplyUpdatesToInstancesRegionInstanceGroupManagerRequest, dict])
def test_apply_updates_to_instances_unary_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_managers_apply_updates_request_resource'] = {'all_instances': True, 'instances': ['instances_value1', 'instances_value2'], 'minimal_action': 'minimal_action_value', 'most_disruptive_allowed_action': 'most_disruptive_allowed_action_value'}
    test_field = compute.ApplyUpdatesToInstancesRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_managers_apply_updates_request_resource']

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
    for (field, value) in request_init['region_instance_group_managers_apply_updates_request_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_managers_apply_updates_request_resource'][field])):
                    del request_init['region_instance_group_managers_apply_updates_request_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_managers_apply_updates_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.apply_updates_to_instances_unary(request)
    assert isinstance(response, compute.Operation)

def test_apply_updates_to_instances_unary_rest_required_fields(request_type=compute.ApplyUpdatesToInstancesRegionInstanceGroupManagerRequest):
    if False:
        return 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).apply_updates_to_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).apply_updates_to_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.apply_updates_to_instances_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_apply_updates_to_instances_unary_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.apply_updates_to_instances._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagersApplyUpdatesRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_apply_updates_to_instances_unary_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_apply_updates_to_instances') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_apply_updates_to_instances') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.ApplyUpdatesToInstancesRegionInstanceGroupManagerRequest.pb(compute.ApplyUpdatesToInstancesRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.ApplyUpdatesToInstancesRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.apply_updates_to_instances_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_apply_updates_to_instances_unary_rest_bad_request(transport: str='rest', request_type=compute.ApplyUpdatesToInstancesRegionInstanceGroupManagerRequest):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.apply_updates_to_instances_unary(request)

def test_apply_updates_to_instances_unary_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_apply_updates_request_resource=compute.RegionInstanceGroupManagersApplyUpdatesRequest(all_instances=True))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.apply_updates_to_instances_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/applyUpdatesToInstances' % client.transport._host, args[1])

def test_apply_updates_to_instances_unary_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.apply_updates_to_instances_unary(compute.ApplyUpdatesToInstancesRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_apply_updates_request_resource=compute.RegionInstanceGroupManagersApplyUpdatesRequest(all_instances=True))

def test_apply_updates_to_instances_unary_rest_error():
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.CreateInstancesRegionInstanceGroupManagerRequest, dict])
def test_create_instances_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_managers_create_instances_request_resource'] = {'instances': [{'fingerprint': 'fingerprint_value', 'name': 'name_value', 'preserved_state': {'disks': {}, 'metadata': {}}, 'status': 'status_value'}]}
    test_field = compute.CreateInstancesRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_managers_create_instances_request_resource']

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
    for (field, value) in request_init['region_instance_group_managers_create_instances_request_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_managers_create_instances_request_resource'][field])):
                    del request_init['region_instance_group_managers_create_instances_request_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_managers_create_instances_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_instances(request)
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

def test_create_instances_rest_required_fields(request_type=compute.CreateInstancesRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_instances._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_instances(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_instances_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_instances._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagersCreateInstancesRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_instances_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_create_instances') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_create_instances') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.CreateInstancesRegionInstanceGroupManagerRequest.pb(compute.CreateInstancesRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.CreateInstancesRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.create_instances(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_instances_rest_bad_request(transport: str='rest', request_type=compute.CreateInstancesRegionInstanceGroupManagerRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_instances(request)

def test_create_instances_rest_flattened():
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_create_instances_request_resource=compute.RegionInstanceGroupManagersCreateInstancesRequest(instances=[compute.PerInstanceConfig(fingerprint='fingerprint_value')]))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_instances(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/createInstances' % client.transport._host, args[1])

def test_create_instances_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_instances(compute.CreateInstancesRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_create_instances_request_resource=compute.RegionInstanceGroupManagersCreateInstancesRequest(instances=[compute.PerInstanceConfig(fingerprint='fingerprint_value')]))

def test_create_instances_rest_error():
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.CreateInstancesRegionInstanceGroupManagerRequest, dict])
def test_create_instances_unary_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_managers_create_instances_request_resource'] = {'instances': [{'fingerprint': 'fingerprint_value', 'name': 'name_value', 'preserved_state': {'disks': {}, 'metadata': {}}, 'status': 'status_value'}]}
    test_field = compute.CreateInstancesRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_managers_create_instances_request_resource']

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
    for (field, value) in request_init['region_instance_group_managers_create_instances_request_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_managers_create_instances_request_resource'][field])):
                    del request_init['region_instance_group_managers_create_instances_request_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_managers_create_instances_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_instances_unary(request)
    assert isinstance(response, compute.Operation)

def test_create_instances_unary_rest_required_fields(request_type=compute.CreateInstancesRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_instances._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_instances_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_instances_unary_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_instances._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagersCreateInstancesRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_instances_unary_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_create_instances') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_create_instances') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.CreateInstancesRegionInstanceGroupManagerRequest.pb(compute.CreateInstancesRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.CreateInstancesRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.create_instances_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_instances_unary_rest_bad_request(transport: str='rest', request_type=compute.CreateInstancesRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_instances_unary(request)

def test_create_instances_unary_rest_flattened():
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_create_instances_request_resource=compute.RegionInstanceGroupManagersCreateInstancesRequest(instances=[compute.PerInstanceConfig(fingerprint='fingerprint_value')]))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_instances_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/createInstances' % client.transport._host, args[1])

def test_create_instances_unary_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_instances_unary(compute.CreateInstancesRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_create_instances_request_resource=compute.RegionInstanceGroupManagersCreateInstancesRequest(instances=[compute.PerInstanceConfig(fingerprint='fingerprint_value')]))

def test_create_instances_unary_rest_error():
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.DeleteRegionInstanceGroupManagerRequest, dict])
def test_delete_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
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

def test_delete_rest_required_fields(request_type=compute.DeleteRegionInstanceGroupManagerRequest):
    if False:
        return 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_delete') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_delete') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteRegionInstanceGroupManagerRequest.pb(compute.DeleteRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_rest_bad_request(transport: str='rest', request_type=compute.DeleteRegionInstanceGroupManagerRequest):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
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
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value')
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}' % client.transport._host, args[1])

def test_delete_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete(compute.DeleteRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value')

def test_delete_rest_error():
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.DeleteRegionInstanceGroupManagerRequest, dict])
def test_delete_unary_rest(request_type):
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
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

def test_delete_unary_rest_required_fields(request_type=compute.DeleteRegionInstanceGroupManagerRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_unary_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_delete') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_delete') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteRegionInstanceGroupManagerRequest.pb(compute.DeleteRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_unary_rest_bad_request(transport: str='rest', request_type=compute.DeleteRegionInstanceGroupManagerRequest):
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_unary(request)

def test_delete_unary_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value')
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}' % client.transport._host, args[1])

def test_delete_unary_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_unary(compute.DeleteRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value')

def test_delete_unary_rest_error():
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.DeleteInstancesRegionInstanceGroupManagerRequest, dict])
def test_delete_instances_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_managers_delete_instances_request_resource'] = {'instances': ['instances_value1', 'instances_value2'], 'skip_instances_on_validation_error': True}
    test_field = compute.DeleteInstancesRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_managers_delete_instances_request_resource']

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
    for (field, value) in request_init['region_instance_group_managers_delete_instances_request_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_managers_delete_instances_request_resource'][field])):
                    del request_init['region_instance_group_managers_delete_instances_request_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_managers_delete_instances_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_instances(request)
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

def test_delete_instances_rest_required_fields(request_type=compute.DeleteInstancesRegionInstanceGroupManagerRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_instances._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_instances(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_instances_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_instances._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagersDeleteInstancesRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_instances_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_delete_instances') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_delete_instances') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteInstancesRegionInstanceGroupManagerRequest.pb(compute.DeleteInstancesRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteInstancesRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete_instances(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_instances_rest_bad_request(transport: str='rest', request_type=compute.DeleteInstancesRegionInstanceGroupManagerRequest):
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_instances(request)

def test_delete_instances_rest_flattened():
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_delete_instances_request_resource=compute.RegionInstanceGroupManagersDeleteInstancesRequest(instances=['instances_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_instances(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/deleteInstances' % client.transport._host, args[1])

def test_delete_instances_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_instances(compute.DeleteInstancesRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_delete_instances_request_resource=compute.RegionInstanceGroupManagersDeleteInstancesRequest(instances=['instances_value']))

def test_delete_instances_rest_error():
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.DeleteInstancesRegionInstanceGroupManagerRequest, dict])
def test_delete_instances_unary_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_managers_delete_instances_request_resource'] = {'instances': ['instances_value1', 'instances_value2'], 'skip_instances_on_validation_error': True}
    test_field = compute.DeleteInstancesRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_managers_delete_instances_request_resource']

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
    for (field, value) in request_init['region_instance_group_managers_delete_instances_request_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_managers_delete_instances_request_resource'][field])):
                    del request_init['region_instance_group_managers_delete_instances_request_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_managers_delete_instances_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_instances_unary(request)
    assert isinstance(response, compute.Operation)

def test_delete_instances_unary_rest_required_fields(request_type=compute.DeleteInstancesRegionInstanceGroupManagerRequest):
    if False:
        print('Hello World!')
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_instances._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_instances_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_instances_unary_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_instances._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagersDeleteInstancesRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_instances_unary_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_delete_instances') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_delete_instances') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteInstancesRegionInstanceGroupManagerRequest.pb(compute.DeleteInstancesRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteInstancesRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete_instances_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_instances_unary_rest_bad_request(transport: str='rest', request_type=compute.DeleteInstancesRegionInstanceGroupManagerRequest):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_instances_unary(request)

def test_delete_instances_unary_rest_flattened():
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_delete_instances_request_resource=compute.RegionInstanceGroupManagersDeleteInstancesRequest(instances=['instances_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_instances_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/deleteInstances' % client.transport._host, args[1])

def test_delete_instances_unary_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_instances_unary(compute.DeleteInstancesRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_delete_instances_request_resource=compute.RegionInstanceGroupManagersDeleteInstancesRequest(instances=['instances_value']))

def test_delete_instances_unary_rest_error():
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.DeletePerInstanceConfigsRegionInstanceGroupManagerRequest, dict])
def test_delete_per_instance_configs_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_manager_delete_instance_config_req_resource'] = {'names': ['names_value1', 'names_value2']}
    test_field = compute.DeletePerInstanceConfigsRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_manager_delete_instance_config_req_resource']

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
    for (field, value) in request_init['region_instance_group_manager_delete_instance_config_req_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_manager_delete_instance_config_req_resource'][field])):
                    del request_init['region_instance_group_manager_delete_instance_config_req_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_manager_delete_instance_config_req_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_per_instance_configs(request)
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

def test_delete_per_instance_configs_rest_required_fields(request_type=compute.DeletePerInstanceConfigsRegionInstanceGroupManagerRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_per_instance_configs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_per_instance_configs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_per_instance_configs(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_per_instance_configs_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_per_instance_configs._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagerDeleteInstanceConfigReqResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_per_instance_configs_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_delete_per_instance_configs') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_delete_per_instance_configs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeletePerInstanceConfigsRegionInstanceGroupManagerRequest.pb(compute.DeletePerInstanceConfigsRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeletePerInstanceConfigsRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete_per_instance_configs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_per_instance_configs_rest_bad_request(transport: str='rest', request_type=compute.DeletePerInstanceConfigsRegionInstanceGroupManagerRequest):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_per_instance_configs(request)

def test_delete_per_instance_configs_rest_flattened():
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_manager_delete_instance_config_req_resource=compute.RegionInstanceGroupManagerDeleteInstanceConfigReq(names=['names_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_per_instance_configs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/deletePerInstanceConfigs' % client.transport._host, args[1])

def test_delete_per_instance_configs_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_per_instance_configs(compute.DeletePerInstanceConfigsRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_manager_delete_instance_config_req_resource=compute.RegionInstanceGroupManagerDeleteInstanceConfigReq(names=['names_value']))

def test_delete_per_instance_configs_rest_error():
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.DeletePerInstanceConfigsRegionInstanceGroupManagerRequest, dict])
def test_delete_per_instance_configs_unary_rest(request_type):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_manager_delete_instance_config_req_resource'] = {'names': ['names_value1', 'names_value2']}
    test_field = compute.DeletePerInstanceConfigsRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_manager_delete_instance_config_req_resource']

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
    for (field, value) in request_init['region_instance_group_manager_delete_instance_config_req_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_manager_delete_instance_config_req_resource'][field])):
                    del request_init['region_instance_group_manager_delete_instance_config_req_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_manager_delete_instance_config_req_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_per_instance_configs_unary(request)
    assert isinstance(response, compute.Operation)

def test_delete_per_instance_configs_unary_rest_required_fields(request_type=compute.DeletePerInstanceConfigsRegionInstanceGroupManagerRequest):
    if False:
        print('Hello World!')
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_per_instance_configs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_per_instance_configs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_per_instance_configs_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_per_instance_configs_unary_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_per_instance_configs._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagerDeleteInstanceConfigReqResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_per_instance_configs_unary_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_delete_per_instance_configs') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_delete_per_instance_configs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeletePerInstanceConfigsRegionInstanceGroupManagerRequest.pb(compute.DeletePerInstanceConfigsRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeletePerInstanceConfigsRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete_per_instance_configs_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_per_instance_configs_unary_rest_bad_request(transport: str='rest', request_type=compute.DeletePerInstanceConfigsRegionInstanceGroupManagerRequest):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_per_instance_configs_unary(request)

def test_delete_per_instance_configs_unary_rest_flattened():
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_manager_delete_instance_config_req_resource=compute.RegionInstanceGroupManagerDeleteInstanceConfigReq(names=['names_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_per_instance_configs_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/deletePerInstanceConfigs' % client.transport._host, args[1])

def test_delete_per_instance_configs_unary_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_per_instance_configs_unary(compute.DeletePerInstanceConfigsRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_manager_delete_instance_config_req_resource=compute.RegionInstanceGroupManagerDeleteInstanceConfigReq(names=['names_value']))

def test_delete_per_instance_configs_unary_rest_error():
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.GetRegionInstanceGroupManagerRequest, dict])
def test_get_rest(request_type):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.InstanceGroupManager(base_instance_name='base_instance_name_value', creation_timestamp='creation_timestamp_value', description='description_value', fingerprint='fingerprint_value', id=205, instance_group='instance_group_value', instance_template='instance_template_value', kind='kind_value', list_managed_instances_results='list_managed_instances_results_value', name='name_value', region='region_value', self_link='self_link_value', target_pools=['target_pools_value'], target_size=1185, zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.InstanceGroupManager.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get(request)
    assert isinstance(response, compute.InstanceGroupManager)
    assert response.base_instance_name == 'base_instance_name_value'
    assert response.creation_timestamp == 'creation_timestamp_value'
    assert response.description == 'description_value'
    assert response.fingerprint == 'fingerprint_value'
    assert response.id == 205
    assert response.instance_group == 'instance_group_value'
    assert response.instance_template == 'instance_template_value'
    assert response.kind == 'kind_value'
    assert response.list_managed_instances_results == 'list_managed_instances_results_value'
    assert response.name == 'name_value'
    assert response.region == 'region_value'
    assert response.self_link == 'self_link_value'
    assert response.target_pools == ['target_pools_value']
    assert response.target_size == 1185
    assert response.zone == 'zone_value'

def test_get_rest_required_fields(request_type=compute.GetRegionInstanceGroupManagerRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.InstanceGroupManager()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.InstanceGroupManager.pb(return_value)
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
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('instanceGroupManager', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_get') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_get') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.GetRegionInstanceGroupManagerRequest.pb(compute.GetRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.InstanceGroupManager.to_json(compute.InstanceGroupManager())
        request = compute.GetRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.InstanceGroupManager()
        client.get(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_rest_bad_request(transport: str='rest', request_type=compute.GetRegionInstanceGroupManagerRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
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
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.InstanceGroupManager()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.InstanceGroupManager.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}' % client.transport._host, args[1])

def test_get_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get(compute.GetRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value')

def test_get_rest_error():
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.InsertRegionInstanceGroupManagerRequest, dict])
def test_insert_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request_init['instance_group_manager_resource'] = {'auto_healing_policies': [{'health_check': 'health_check_value', 'initial_delay_sec': 1778}], 'base_instance_name': 'base_instance_name_value', 'creation_timestamp': 'creation_timestamp_value', 'current_actions': {'abandoning': 1041, 'creating': 845, 'creating_without_retries': 2589, 'deleting': 844, 'none': 432, 'recreating': 1060, 'refreshing': 1069, 'restarting': 1091, 'resuming': 874, 'starting': 876, 'stopping': 884, 'suspending': 1088, 'verifying': 979}, 'description': 'description_value', 'distribution_policy': {'target_shape': 'target_shape_value', 'zones': [{'zone': 'zone_value'}]}, 'fingerprint': 'fingerprint_value', 'id': 205, 'instance_group': 'instance_group_value', 'instance_lifecycle_policy': {'force_update_on_repair': 'force_update_on_repair_value'}, 'instance_template': 'instance_template_value', 'kind': 'kind_value', 'list_managed_instances_results': 'list_managed_instances_results_value', 'name': 'name_value', 'named_ports': [{'name': 'name_value', 'port': 453}], 'region': 'region_value', 'self_link': 'self_link_value', 'stateful_policy': {'preserved_state': {'disks': {}}}, 'status': {'autoscaler': 'autoscaler_value', 'is_stable': True, 'stateful': {'has_stateful_config': True, 'per_instance_configs': {'all_effective': True}}, 'version_target': {'is_reached': True}}, 'target_pools': ['target_pools_value1', 'target_pools_value2'], 'target_size': 1185, 'update_policy': {'instance_redistribution_type': 'instance_redistribution_type_value', 'max_surge': {'calculated': 1042, 'fixed': 528, 'percent': 753}, 'max_unavailable': {}, 'minimal_action': 'minimal_action_value', 'most_disruptive_allowed_action': 'most_disruptive_allowed_action_value', 'replacement_method': 'replacement_method_value', 'type_': 'type__value'}, 'versions': [{'instance_template': 'instance_template_value', 'name': 'name_value', 'target_size': {}}], 'zone': 'zone_value'}
    test_field = compute.InsertRegionInstanceGroupManagerRequest.meta.fields['instance_group_manager_resource']

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
    for (field, value) in request_init['instance_group_manager_resource'].items():
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
                for i in range(0, len(request_init['instance_group_manager_resource'][field])):
                    del request_init['instance_group_manager_resource'][field][i][subfield]
            else:
                del request_init['instance_group_manager_resource'][field][subfield]
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

def test_insert_rest_required_fields(request_type=compute.InsertRegionInstanceGroupManagerRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegionInstanceGroupManagersRestTransport
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
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        while True:
            i = 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.insert._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManagerResource', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_insert_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_insert') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_insert') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.InsertRegionInstanceGroupManagerRequest.pb(compute.InsertRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.InsertRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.insert(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_insert_rest_bad_request(transport: str='rest', request_type=compute.InsertRegionInstanceGroupManagerRequest):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager_resource=compute.InstanceGroupManager(auto_healing_policies=[compute.InstanceGroupManagerAutoHealingPolicy(health_check='health_check_value')]))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers' % client.transport._host, args[1])

def test_insert_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.insert(compute.InsertRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager_resource=compute.InstanceGroupManager(auto_healing_policies=[compute.InstanceGroupManagerAutoHealingPolicy(health_check='health_check_value')]))

def test_insert_rest_error():
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.InsertRegionInstanceGroupManagerRequest, dict])
def test_insert_unary_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request_init['instance_group_manager_resource'] = {'auto_healing_policies': [{'health_check': 'health_check_value', 'initial_delay_sec': 1778}], 'base_instance_name': 'base_instance_name_value', 'creation_timestamp': 'creation_timestamp_value', 'current_actions': {'abandoning': 1041, 'creating': 845, 'creating_without_retries': 2589, 'deleting': 844, 'none': 432, 'recreating': 1060, 'refreshing': 1069, 'restarting': 1091, 'resuming': 874, 'starting': 876, 'stopping': 884, 'suspending': 1088, 'verifying': 979}, 'description': 'description_value', 'distribution_policy': {'target_shape': 'target_shape_value', 'zones': [{'zone': 'zone_value'}]}, 'fingerprint': 'fingerprint_value', 'id': 205, 'instance_group': 'instance_group_value', 'instance_lifecycle_policy': {'force_update_on_repair': 'force_update_on_repair_value'}, 'instance_template': 'instance_template_value', 'kind': 'kind_value', 'list_managed_instances_results': 'list_managed_instances_results_value', 'name': 'name_value', 'named_ports': [{'name': 'name_value', 'port': 453}], 'region': 'region_value', 'self_link': 'self_link_value', 'stateful_policy': {'preserved_state': {'disks': {}}}, 'status': {'autoscaler': 'autoscaler_value', 'is_stable': True, 'stateful': {'has_stateful_config': True, 'per_instance_configs': {'all_effective': True}}, 'version_target': {'is_reached': True}}, 'target_pools': ['target_pools_value1', 'target_pools_value2'], 'target_size': 1185, 'update_policy': {'instance_redistribution_type': 'instance_redistribution_type_value', 'max_surge': {'calculated': 1042, 'fixed': 528, 'percent': 753}, 'max_unavailable': {}, 'minimal_action': 'minimal_action_value', 'most_disruptive_allowed_action': 'most_disruptive_allowed_action_value', 'replacement_method': 'replacement_method_value', 'type_': 'type__value'}, 'versions': [{'instance_template': 'instance_template_value', 'name': 'name_value', 'target_size': {}}], 'zone': 'zone_value'}
    test_field = compute.InsertRegionInstanceGroupManagerRequest.meta.fields['instance_group_manager_resource']

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
    for (field, value) in request_init['instance_group_manager_resource'].items():
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
                for i in range(0, len(request_init['instance_group_manager_resource'][field])):
                    del request_init['instance_group_manager_resource'][field][i][subfield]
            else:
                del request_init['instance_group_manager_resource'][field][subfield]
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

def test_insert_unary_rest_required_fields(request_type=compute.InsertRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
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
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.insert._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManagerResource', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_insert_unary_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_insert') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_insert') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.InsertRegionInstanceGroupManagerRequest.pb(compute.InsertRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.InsertRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.insert_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_insert_unary_rest_bad_request(transport: str='rest', request_type=compute.InsertRegionInstanceGroupManagerRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager_resource=compute.InstanceGroupManager(auto_healing_policies=[compute.InstanceGroupManagerAutoHealingPolicy(health_check='health_check_value')]))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers' % client.transport._host, args[1])

def test_insert_unary_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.insert_unary(compute.InsertRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager_resource=compute.InstanceGroupManager(auto_healing_policies=[compute.InstanceGroupManagerAutoHealingPolicy(health_check='health_check_value')]))

def test_insert_unary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.ListRegionInstanceGroupManagersRequest, dict])
def test_list_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.RegionInstanceGroupManagerList(id='id_value', kind='kind_value', next_page_token='next_page_token_value', self_link='self_link_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.RegionInstanceGroupManagerList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list(request)
    assert isinstance(response, pagers.ListPager)
    assert response.id == 'id_value'
    assert response.kind == 'kind_value'
    assert response.next_page_token == 'next_page_token_value'
    assert response.self_link == 'self_link_value'

def test_list_rest_required_fields(request_type=compute.ListRegionInstanceGroupManagersRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.RegionInstanceGroupManagersRestTransport
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
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.RegionInstanceGroupManagerList()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.RegionInstanceGroupManagerList.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess')) & set(('project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_list') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_list') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.ListRegionInstanceGroupManagersRequest.pb(compute.ListRegionInstanceGroupManagersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.RegionInstanceGroupManagerList.to_json(compute.RegionInstanceGroupManagerList())
        request = compute.ListRegionInstanceGroupManagersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.RegionInstanceGroupManagerList()
        client.list(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_rest_bad_request(transport: str='rest', request_type=compute.ListRegionInstanceGroupManagersRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.RegionInstanceGroupManagerList()
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        mock_args = dict(project='project_value', region='region_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.RegionInstanceGroupManagerList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers' % client.transport._host, args[1])

def test_list_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list(compute.ListRegionInstanceGroupManagersRequest(), project='project_value', region='region_value')

def test_list_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (compute.RegionInstanceGroupManagerList(items=[compute.InstanceGroupManager(), compute.InstanceGroupManager(), compute.InstanceGroupManager()], next_page_token='abc'), compute.RegionInstanceGroupManagerList(items=[], next_page_token='def'), compute.RegionInstanceGroupManagerList(items=[compute.InstanceGroupManager()], next_page_token='ghi'), compute.RegionInstanceGroupManagerList(items=[compute.InstanceGroupManager(), compute.InstanceGroupManager()]))
        response = response + response
        response = tuple((compute.RegionInstanceGroupManagerList.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        pager = client.list(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, compute.InstanceGroupManager) for i in results))
        pages = list(client.list(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [compute.ListErrorsRegionInstanceGroupManagersRequest, dict])
def test_list_errors_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.RegionInstanceGroupManagersListErrorsResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.RegionInstanceGroupManagersListErrorsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_errors(request)
    assert isinstance(response, pagers.ListErrorsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_errors_rest_required_fields(request_type=compute.ListErrorsRegionInstanceGroupManagersRequest):
    if False:
        print('Hello World!')
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_errors._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_errors._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'max_results', 'order_by', 'page_token', 'return_partial_success'))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.RegionInstanceGroupManagersListErrorsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.RegionInstanceGroupManagersListErrorsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_errors(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_errors_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_errors._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess')) & set(('instanceGroupManager', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_errors_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_list_errors') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_list_errors') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.ListErrorsRegionInstanceGroupManagersRequest.pb(compute.ListErrorsRegionInstanceGroupManagersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.RegionInstanceGroupManagersListErrorsResponse.to_json(compute.RegionInstanceGroupManagersListErrorsResponse())
        request = compute.ListErrorsRegionInstanceGroupManagersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.RegionInstanceGroupManagersListErrorsResponse()
        client.list_errors(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_errors_rest_bad_request(transport: str='rest', request_type=compute.ListErrorsRegionInstanceGroupManagersRequest):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_errors(request)

def test_list_errors_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.RegionInstanceGroupManagersListErrorsResponse()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.RegionInstanceGroupManagersListErrorsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_errors(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/listErrors' % client.transport._host, args[1])

def test_list_errors_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_errors(compute.ListErrorsRegionInstanceGroupManagersRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value')

def test_list_errors_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (compute.RegionInstanceGroupManagersListErrorsResponse(items=[compute.InstanceManagedByIgmError(), compute.InstanceManagedByIgmError(), compute.InstanceManagedByIgmError()], next_page_token='abc'), compute.RegionInstanceGroupManagersListErrorsResponse(items=[], next_page_token='def'), compute.RegionInstanceGroupManagersListErrorsResponse(items=[compute.InstanceManagedByIgmError()], next_page_token='ghi'), compute.RegionInstanceGroupManagersListErrorsResponse(items=[compute.InstanceManagedByIgmError(), compute.InstanceManagedByIgmError()]))
        response = response + response
        response = tuple((compute.RegionInstanceGroupManagersListErrorsResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        pager = client.list_errors(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, compute.InstanceManagedByIgmError) for i in results))
        pages = list(client.list_errors(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [compute.ListManagedInstancesRegionInstanceGroupManagersRequest, dict])
def test_list_managed_instances_rest(request_type):
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.RegionInstanceGroupManagersListInstancesResponse(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.RegionInstanceGroupManagersListInstancesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_managed_instances(request)
    assert isinstance(response, pagers.ListManagedInstancesPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_managed_instances_rest_required_fields(request_type=compute.ListManagedInstancesRegionInstanceGroupManagersRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_managed_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_managed_instances._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'max_results', 'order_by', 'page_token', 'return_partial_success'))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.RegionInstanceGroupManagersListInstancesResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.RegionInstanceGroupManagersListInstancesResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_managed_instances(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_managed_instances_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_managed_instances._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess')) & set(('instanceGroupManager', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_managed_instances_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_list_managed_instances') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_list_managed_instances') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.ListManagedInstancesRegionInstanceGroupManagersRequest.pb(compute.ListManagedInstancesRegionInstanceGroupManagersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.RegionInstanceGroupManagersListInstancesResponse.to_json(compute.RegionInstanceGroupManagersListInstancesResponse())
        request = compute.ListManagedInstancesRegionInstanceGroupManagersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.RegionInstanceGroupManagersListInstancesResponse()
        client.list_managed_instances(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_managed_instances_rest_bad_request(transport: str='rest', request_type=compute.ListManagedInstancesRegionInstanceGroupManagersRequest):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_managed_instances(request)

def test_list_managed_instances_rest_flattened():
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.RegionInstanceGroupManagersListInstancesResponse()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.RegionInstanceGroupManagersListInstancesResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_managed_instances(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/listManagedInstances' % client.transport._host, args[1])

def test_list_managed_instances_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_managed_instances(compute.ListManagedInstancesRegionInstanceGroupManagersRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value')

def test_list_managed_instances_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (compute.RegionInstanceGroupManagersListInstancesResponse(managed_instances=[compute.ManagedInstance(), compute.ManagedInstance(), compute.ManagedInstance()], next_page_token='abc'), compute.RegionInstanceGroupManagersListInstancesResponse(managed_instances=[], next_page_token='def'), compute.RegionInstanceGroupManagersListInstancesResponse(managed_instances=[compute.ManagedInstance()], next_page_token='ghi'), compute.RegionInstanceGroupManagersListInstancesResponse(managed_instances=[compute.ManagedInstance(), compute.ManagedInstance()]))
        response = response + response
        response = tuple((compute.RegionInstanceGroupManagersListInstancesResponse.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        pager = client.list_managed_instances(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, compute.ManagedInstance) for i in results))
        pages = list(client.list_managed_instances(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [compute.ListPerInstanceConfigsRegionInstanceGroupManagersRequest, dict])
def test_list_per_instance_configs_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.RegionInstanceGroupManagersListInstanceConfigsResp(next_page_token='next_page_token_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.RegionInstanceGroupManagersListInstanceConfigsResp.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_per_instance_configs(request)
    assert isinstance(response, pagers.ListPerInstanceConfigsPager)
    assert response.next_page_token == 'next_page_token_value'

def test_list_per_instance_configs_rest_required_fields(request_type=compute.ListPerInstanceConfigsRegionInstanceGroupManagersRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_per_instance_configs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_per_instance_configs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'max_results', 'order_by', 'page_token', 'return_partial_success'))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.RegionInstanceGroupManagersListInstanceConfigsResp()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.RegionInstanceGroupManagersListInstanceConfigsResp.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_per_instance_configs(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_per_instance_configs_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_per_instance_configs._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess')) & set(('instanceGroupManager', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_per_instance_configs_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_list_per_instance_configs') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_list_per_instance_configs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.ListPerInstanceConfigsRegionInstanceGroupManagersRequest.pb(compute.ListPerInstanceConfigsRegionInstanceGroupManagersRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.RegionInstanceGroupManagersListInstanceConfigsResp.to_json(compute.RegionInstanceGroupManagersListInstanceConfigsResp())
        request = compute.ListPerInstanceConfigsRegionInstanceGroupManagersRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.RegionInstanceGroupManagersListInstanceConfigsResp()
        client.list_per_instance_configs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_per_instance_configs_rest_bad_request(transport: str='rest', request_type=compute.ListPerInstanceConfigsRegionInstanceGroupManagersRequest):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_per_instance_configs(request)

def test_list_per_instance_configs_rest_flattened():
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.RegionInstanceGroupManagersListInstanceConfigsResp()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.RegionInstanceGroupManagersListInstanceConfigsResp.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_per_instance_configs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/listPerInstanceConfigs' % client.transport._host, args[1])

def test_list_per_instance_configs_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_per_instance_configs(compute.ListPerInstanceConfigsRegionInstanceGroupManagersRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value')

def test_list_per_instance_configs_rest_pager(transport: str='rest'):
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (compute.RegionInstanceGroupManagersListInstanceConfigsResp(items=[compute.PerInstanceConfig(), compute.PerInstanceConfig(), compute.PerInstanceConfig()], next_page_token='abc'), compute.RegionInstanceGroupManagersListInstanceConfigsResp(items=[], next_page_token='def'), compute.RegionInstanceGroupManagersListInstanceConfigsResp(items=[compute.PerInstanceConfig()], next_page_token='ghi'), compute.RegionInstanceGroupManagersListInstanceConfigsResp(items=[compute.PerInstanceConfig(), compute.PerInstanceConfig()]))
        response = response + response
        response = tuple((compute.RegionInstanceGroupManagersListInstanceConfigsResp.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        pager = client.list_per_instance_configs(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, compute.PerInstanceConfig) for i in results))
        pages = list(client.list_per_instance_configs(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [compute.PatchRegionInstanceGroupManagerRequest, dict])
def test_patch_rest(request_type):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['instance_group_manager_resource'] = {'auto_healing_policies': [{'health_check': 'health_check_value', 'initial_delay_sec': 1778}], 'base_instance_name': 'base_instance_name_value', 'creation_timestamp': 'creation_timestamp_value', 'current_actions': {'abandoning': 1041, 'creating': 845, 'creating_without_retries': 2589, 'deleting': 844, 'none': 432, 'recreating': 1060, 'refreshing': 1069, 'restarting': 1091, 'resuming': 874, 'starting': 876, 'stopping': 884, 'suspending': 1088, 'verifying': 979}, 'description': 'description_value', 'distribution_policy': {'target_shape': 'target_shape_value', 'zones': [{'zone': 'zone_value'}]}, 'fingerprint': 'fingerprint_value', 'id': 205, 'instance_group': 'instance_group_value', 'instance_lifecycle_policy': {'force_update_on_repair': 'force_update_on_repair_value'}, 'instance_template': 'instance_template_value', 'kind': 'kind_value', 'list_managed_instances_results': 'list_managed_instances_results_value', 'name': 'name_value', 'named_ports': [{'name': 'name_value', 'port': 453}], 'region': 'region_value', 'self_link': 'self_link_value', 'stateful_policy': {'preserved_state': {'disks': {}}}, 'status': {'autoscaler': 'autoscaler_value', 'is_stable': True, 'stateful': {'has_stateful_config': True, 'per_instance_configs': {'all_effective': True}}, 'version_target': {'is_reached': True}}, 'target_pools': ['target_pools_value1', 'target_pools_value2'], 'target_size': 1185, 'update_policy': {'instance_redistribution_type': 'instance_redistribution_type_value', 'max_surge': {'calculated': 1042, 'fixed': 528, 'percent': 753}, 'max_unavailable': {}, 'minimal_action': 'minimal_action_value', 'most_disruptive_allowed_action': 'most_disruptive_allowed_action_value', 'replacement_method': 'replacement_method_value', 'type_': 'type__value'}, 'versions': [{'instance_template': 'instance_template_value', 'name': 'name_value', 'target_size': {}}], 'zone': 'zone_value'}
    test_field = compute.PatchRegionInstanceGroupManagerRequest.meta.fields['instance_group_manager_resource']

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
    for (field, value) in request_init['instance_group_manager_resource'].items():
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
                for i in range(0, len(request_init['instance_group_manager_resource'][field])):
                    del request_init['instance_group_manager_resource'][field][i][subfield]
            else:
                del request_init['instance_group_manager_resource'][field][subfield]
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

def test_patch_rest_required_fields(request_type=compute.PatchRegionInstanceGroupManagerRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.patch._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'instanceGroupManagerResource', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_patch_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_patch') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_patch') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.PatchRegionInstanceGroupManagerRequest.pb(compute.PatchRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.PatchRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.patch(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_patch_rest_bad_request(transport: str='rest', request_type=compute.PatchRegionInstanceGroupManagerRequest):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
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
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', instance_group_manager_resource=compute.InstanceGroupManager(auto_healing_policies=[compute.InstanceGroupManagerAutoHealingPolicy(health_check='health_check_value')]))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}' % client.transport._host, args[1])

def test_patch_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.patch(compute.PatchRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', instance_group_manager_resource=compute.InstanceGroupManager(auto_healing_policies=[compute.InstanceGroupManagerAutoHealingPolicy(health_check='health_check_value')]))

def test_patch_rest_error():
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.PatchRegionInstanceGroupManagerRequest, dict])
def test_patch_unary_rest(request_type):
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['instance_group_manager_resource'] = {'auto_healing_policies': [{'health_check': 'health_check_value', 'initial_delay_sec': 1778}], 'base_instance_name': 'base_instance_name_value', 'creation_timestamp': 'creation_timestamp_value', 'current_actions': {'abandoning': 1041, 'creating': 845, 'creating_without_retries': 2589, 'deleting': 844, 'none': 432, 'recreating': 1060, 'refreshing': 1069, 'restarting': 1091, 'resuming': 874, 'starting': 876, 'stopping': 884, 'suspending': 1088, 'verifying': 979}, 'description': 'description_value', 'distribution_policy': {'target_shape': 'target_shape_value', 'zones': [{'zone': 'zone_value'}]}, 'fingerprint': 'fingerprint_value', 'id': 205, 'instance_group': 'instance_group_value', 'instance_lifecycle_policy': {'force_update_on_repair': 'force_update_on_repair_value'}, 'instance_template': 'instance_template_value', 'kind': 'kind_value', 'list_managed_instances_results': 'list_managed_instances_results_value', 'name': 'name_value', 'named_ports': [{'name': 'name_value', 'port': 453}], 'region': 'region_value', 'self_link': 'self_link_value', 'stateful_policy': {'preserved_state': {'disks': {}}}, 'status': {'autoscaler': 'autoscaler_value', 'is_stable': True, 'stateful': {'has_stateful_config': True, 'per_instance_configs': {'all_effective': True}}, 'version_target': {'is_reached': True}}, 'target_pools': ['target_pools_value1', 'target_pools_value2'], 'target_size': 1185, 'update_policy': {'instance_redistribution_type': 'instance_redistribution_type_value', 'max_surge': {'calculated': 1042, 'fixed': 528, 'percent': 753}, 'max_unavailable': {}, 'minimal_action': 'minimal_action_value', 'most_disruptive_allowed_action': 'most_disruptive_allowed_action_value', 'replacement_method': 'replacement_method_value', 'type_': 'type__value'}, 'versions': [{'instance_template': 'instance_template_value', 'name': 'name_value', 'target_size': {}}], 'zone': 'zone_value'}
    test_field = compute.PatchRegionInstanceGroupManagerRequest.meta.fields['instance_group_manager_resource']

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
    for (field, value) in request_init['instance_group_manager_resource'].items():
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
                for i in range(0, len(request_init['instance_group_manager_resource'][field])):
                    del request_init['instance_group_manager_resource'][field][i][subfield]
            else:
                del request_init['instance_group_manager_resource'][field][subfield]
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

def test_patch_unary_rest_required_fields(request_type=compute.PatchRegionInstanceGroupManagerRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.patch._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'instanceGroupManagerResource', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_patch_unary_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_patch') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_patch') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.PatchRegionInstanceGroupManagerRequest.pb(compute.PatchRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.PatchRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.patch_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_patch_unary_rest_bad_request(transport: str='rest', request_type=compute.PatchRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.patch_unary(request)

def test_patch_unary_rest_flattened():
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', instance_group_manager_resource=compute.InstanceGroupManager(auto_healing_policies=[compute.InstanceGroupManagerAutoHealingPolicy(health_check='health_check_value')]))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}' % client.transport._host, args[1])

def test_patch_unary_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.patch_unary(compute.PatchRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', instance_group_manager_resource=compute.InstanceGroupManager(auto_healing_policies=[compute.InstanceGroupManagerAutoHealingPolicy(health_check='health_check_value')]))

def test_patch_unary_rest_error():
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.PatchPerInstanceConfigsRegionInstanceGroupManagerRequest, dict])
def test_patch_per_instance_configs_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_manager_patch_instance_config_req_resource'] = {'per_instance_configs': [{'fingerprint': 'fingerprint_value', 'name': 'name_value', 'preserved_state': {'disks': {}, 'metadata': {}}, 'status': 'status_value'}]}
    test_field = compute.PatchPerInstanceConfigsRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_manager_patch_instance_config_req_resource']

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
    for (field, value) in request_init['region_instance_group_manager_patch_instance_config_req_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_manager_patch_instance_config_req_resource'][field])):
                    del request_init['region_instance_group_manager_patch_instance_config_req_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_manager_patch_instance_config_req_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.patch_per_instance_configs(request)
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

def test_patch_per_instance_configs_rest_required_fields(request_type=compute.PatchPerInstanceConfigsRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch_per_instance_configs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch_per_instance_configs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.patch_per_instance_configs(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_patch_per_instance_configs_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.patch_per_instance_configs._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagerPatchInstanceConfigReqResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_patch_per_instance_configs_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_patch_per_instance_configs') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_patch_per_instance_configs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.PatchPerInstanceConfigsRegionInstanceGroupManagerRequest.pb(compute.PatchPerInstanceConfigsRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.PatchPerInstanceConfigsRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.patch_per_instance_configs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_patch_per_instance_configs_rest_bad_request(transport: str='rest', request_type=compute.PatchPerInstanceConfigsRegionInstanceGroupManagerRequest):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.patch_per_instance_configs(request)

def test_patch_per_instance_configs_rest_flattened():
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_manager_patch_instance_config_req_resource=compute.RegionInstanceGroupManagerPatchInstanceConfigReq(per_instance_configs=[compute.PerInstanceConfig(fingerprint='fingerprint_value')]))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.patch_per_instance_configs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/patchPerInstanceConfigs' % client.transport._host, args[1])

def test_patch_per_instance_configs_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.patch_per_instance_configs(compute.PatchPerInstanceConfigsRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_manager_patch_instance_config_req_resource=compute.RegionInstanceGroupManagerPatchInstanceConfigReq(per_instance_configs=[compute.PerInstanceConfig(fingerprint='fingerprint_value')]))

def test_patch_per_instance_configs_rest_error():
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.PatchPerInstanceConfigsRegionInstanceGroupManagerRequest, dict])
def test_patch_per_instance_configs_unary_rest(request_type):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_manager_patch_instance_config_req_resource'] = {'per_instance_configs': [{'fingerprint': 'fingerprint_value', 'name': 'name_value', 'preserved_state': {'disks': {}, 'metadata': {}}, 'status': 'status_value'}]}
    test_field = compute.PatchPerInstanceConfigsRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_manager_patch_instance_config_req_resource']

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
    for (field, value) in request_init['region_instance_group_manager_patch_instance_config_req_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_manager_patch_instance_config_req_resource'][field])):
                    del request_init['region_instance_group_manager_patch_instance_config_req_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_manager_patch_instance_config_req_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.patch_per_instance_configs_unary(request)
    assert isinstance(response, compute.Operation)

def test_patch_per_instance_configs_unary_rest_required_fields(request_type=compute.PatchPerInstanceConfigsRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch_per_instance_configs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch_per_instance_configs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.patch_per_instance_configs_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_patch_per_instance_configs_unary_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.patch_per_instance_configs._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagerPatchInstanceConfigReqResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_patch_per_instance_configs_unary_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_patch_per_instance_configs') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_patch_per_instance_configs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.PatchPerInstanceConfigsRegionInstanceGroupManagerRequest.pb(compute.PatchPerInstanceConfigsRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.PatchPerInstanceConfigsRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.patch_per_instance_configs_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_patch_per_instance_configs_unary_rest_bad_request(transport: str='rest', request_type=compute.PatchPerInstanceConfigsRegionInstanceGroupManagerRequest):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.patch_per_instance_configs_unary(request)

def test_patch_per_instance_configs_unary_rest_flattened():
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_manager_patch_instance_config_req_resource=compute.RegionInstanceGroupManagerPatchInstanceConfigReq(per_instance_configs=[compute.PerInstanceConfig(fingerprint='fingerprint_value')]))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.patch_per_instance_configs_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/patchPerInstanceConfigs' % client.transport._host, args[1])

def test_patch_per_instance_configs_unary_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.patch_per_instance_configs_unary(compute.PatchPerInstanceConfigsRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_manager_patch_instance_config_req_resource=compute.RegionInstanceGroupManagerPatchInstanceConfigReq(per_instance_configs=[compute.PerInstanceConfig(fingerprint='fingerprint_value')]))

def test_patch_per_instance_configs_unary_rest_error():
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.RecreateInstancesRegionInstanceGroupManagerRequest, dict])
def test_recreate_instances_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_managers_recreate_request_resource'] = {'instances': ['instances_value1', 'instances_value2']}
    test_field = compute.RecreateInstancesRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_managers_recreate_request_resource']

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
    for (field, value) in request_init['region_instance_group_managers_recreate_request_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_managers_recreate_request_resource'][field])):
                    del request_init['region_instance_group_managers_recreate_request_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_managers_recreate_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.recreate_instances(request)
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

def test_recreate_instances_rest_required_fields(request_type=compute.RecreateInstancesRegionInstanceGroupManagerRequest):
    if False:
        print('Hello World!')
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).recreate_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).recreate_instances._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.recreate_instances(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_recreate_instances_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.recreate_instances._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagersRecreateRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_recreate_instances_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_recreate_instances') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_recreate_instances') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.RecreateInstancesRegionInstanceGroupManagerRequest.pb(compute.RecreateInstancesRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.RecreateInstancesRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.recreate_instances(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_recreate_instances_rest_bad_request(transport: str='rest', request_type=compute.RecreateInstancesRegionInstanceGroupManagerRequest):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.recreate_instances(request)

def test_recreate_instances_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_recreate_request_resource=compute.RegionInstanceGroupManagersRecreateRequest(instances=['instances_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.recreate_instances(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/recreateInstances' % client.transport._host, args[1])

def test_recreate_instances_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.recreate_instances(compute.RecreateInstancesRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_recreate_request_resource=compute.RegionInstanceGroupManagersRecreateRequest(instances=['instances_value']))

def test_recreate_instances_rest_error():
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.RecreateInstancesRegionInstanceGroupManagerRequest, dict])
def test_recreate_instances_unary_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_managers_recreate_request_resource'] = {'instances': ['instances_value1', 'instances_value2']}
    test_field = compute.RecreateInstancesRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_managers_recreate_request_resource']

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
    for (field, value) in request_init['region_instance_group_managers_recreate_request_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_managers_recreate_request_resource'][field])):
                    del request_init['region_instance_group_managers_recreate_request_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_managers_recreate_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.recreate_instances_unary(request)
    assert isinstance(response, compute.Operation)

def test_recreate_instances_unary_rest_required_fields(request_type=compute.RecreateInstancesRegionInstanceGroupManagerRequest):
    if False:
        return 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).recreate_instances._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).recreate_instances._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.recreate_instances_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_recreate_instances_unary_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.recreate_instances._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagersRecreateRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_recreate_instances_unary_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_recreate_instances') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_recreate_instances') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.RecreateInstancesRegionInstanceGroupManagerRequest.pb(compute.RecreateInstancesRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.RecreateInstancesRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.recreate_instances_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_recreate_instances_unary_rest_bad_request(transport: str='rest', request_type=compute.RecreateInstancesRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.recreate_instances_unary(request)

def test_recreate_instances_unary_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_recreate_request_resource=compute.RegionInstanceGroupManagersRecreateRequest(instances=['instances_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.recreate_instances_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/recreateInstances' % client.transport._host, args[1])

def test_recreate_instances_unary_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.recreate_instances_unary(compute.RecreateInstancesRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_recreate_request_resource=compute.RegionInstanceGroupManagersRecreateRequest(instances=['instances_value']))

def test_recreate_instances_unary_rest_error():
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.ResizeRegionInstanceGroupManagerRequest, dict])
def test_resize_rest(request_type):
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.resize(request)
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

def test_resize_rest_required_fields(request_type=compute.ResizeRegionInstanceGroupManagerRequest):
    if False:
        return 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request_init['size'] = 0
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'size' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resize._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'size' in jsonified_request
    assert jsonified_request['size'] == request_init['size']
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['size'] = 443
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resize._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'size'))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'size' in jsonified_request
    assert jsonified_request['size'] == 443
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.resize(request)
            expected_params = [('size', str(0))]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_resize_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.resize._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'size')) & set(('instanceGroupManager', 'project', 'region', 'size'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_resize_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_resize') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_resize') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.ResizeRegionInstanceGroupManagerRequest.pb(compute.ResizeRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.ResizeRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.resize(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_resize_rest_bad_request(transport: str='rest', request_type=compute.ResizeRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.resize(request)

def test_resize_rest_flattened():
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', size=443)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.resize(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/resize' % client.transport._host, args[1])

def test_resize_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.resize(compute.ResizeRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', size=443)

def test_resize_rest_error():
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.ResizeRegionInstanceGroupManagerRequest, dict])
def test_resize_unary_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.resize_unary(request)
    assert isinstance(response, compute.Operation)

def test_resize_unary_rest_required_fields(request_type=compute.ResizeRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request_init['size'] = 0
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'size' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resize._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'size' in jsonified_request
    assert jsonified_request['size'] == request_init['size']
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['size'] = 443
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resize._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id', 'size'))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'size' in jsonified_request
    assert jsonified_request['size'] == 443
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.resize_unary(request)
            expected_params = [('size', str(0))]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_resize_unary_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.resize._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'size')) & set(('instanceGroupManager', 'project', 'region', 'size'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_resize_unary_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_resize') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_resize') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.ResizeRegionInstanceGroupManagerRequest.pb(compute.ResizeRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.ResizeRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.resize_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_resize_unary_rest_bad_request(transport: str='rest', request_type=compute.ResizeRegionInstanceGroupManagerRequest):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.resize_unary(request)

def test_resize_unary_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', size=443)
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.resize_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/resize' % client.transport._host, args[1])

def test_resize_unary_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.resize_unary(compute.ResizeRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', size=443)

def test_resize_unary_rest_error():
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetInstanceTemplateRegionInstanceGroupManagerRequest, dict])
def test_set_instance_template_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_managers_set_template_request_resource'] = {'instance_template': 'instance_template_value'}
    test_field = compute.SetInstanceTemplateRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_managers_set_template_request_resource']

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
    for (field, value) in request_init['region_instance_group_managers_set_template_request_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_managers_set_template_request_resource'][field])):
                    del request_init['region_instance_group_managers_set_template_request_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_managers_set_template_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_instance_template(request)
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

def test_set_instance_template_rest_required_fields(request_type=compute.SetInstanceTemplateRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_instance_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_instance_template._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.set_instance_template(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_instance_template_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_instance_template._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagersSetTemplateRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_instance_template_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_set_instance_template') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_set_instance_template') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetInstanceTemplateRegionInstanceGroupManagerRequest.pb(compute.SetInstanceTemplateRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SetInstanceTemplateRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.set_instance_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_instance_template_rest_bad_request(transport: str='rest', request_type=compute.SetInstanceTemplateRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_instance_template(request)

def test_set_instance_template_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_set_template_request_resource=compute.RegionInstanceGroupManagersSetTemplateRequest(instance_template='instance_template_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_instance_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/setInstanceTemplate' % client.transport._host, args[1])

def test_set_instance_template_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_instance_template(compute.SetInstanceTemplateRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_set_template_request_resource=compute.RegionInstanceGroupManagersSetTemplateRequest(instance_template='instance_template_value'))

def test_set_instance_template_rest_error():
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetInstanceTemplateRegionInstanceGroupManagerRequest, dict])
def test_set_instance_template_unary_rest(request_type):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_managers_set_template_request_resource'] = {'instance_template': 'instance_template_value'}
    test_field = compute.SetInstanceTemplateRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_managers_set_template_request_resource']

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
    for (field, value) in request_init['region_instance_group_managers_set_template_request_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_managers_set_template_request_resource'][field])):
                    del request_init['region_instance_group_managers_set_template_request_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_managers_set_template_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_instance_template_unary(request)
    assert isinstance(response, compute.Operation)

def test_set_instance_template_unary_rest_required_fields(request_type=compute.SetInstanceTemplateRegionInstanceGroupManagerRequest):
    if False:
        return 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_instance_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_instance_template._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.set_instance_template_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_instance_template_unary_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_instance_template._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagersSetTemplateRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_instance_template_unary_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_set_instance_template') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_set_instance_template') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetInstanceTemplateRegionInstanceGroupManagerRequest.pb(compute.SetInstanceTemplateRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SetInstanceTemplateRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.set_instance_template_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_instance_template_unary_rest_bad_request(transport: str='rest', request_type=compute.SetInstanceTemplateRegionInstanceGroupManagerRequest):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_instance_template_unary(request)

def test_set_instance_template_unary_rest_flattened():
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_set_template_request_resource=compute.RegionInstanceGroupManagersSetTemplateRequest(instance_template='instance_template_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_instance_template_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/setInstanceTemplate' % client.transport._host, args[1])

def test_set_instance_template_unary_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_instance_template_unary(compute.SetInstanceTemplateRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_set_template_request_resource=compute.RegionInstanceGroupManagersSetTemplateRequest(instance_template='instance_template_value'))

def test_set_instance_template_unary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetTargetPoolsRegionInstanceGroupManagerRequest, dict])
def test_set_target_pools_rest(request_type):
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_managers_set_target_pools_request_resource'] = {'fingerprint': 'fingerprint_value', 'target_pools': ['target_pools_value1', 'target_pools_value2']}
    test_field = compute.SetTargetPoolsRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_managers_set_target_pools_request_resource']

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
    for (field, value) in request_init['region_instance_group_managers_set_target_pools_request_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_managers_set_target_pools_request_resource'][field])):
                    del request_init['region_instance_group_managers_set_target_pools_request_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_managers_set_target_pools_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_target_pools(request)
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

def test_set_target_pools_rest_required_fields(request_type=compute.SetTargetPoolsRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_target_pools._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_target_pools._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.set_target_pools(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_target_pools_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_target_pools._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagersSetTargetPoolsRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_target_pools_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_set_target_pools') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_set_target_pools') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetTargetPoolsRegionInstanceGroupManagerRequest.pb(compute.SetTargetPoolsRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SetTargetPoolsRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.set_target_pools(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_target_pools_rest_bad_request(transport: str='rest', request_type=compute.SetTargetPoolsRegionInstanceGroupManagerRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_target_pools(request)

def test_set_target_pools_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_set_target_pools_request_resource=compute.RegionInstanceGroupManagersSetTargetPoolsRequest(fingerprint='fingerprint_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_target_pools(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/setTargetPools' % client.transport._host, args[1])

def test_set_target_pools_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_target_pools(compute.SetTargetPoolsRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_set_target_pools_request_resource=compute.RegionInstanceGroupManagersSetTargetPoolsRequest(fingerprint='fingerprint_value'))

def test_set_target_pools_rest_error():
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetTargetPoolsRegionInstanceGroupManagerRequest, dict])
def test_set_target_pools_unary_rest(request_type):
    if False:
        print('Hello World!')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_managers_set_target_pools_request_resource'] = {'fingerprint': 'fingerprint_value', 'target_pools': ['target_pools_value1', 'target_pools_value2']}
    test_field = compute.SetTargetPoolsRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_managers_set_target_pools_request_resource']

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
    for (field, value) in request_init['region_instance_group_managers_set_target_pools_request_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_managers_set_target_pools_request_resource'][field])):
                    del request_init['region_instance_group_managers_set_target_pools_request_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_managers_set_target_pools_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_target_pools_unary(request)
    assert isinstance(response, compute.Operation)

def test_set_target_pools_unary_rest_required_fields(request_type=compute.SetTargetPoolsRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_target_pools._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_target_pools._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.set_target_pools_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_target_pools_unary_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_target_pools._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagersSetTargetPoolsRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_target_pools_unary_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_set_target_pools') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_set_target_pools') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetTargetPoolsRegionInstanceGroupManagerRequest.pb(compute.SetTargetPoolsRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SetTargetPoolsRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.set_target_pools_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_target_pools_unary_rest_bad_request(transport: str='rest', request_type=compute.SetTargetPoolsRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_target_pools_unary(request)

def test_set_target_pools_unary_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_set_target_pools_request_resource=compute.RegionInstanceGroupManagersSetTargetPoolsRequest(fingerprint='fingerprint_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_target_pools_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/setTargetPools' % client.transport._host, args[1])

def test_set_target_pools_unary_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_target_pools_unary(compute.SetTargetPoolsRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_managers_set_target_pools_request_resource=compute.RegionInstanceGroupManagersSetTargetPoolsRequest(fingerprint='fingerprint_value'))

def test_set_target_pools_unary_rest_error():
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.UpdatePerInstanceConfigsRegionInstanceGroupManagerRequest, dict])
def test_update_per_instance_configs_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_manager_update_instance_config_req_resource'] = {'per_instance_configs': [{'fingerprint': 'fingerprint_value', 'name': 'name_value', 'preserved_state': {'disks': {}, 'metadata': {}}, 'status': 'status_value'}]}
    test_field = compute.UpdatePerInstanceConfigsRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_manager_update_instance_config_req_resource']

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
    for (field, value) in request_init['region_instance_group_manager_update_instance_config_req_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_manager_update_instance_config_req_resource'][field])):
                    del request_init['region_instance_group_manager_update_instance_config_req_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_manager_update_instance_config_req_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_per_instance_configs(request)
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

def test_update_per_instance_configs_rest_required_fields(request_type=compute.UpdatePerInstanceConfigsRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_per_instance_configs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_per_instance_configs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_per_instance_configs(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_per_instance_configs_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_per_instance_configs._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagerUpdateInstanceConfigReqResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_per_instance_configs_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_update_per_instance_configs') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_update_per_instance_configs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.UpdatePerInstanceConfigsRegionInstanceGroupManagerRequest.pb(compute.UpdatePerInstanceConfigsRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.UpdatePerInstanceConfigsRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.update_per_instance_configs(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_per_instance_configs_rest_bad_request(transport: str='rest', request_type=compute.UpdatePerInstanceConfigsRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_per_instance_configs(request)

def test_update_per_instance_configs_rest_flattened():
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_manager_update_instance_config_req_resource=compute.RegionInstanceGroupManagerUpdateInstanceConfigReq(per_instance_configs=[compute.PerInstanceConfig(fingerprint='fingerprint_value')]))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_per_instance_configs(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/updatePerInstanceConfigs' % client.transport._host, args[1])

def test_update_per_instance_configs_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_per_instance_configs(compute.UpdatePerInstanceConfigsRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_manager_update_instance_config_req_resource=compute.RegionInstanceGroupManagerUpdateInstanceConfigReq(per_instance_configs=[compute.PerInstanceConfig(fingerprint='fingerprint_value')]))

def test_update_per_instance_configs_rest_error():
    if False:
        while True:
            i = 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.UpdatePerInstanceConfigsRegionInstanceGroupManagerRequest, dict])
def test_update_per_instance_configs_unary_rest(request_type):
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request_init['region_instance_group_manager_update_instance_config_req_resource'] = {'per_instance_configs': [{'fingerprint': 'fingerprint_value', 'name': 'name_value', 'preserved_state': {'disks': {}, 'metadata': {}}, 'status': 'status_value'}]}
    test_field = compute.UpdatePerInstanceConfigsRegionInstanceGroupManagerRequest.meta.fields['region_instance_group_manager_update_instance_config_req_resource']

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
    for (field, value) in request_init['region_instance_group_manager_update_instance_config_req_resource'].items():
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
                for i in range(0, len(request_init['region_instance_group_manager_update_instance_config_req_resource'][field])):
                    del request_init['region_instance_group_manager_update_instance_config_req_resource'][field][i][subfield]
            else:
                del request_init['region_instance_group_manager_update_instance_config_req_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.update_per_instance_configs_unary(request)
    assert isinstance(response, compute.Operation)

def test_update_per_instance_configs_unary_rest_required_fields(request_type=compute.UpdatePerInstanceConfigsRegionInstanceGroupManagerRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionInstanceGroupManagersRestTransport
    request_init = {}
    request_init['instance_group_manager'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_per_instance_configs._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['instanceGroupManager'] = 'instance_group_manager_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update_per_instance_configs._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'instanceGroupManager' in jsonified_request
    assert jsonified_request['instanceGroupManager'] == 'instance_group_manager_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_per_instance_configs_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_per_instance_configs_unary_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update_per_instance_configs._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('instanceGroupManager', 'project', 'region', 'regionInstanceGroupManagerUpdateInstanceConfigReqResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_per_instance_configs_unary_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionInstanceGroupManagersRestInterceptor())
    client = RegionInstanceGroupManagersClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'post_update_per_instance_configs') as post, mock.patch.object(transports.RegionInstanceGroupManagersRestInterceptor, 'pre_update_per_instance_configs') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.UpdatePerInstanceConfigsRegionInstanceGroupManagerRequest.pb(compute.UpdatePerInstanceConfigsRegionInstanceGroupManagerRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.UpdatePerInstanceConfigsRegionInstanceGroupManagerRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.update_per_instance_configs_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_per_instance_configs_unary_rest_bad_request(transport: str='rest', request_type=compute.UpdatePerInstanceConfigsRegionInstanceGroupManagerRequest):
    if False:
        i = 10
        return i + 15
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_per_instance_configs_unary(request)

def test_update_per_instance_configs_unary_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'instance_group_manager': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_manager_update_instance_config_req_resource=compute.RegionInstanceGroupManagerUpdateInstanceConfigReq(per_instance_configs=[compute.PerInstanceConfig(fingerprint='fingerprint_value')]))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.update_per_instance_configs_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/instanceGroupManagers/{instance_group_manager}/updatePerInstanceConfigs' % client.transport._host, args[1])

def test_update_per_instance_configs_unary_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_per_instance_configs_unary(compute.UpdatePerInstanceConfigsRegionInstanceGroupManagerRequest(), project='project_value', region='region_value', instance_group_manager='instance_group_manager_value', region_instance_group_manager_update_instance_config_req_resource=compute.RegionInstanceGroupManagerUpdateInstanceConfigReq(per_instance_configs=[compute.PerInstanceConfig(fingerprint='fingerprint_value')]))

def test_update_per_instance_configs_unary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = RegionInstanceGroupManagersClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = RegionInstanceGroupManagersClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = RegionInstanceGroupManagersClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = RegionInstanceGroupManagersClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionInstanceGroupManagersRestTransport(credentials=ga_credentials.AnonymousCredentials())
    client = RegionInstanceGroupManagersClient(transport=transport)
    assert client.transport is transport

@pytest.mark.parametrize('transport_class', [transports.RegionInstanceGroupManagersRestTransport])
def test_transport_adc(transport_class):
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default') as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport_class()
        adc.assert_called_once()

@pytest.mark.parametrize('transport_name', ['rest'])
def test_transport_kind(transport_name):
    if False:
        while True:
            i = 10
    transport = RegionInstanceGroupManagersClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_region_instance_group_managers_base_transport_error():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.RegionInstanceGroupManagersTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_region_instance_group_managers_base_transport():
    if False:
        return 10
    with mock.patch('google.cloud.compute_v1.services.region_instance_group_managers.transports.RegionInstanceGroupManagersTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.RegionInstanceGroupManagersTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('abandon_instances', 'apply_updates_to_instances', 'create_instances', 'delete', 'delete_instances', 'delete_per_instance_configs', 'get', 'insert', 'list', 'list_errors', 'list_managed_instances', 'list_per_instance_configs', 'patch', 'patch_per_instance_configs', 'recreate_instances', 'resize', 'set_instance_template', 'set_target_pools', 'update_per_instance_configs')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_region_instance_group_managers_base_transport_with_credentials_file():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.compute_v1.services.region_instance_group_managers.transports.RegionInstanceGroupManagersTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.RegionInstanceGroupManagersTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id='octopus')

def test_region_instance_group_managers_base_transport_with_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.compute_v1.services.region_instance_group_managers.transports.RegionInstanceGroupManagersTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.RegionInstanceGroupManagersTransport()
        adc.assert_called_once()

def test_region_instance_group_managers_auth_adc():
    if False:
        return 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        RegionInstanceGroupManagersClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id=None)

def test_region_instance_group_managers_http_transport_client_cert_source_for_mtls():
    if False:
        while True:
            i = 10
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.RegionInstanceGroupManagersRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['rest'])
def test_region_instance_group_managers_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='compute.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('compute.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_region_instance_group_managers_host_with_port(transport_name):
    if False:
        return 10
    client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='compute.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('compute.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_region_instance_group_managers_client_transport_session_collision(transport_name):
    if False:
        return 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = RegionInstanceGroupManagersClient(credentials=creds1, transport=transport_name)
    client2 = RegionInstanceGroupManagersClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.abandon_instances._session
    session2 = client2.transport.abandon_instances._session
    assert session1 != session2
    session1 = client1.transport.apply_updates_to_instances._session
    session2 = client2.transport.apply_updates_to_instances._session
    assert session1 != session2
    session1 = client1.transport.create_instances._session
    session2 = client2.transport.create_instances._session
    assert session1 != session2
    session1 = client1.transport.delete._session
    session2 = client2.transport.delete._session
    assert session1 != session2
    session1 = client1.transport.delete_instances._session
    session2 = client2.transport.delete_instances._session
    assert session1 != session2
    session1 = client1.transport.delete_per_instance_configs._session
    session2 = client2.transport.delete_per_instance_configs._session
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
    session1 = client1.transport.list_errors._session
    session2 = client2.transport.list_errors._session
    assert session1 != session2
    session1 = client1.transport.list_managed_instances._session
    session2 = client2.transport.list_managed_instances._session
    assert session1 != session2
    session1 = client1.transport.list_per_instance_configs._session
    session2 = client2.transport.list_per_instance_configs._session
    assert session1 != session2
    session1 = client1.transport.patch._session
    session2 = client2.transport.patch._session
    assert session1 != session2
    session1 = client1.transport.patch_per_instance_configs._session
    session2 = client2.transport.patch_per_instance_configs._session
    assert session1 != session2
    session1 = client1.transport.recreate_instances._session
    session2 = client2.transport.recreate_instances._session
    assert session1 != session2
    session1 = client1.transport.resize._session
    session2 = client2.transport.resize._session
    assert session1 != session2
    session1 = client1.transport.set_instance_template._session
    session2 = client2.transport.set_instance_template._session
    assert session1 != session2
    session1 = client1.transport.set_target_pools._session
    session2 = client2.transport.set_target_pools._session
    assert session1 != session2
    session1 = client1.transport.update_per_instance_configs._session
    session2 = client2.transport.update_per_instance_configs._session
    assert session1 != session2

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = RegionInstanceGroupManagersClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'clam'}
    path = RegionInstanceGroupManagersClient.common_billing_account_path(**expected)
    actual = RegionInstanceGroupManagersClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        return 10
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = RegionInstanceGroupManagersClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        while True:
            i = 10
    expected = {'folder': 'octopus'}
    path = RegionInstanceGroupManagersClient.common_folder_path(**expected)
    actual = RegionInstanceGroupManagersClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        for i in range(10):
            print('nop')
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = RegionInstanceGroupManagersClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        print('Hello World!')
    expected = {'organization': 'nudibranch'}
    path = RegionInstanceGroupManagersClient.common_organization_path(**expected)
    actual = RegionInstanceGroupManagersClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = RegionInstanceGroupManagersClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'mussel'}
    path = RegionInstanceGroupManagersClient.common_project_path(**expected)
    actual = RegionInstanceGroupManagersClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = RegionInstanceGroupManagersClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = RegionInstanceGroupManagersClient.common_location_path(**expected)
    actual = RegionInstanceGroupManagersClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.RegionInstanceGroupManagersTransport, '_prep_wrapped_messages') as prep:
        client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.RegionInstanceGroupManagersTransport, '_prep_wrapped_messages') as prep:
        transport_class = RegionInstanceGroupManagersClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

def test_transport_close():
    if False:
        while True:
            i = 10
    transports = {'rest': '_session'}
    for (transport, close_name) in transports.items():
        client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest']
    for transport in transports:
        client = RegionInstanceGroupManagersClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(RegionInstanceGroupManagersClient, transports.RegionInstanceGroupManagersRestTransport)])
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