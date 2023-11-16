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
from google.cloud.compute_v1.services.region_disks import RegionDisksClient, pagers, transports
from google.cloud.compute_v1.types import compute

def client_cert_source_callback():
    if False:
        return 10
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
    assert RegionDisksClient._get_default_mtls_endpoint(None) is None
    assert RegionDisksClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert RegionDisksClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert RegionDisksClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert RegionDisksClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert RegionDisksClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(RegionDisksClient, 'rest')])
def test_region_disks_client_from_service_account_info(client_class, transport_name):
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

@pytest.mark.parametrize('transport_class,transport_name', [(transports.RegionDisksRestTransport, 'rest')])
def test_region_disks_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(RegionDisksClient, 'rest')])
def test_region_disks_client_from_service_account_file(client_class, transport_name):
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

def test_region_disks_client_get_transport_class():
    if False:
        while True:
            i = 10
    transport = RegionDisksClient.get_transport_class()
    available_transports = [transports.RegionDisksRestTransport]
    assert transport in available_transports
    transport = RegionDisksClient.get_transport_class('rest')
    assert transport == transports.RegionDisksRestTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(RegionDisksClient, transports.RegionDisksRestTransport, 'rest')])
@mock.patch.object(RegionDisksClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RegionDisksClient))
def test_region_disks_client_client_options(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    with mock.patch.object(RegionDisksClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(RegionDisksClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(RegionDisksClient, transports.RegionDisksRestTransport, 'rest', 'true'), (RegionDisksClient, transports.RegionDisksRestTransport, 'rest', 'false')])
@mock.patch.object(RegionDisksClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RegionDisksClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_region_disks_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [RegionDisksClient])
@mock.patch.object(RegionDisksClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(RegionDisksClient))
def test_region_disks_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(RegionDisksClient, transports.RegionDisksRestTransport, 'rest')])
def test_region_disks_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(RegionDisksClient, transports.RegionDisksRestTransport, 'rest', None)])
def test_region_disks_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('request_type', [compute.AddResourcePoliciesRegionDiskRequest, dict])
def test_add_resource_policies_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request_init['region_disks_add_resource_policies_request_resource'] = {'resource_policies': ['resource_policies_value1', 'resource_policies_value2']}
    test_field = compute.AddResourcePoliciesRegionDiskRequest.meta.fields['region_disks_add_resource_policies_request_resource']

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
    for (field, value) in request_init['region_disks_add_resource_policies_request_resource'].items():
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
                for i in range(0, len(request_init['region_disks_add_resource_policies_request_resource'][field])):
                    del request_init['region_disks_add_resource_policies_request_resource'][field][i][subfield]
            else:
                del request_init['region_disks_add_resource_policies_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.add_resource_policies(request)
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

def test_add_resource_policies_rest_required_fields(request_type=compute.AddResourcePoliciesRegionDiskRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['disk'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_resource_policies._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['disk'] = 'disk_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_resource_policies._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'disk' in jsonified_request
    assert jsonified_request['disk'] == 'disk_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.add_resource_policies(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_add_resource_policies_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.add_resource_policies._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('disk', 'project', 'region', 'regionDisksAddResourcePoliciesRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_add_resource_policies_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_add_resource_policies') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_add_resource_policies') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.AddResourcePoliciesRegionDiskRequest.pb(compute.AddResourcePoliciesRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.AddResourcePoliciesRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.add_resource_policies(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_add_resource_policies_rest_bad_request(transport: str='rest', request_type=compute.AddResourcePoliciesRegionDiskRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.add_resource_policies(request)

def test_add_resource_policies_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', disk='disk_value', region_disks_add_resource_policies_request_resource=compute.RegionDisksAddResourcePoliciesRequest(resource_policies=['resource_policies_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.add_resource_policies(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{disk}/addResourcePolicies' % client.transport._host, args[1])

def test_add_resource_policies_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.add_resource_policies(compute.AddResourcePoliciesRegionDiskRequest(), project='project_value', region='region_value', disk='disk_value', region_disks_add_resource_policies_request_resource=compute.RegionDisksAddResourcePoliciesRequest(resource_policies=['resource_policies_value']))

def test_add_resource_policies_rest_error():
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.AddResourcePoliciesRegionDiskRequest, dict])
def test_add_resource_policies_unary_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request_init['region_disks_add_resource_policies_request_resource'] = {'resource_policies': ['resource_policies_value1', 'resource_policies_value2']}
    test_field = compute.AddResourcePoliciesRegionDiskRequest.meta.fields['region_disks_add_resource_policies_request_resource']

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
    for (field, value) in request_init['region_disks_add_resource_policies_request_resource'].items():
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
                for i in range(0, len(request_init['region_disks_add_resource_policies_request_resource'][field])):
                    del request_init['region_disks_add_resource_policies_request_resource'][field][i][subfield]
            else:
                del request_init['region_disks_add_resource_policies_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.add_resource_policies_unary(request)
    assert isinstance(response, compute.Operation)

def test_add_resource_policies_unary_rest_required_fields(request_type=compute.AddResourcePoliciesRegionDiskRequest):
    if False:
        return 10
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['disk'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_resource_policies._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['disk'] = 'disk_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_resource_policies._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'disk' in jsonified_request
    assert jsonified_request['disk'] == 'disk_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.add_resource_policies_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_add_resource_policies_unary_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.add_resource_policies._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('disk', 'project', 'region', 'regionDisksAddResourcePoliciesRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_add_resource_policies_unary_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_add_resource_policies') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_add_resource_policies') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.AddResourcePoliciesRegionDiskRequest.pb(compute.AddResourcePoliciesRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.AddResourcePoliciesRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.add_resource_policies_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_add_resource_policies_unary_rest_bad_request(transport: str='rest', request_type=compute.AddResourcePoliciesRegionDiskRequest):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.add_resource_policies_unary(request)

def test_add_resource_policies_unary_rest_flattened():
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', disk='disk_value', region_disks_add_resource_policies_request_resource=compute.RegionDisksAddResourcePoliciesRequest(resource_policies=['resource_policies_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.add_resource_policies_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{disk}/addResourcePolicies' % client.transport._host, args[1])

def test_add_resource_policies_unary_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.add_resource_policies_unary(compute.AddResourcePoliciesRegionDiskRequest(), project='project_value', region='region_value', disk='disk_value', region_disks_add_resource_policies_request_resource=compute.RegionDisksAddResourcePoliciesRequest(resource_policies=['resource_policies_value']))

def test_add_resource_policies_unary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.BulkInsertRegionDiskRequest, dict])
def test_bulk_insert_rest(request_type):
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request_init['bulk_insert_disk_resource_resource'] = {'source_consistency_group_policy': 'source_consistency_group_policy_value'}
    test_field = compute.BulkInsertRegionDiskRequest.meta.fields['bulk_insert_disk_resource_resource']

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
    for (field, value) in request_init['bulk_insert_disk_resource_resource'].items():
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
                for i in range(0, len(request_init['bulk_insert_disk_resource_resource'][field])):
                    del request_init['bulk_insert_disk_resource_resource'][field][i][subfield]
            else:
                del request_init['bulk_insert_disk_resource_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.bulk_insert(request)
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

def test_bulk_insert_rest_required_fields(request_type=compute.BulkInsertRegionDiskRequest):
    if False:
        return 10
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).bulk_insert._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).bulk_insert._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.bulk_insert(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_bulk_insert_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.bulk_insert._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('bulkInsertDiskResourceResource', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_bulk_insert_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_bulk_insert') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_bulk_insert') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.BulkInsertRegionDiskRequest.pb(compute.BulkInsertRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.BulkInsertRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.bulk_insert(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_bulk_insert_rest_bad_request(transport: str='rest', request_type=compute.BulkInsertRegionDiskRequest):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.bulk_insert(request)

def test_bulk_insert_rest_flattened():
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        mock_args = dict(project='project_value', region='region_value', bulk_insert_disk_resource_resource=compute.BulkInsertDiskResource(source_consistency_group_policy='source_consistency_group_policy_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.bulk_insert(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/bulkInsert' % client.transport._host, args[1])

def test_bulk_insert_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.bulk_insert(compute.BulkInsertRegionDiskRequest(), project='project_value', region='region_value', bulk_insert_disk_resource_resource=compute.BulkInsertDiskResource(source_consistency_group_policy='source_consistency_group_policy_value'))

def test_bulk_insert_rest_error():
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.BulkInsertRegionDiskRequest, dict])
def test_bulk_insert_unary_rest(request_type):
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request_init['bulk_insert_disk_resource_resource'] = {'source_consistency_group_policy': 'source_consistency_group_policy_value'}
    test_field = compute.BulkInsertRegionDiskRequest.meta.fields['bulk_insert_disk_resource_resource']

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
    for (field, value) in request_init['bulk_insert_disk_resource_resource'].items():
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
                for i in range(0, len(request_init['bulk_insert_disk_resource_resource'][field])):
                    del request_init['bulk_insert_disk_resource_resource'][field][i][subfield]
            else:
                del request_init['bulk_insert_disk_resource_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.bulk_insert_unary(request)
    assert isinstance(response, compute.Operation)

def test_bulk_insert_unary_rest_required_fields(request_type=compute.BulkInsertRegionDiskRequest):
    if False:
        print('Hello World!')
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).bulk_insert._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).bulk_insert._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.bulk_insert_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_bulk_insert_unary_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.bulk_insert._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('bulkInsertDiskResourceResource', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_bulk_insert_unary_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_bulk_insert') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_bulk_insert') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.BulkInsertRegionDiskRequest.pb(compute.BulkInsertRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.BulkInsertRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.bulk_insert_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_bulk_insert_unary_rest_bad_request(transport: str='rest', request_type=compute.BulkInsertRegionDiskRequest):
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.bulk_insert_unary(request)

def test_bulk_insert_unary_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        mock_args = dict(project='project_value', region='region_value', bulk_insert_disk_resource_resource=compute.BulkInsertDiskResource(source_consistency_group_policy='source_consistency_group_policy_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.bulk_insert_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/bulkInsert' % client.transport._host, args[1])

def test_bulk_insert_unary_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.bulk_insert_unary(compute.BulkInsertRegionDiskRequest(), project='project_value', region='region_value', bulk_insert_disk_resource_resource=compute.BulkInsertDiskResource(source_consistency_group_policy='source_consistency_group_policy_value'))

def test_bulk_insert_unary_rest_error():
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.CreateSnapshotRegionDiskRequest, dict])
def test_create_snapshot_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request_init['snapshot_resource'] = {'architecture': 'architecture_value', 'auto_created': True, 'chain_name': 'chain_name_value', 'creation_size_bytes': 2037, 'creation_timestamp': 'creation_timestamp_value', 'description': 'description_value', 'disk_size_gb': 1261, 'download_bytes': 1502, 'id': 205, 'kind': 'kind_value', 'label_fingerprint': 'label_fingerprint_value', 'labels': {}, 'license_codes': [1361, 1362], 'licenses': ['licenses_value1', 'licenses_value2'], 'location_hint': 'location_hint_value', 'name': 'name_value', 'satisfies_pzs': True, 'self_link': 'self_link_value', 'snapshot_encryption_key': {'kms_key_name': 'kms_key_name_value', 'kms_key_service_account': 'kms_key_service_account_value', 'raw_key': 'raw_key_value', 'rsa_encrypted_key': 'rsa_encrypted_key_value', 'sha256': 'sha256_value'}, 'snapshot_type': 'snapshot_type_value', 'source_disk': 'source_disk_value', 'source_disk_encryption_key': {}, 'source_disk_id': 'source_disk_id_value', 'source_snapshot_schedule_policy': 'source_snapshot_schedule_policy_value', 'source_snapshot_schedule_policy_id': 'source_snapshot_schedule_policy_id_value', 'status': 'status_value', 'storage_bytes': 1403, 'storage_bytes_status': 'storage_bytes_status_value', 'storage_locations': ['storage_locations_value1', 'storage_locations_value2']}
    test_field = compute.CreateSnapshotRegionDiskRequest.meta.fields['snapshot_resource']

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
    for (field, value) in request_init['snapshot_resource'].items():
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
                for i in range(0, len(request_init['snapshot_resource'][field])):
                    del request_init['snapshot_resource'][field][i][subfield]
            else:
                del request_init['snapshot_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_snapshot(request)
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

def test_create_snapshot_rest_required_fields(request_type=compute.CreateSnapshotRegionDiskRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['disk'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_snapshot._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['disk'] = 'disk_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_snapshot._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'disk' in jsonified_request
    assert jsonified_request['disk'] == 'disk_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_snapshot(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_snapshot_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_snapshot._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('disk', 'project', 'region', 'snapshotResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_snapshot_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_create_snapshot') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_create_snapshot') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.CreateSnapshotRegionDiskRequest.pb(compute.CreateSnapshotRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.CreateSnapshotRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.create_snapshot(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_snapshot_rest_bad_request(transport: str='rest', request_type=compute.CreateSnapshotRegionDiskRequest):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_snapshot(request)

def test_create_snapshot_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', disk='disk_value', snapshot_resource=compute.Snapshot(architecture='architecture_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_snapshot(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{disk}/createSnapshot' % client.transport._host, args[1])

def test_create_snapshot_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_snapshot(compute.CreateSnapshotRegionDiskRequest(), project='project_value', region='region_value', disk='disk_value', snapshot_resource=compute.Snapshot(architecture='architecture_value'))

def test_create_snapshot_rest_error():
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.CreateSnapshotRegionDiskRequest, dict])
def test_create_snapshot_unary_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request_init['snapshot_resource'] = {'architecture': 'architecture_value', 'auto_created': True, 'chain_name': 'chain_name_value', 'creation_size_bytes': 2037, 'creation_timestamp': 'creation_timestamp_value', 'description': 'description_value', 'disk_size_gb': 1261, 'download_bytes': 1502, 'id': 205, 'kind': 'kind_value', 'label_fingerprint': 'label_fingerprint_value', 'labels': {}, 'license_codes': [1361, 1362], 'licenses': ['licenses_value1', 'licenses_value2'], 'location_hint': 'location_hint_value', 'name': 'name_value', 'satisfies_pzs': True, 'self_link': 'self_link_value', 'snapshot_encryption_key': {'kms_key_name': 'kms_key_name_value', 'kms_key_service_account': 'kms_key_service_account_value', 'raw_key': 'raw_key_value', 'rsa_encrypted_key': 'rsa_encrypted_key_value', 'sha256': 'sha256_value'}, 'snapshot_type': 'snapshot_type_value', 'source_disk': 'source_disk_value', 'source_disk_encryption_key': {}, 'source_disk_id': 'source_disk_id_value', 'source_snapshot_schedule_policy': 'source_snapshot_schedule_policy_value', 'source_snapshot_schedule_policy_id': 'source_snapshot_schedule_policy_id_value', 'status': 'status_value', 'storage_bytes': 1403, 'storage_bytes_status': 'storage_bytes_status_value', 'storage_locations': ['storage_locations_value1', 'storage_locations_value2']}
    test_field = compute.CreateSnapshotRegionDiskRequest.meta.fields['snapshot_resource']

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
    for (field, value) in request_init['snapshot_resource'].items():
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
                for i in range(0, len(request_init['snapshot_resource'][field])):
                    del request_init['snapshot_resource'][field][i][subfield]
            else:
                del request_init['snapshot_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.create_snapshot_unary(request)
    assert isinstance(response, compute.Operation)

def test_create_snapshot_unary_rest_required_fields(request_type=compute.CreateSnapshotRegionDiskRequest):
    if False:
        print('Hello World!')
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['disk'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_snapshot._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['disk'] = 'disk_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).create_snapshot._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'disk' in jsonified_request
    assert jsonified_request['disk'] == 'disk_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.create_snapshot_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_create_snapshot_unary_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.create_snapshot._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('disk', 'project', 'region', 'snapshotResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_create_snapshot_unary_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_create_snapshot') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_create_snapshot') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.CreateSnapshotRegionDiskRequest.pb(compute.CreateSnapshotRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.CreateSnapshotRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.create_snapshot_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_create_snapshot_unary_rest_bad_request(transport: str='rest', request_type=compute.CreateSnapshotRegionDiskRequest):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.create_snapshot_unary(request)

def test_create_snapshot_unary_rest_flattened():
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', disk='disk_value', snapshot_resource=compute.Snapshot(architecture='architecture_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.create_snapshot_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{disk}/createSnapshot' % client.transport._host, args[1])

def test_create_snapshot_unary_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.create_snapshot_unary(compute.CreateSnapshotRegionDiskRequest(), project='project_value', region='region_value', disk='disk_value', snapshot_resource=compute.Snapshot(architecture='architecture_value'))

def test_create_snapshot_unary_rest_error():
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.DeleteRegionDiskRequest, dict])
def test_delete_rest(request_type):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
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

def test_delete_rest_required_fields(request_type=compute.DeleteRegionDiskRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['disk'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['disk'] = 'disk_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'disk' in jsonified_request
    assert jsonified_request['disk'] == 'disk_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('disk', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_delete') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_delete') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteRegionDiskRequest.pb(compute.DeleteRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_rest_bad_request(transport: str='rest', request_type=compute.DeleteRegionDiskRequest):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
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
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', disk='disk_value')
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{disk}' % client.transport._host, args[1])

def test_delete_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete(compute.DeleteRegionDiskRequest(), project='project_value', region='region_value', disk='disk_value')

def test_delete_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.DeleteRegionDiskRequest, dict])
def test_delete_unary_rest(request_type):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
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

def test_delete_unary_rest_required_fields(request_type=compute.DeleteRegionDiskRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['disk'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['disk'] = 'disk_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'disk' in jsonified_request
    assert jsonified_request['disk'] == 'disk_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('disk', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_unary_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_delete') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_delete') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteRegionDiskRequest.pb(compute.DeleteRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_unary_rest_bad_request(transport: str='rest', request_type=compute.DeleteRegionDiskRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
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
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', disk='disk_value')
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{disk}' % client.transport._host, args[1])

def test_delete_unary_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_unary(compute.DeleteRegionDiskRequest(), project='project_value', region='region_value', disk='disk_value')

def test_delete_unary_rest_error():
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.GetRegionDiskRequest, dict])
def test_get_rest(request_type):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Disk(architecture='architecture_value', creation_timestamp='creation_timestamp_value', description='description_value', id=205, kind='kind_value', label_fingerprint='label_fingerprint_value', last_attach_timestamp='last_attach_timestamp_value', last_detach_timestamp='last_detach_timestamp_value', license_codes=[1360], licenses=['licenses_value'], location_hint='location_hint_value', name='name_value', options='options_value', physical_block_size_bytes=2663, provisioned_iops=1740, provisioned_throughput=2411, region='region_value', replica_zones=['replica_zones_value'], resource_policies=['resource_policies_value'], satisfies_pzs=True, self_link='self_link_value', size_gb=739, source_consistency_group_policy='source_consistency_group_policy_value', source_consistency_group_policy_id='source_consistency_group_policy_id_value', source_disk='source_disk_value', source_disk_id='source_disk_id_value', source_image='source_image_value', source_image_id='source_image_id_value', source_snapshot='source_snapshot_value', source_snapshot_id='source_snapshot_id_value', source_storage_object='source_storage_object_value', status='status_value', type_='type__value', users=['users_value'], zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Disk.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get(request)
    assert isinstance(response, compute.Disk)
    assert response.architecture == 'architecture_value'
    assert response.creation_timestamp == 'creation_timestamp_value'
    assert response.description == 'description_value'
    assert response.id == 205
    assert response.kind == 'kind_value'
    assert response.label_fingerprint == 'label_fingerprint_value'
    assert response.last_attach_timestamp == 'last_attach_timestamp_value'
    assert response.last_detach_timestamp == 'last_detach_timestamp_value'
    assert response.license_codes == [1360]
    assert response.licenses == ['licenses_value']
    assert response.location_hint == 'location_hint_value'
    assert response.name == 'name_value'
    assert response.options == 'options_value'
    assert response.physical_block_size_bytes == 2663
    assert response.provisioned_iops == 1740
    assert response.provisioned_throughput == 2411
    assert response.region == 'region_value'
    assert response.replica_zones == ['replica_zones_value']
    assert response.resource_policies == ['resource_policies_value']
    assert response.satisfies_pzs is True
    assert response.self_link == 'self_link_value'
    assert response.size_gb == 739
    assert response.source_consistency_group_policy == 'source_consistency_group_policy_value'
    assert response.source_consistency_group_policy_id == 'source_consistency_group_policy_id_value'
    assert response.source_disk == 'source_disk_value'
    assert response.source_disk_id == 'source_disk_id_value'
    assert response.source_image == 'source_image_value'
    assert response.source_image_id == 'source_image_id_value'
    assert response.source_snapshot == 'source_snapshot_value'
    assert response.source_snapshot_id == 'source_snapshot_id_value'
    assert response.source_storage_object == 'source_storage_object_value'
    assert response.status == 'status_value'
    assert response.type_ == 'type__value'
    assert response.users == ['users_value']
    assert response.zone == 'zone_value'

def test_get_rest_required_fields(request_type=compute.GetRegionDiskRequest):
    if False:
        print('Hello World!')
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['disk'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['disk'] = 'disk_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'disk' in jsonified_request
    assert jsonified_request['disk'] == 'disk_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.Disk()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.Disk.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('disk', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_get') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_get') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.GetRegionDiskRequest.pb(compute.GetRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Disk.to_json(compute.Disk())
        request = compute.GetRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Disk()
        client.get(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_rest_bad_request(transport: str='rest', request_type=compute.GetRegionDiskRequest):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.get(request)

def test_get_rest_flattened():
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Disk()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', disk='disk_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Disk.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{disk}' % client.transport._host, args[1])

def test_get_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get(compute.GetRegionDiskRequest(), project='project_value', region='region_value', disk='disk_value')

def test_get_rest_error():
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.GetIamPolicyRegionDiskRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'resource': 'sample3'}
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

def test_get_iam_policy_rest_required_fields(request_type=compute.GetIamPolicyRegionDiskRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('options_requested_policy_version',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(('optionsRequestedPolicyVersion',)) & set(('project', 'region', 'resource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_iam_policy_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_get_iam_policy') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_get_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.GetIamPolicyRegionDiskRequest.pb(compute.GetIamPolicyRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Policy.to_json(compute.Policy())
        request = compute.GetIamPolicyRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Policy()
        client.get_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=compute.GetIamPolicyRegionDiskRequest):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'resource': 'sample3'}
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
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Policy()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'resource': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', resource='resource_value')
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{resource}/getIamPolicy' % client.transport._host, args[1])

def test_get_iam_policy_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_iam_policy(compute.GetIamPolicyRegionDiskRequest(), project='project_value', region='region_value', resource='resource_value')

def test_get_iam_policy_rest_error():
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.InsertRegionDiskRequest, dict])
def test_insert_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request_init['disk_resource'] = {'architecture': 'architecture_value', 'async_primary_disk': {'consistency_group_policy': 'consistency_group_policy_value', 'consistency_group_policy_id': 'consistency_group_policy_id_value', 'disk': 'disk_value', 'disk_id': 'disk_id_value'}, 'async_secondary_disks': {}, 'creation_timestamp': 'creation_timestamp_value', 'description': 'description_value', 'disk_encryption_key': {'kms_key_name': 'kms_key_name_value', 'kms_key_service_account': 'kms_key_service_account_value', 'raw_key': 'raw_key_value', 'rsa_encrypted_key': 'rsa_encrypted_key_value', 'sha256': 'sha256_value'}, 'guest_os_features': [{'type_': 'type__value'}], 'id': 205, 'kind': 'kind_value', 'label_fingerprint': 'label_fingerprint_value', 'labels': {}, 'last_attach_timestamp': 'last_attach_timestamp_value', 'last_detach_timestamp': 'last_detach_timestamp_value', 'license_codes': [1361, 1362], 'licenses': ['licenses_value1', 'licenses_value2'], 'location_hint': 'location_hint_value', 'name': 'name_value', 'options': 'options_value', 'params': {'resource_manager_tags': {}}, 'physical_block_size_bytes': 2663, 'provisioned_iops': 1740, 'provisioned_throughput': 2411, 'region': 'region_value', 'replica_zones': ['replica_zones_value1', 'replica_zones_value2'], 'resource_policies': ['resource_policies_value1', 'resource_policies_value2'], 'resource_status': {'async_primary_disk': {'state': 'state_value'}, 'async_secondary_disks': {}}, 'satisfies_pzs': True, 'self_link': 'self_link_value', 'size_gb': 739, 'source_consistency_group_policy': 'source_consistency_group_policy_value', 'source_consistency_group_policy_id': 'source_consistency_group_policy_id_value', 'source_disk': 'source_disk_value', 'source_disk_id': 'source_disk_id_value', 'source_image': 'source_image_value', 'source_image_encryption_key': {}, 'source_image_id': 'source_image_id_value', 'source_snapshot': 'source_snapshot_value', 'source_snapshot_encryption_key': {}, 'source_snapshot_id': 'source_snapshot_id_value', 'source_storage_object': 'source_storage_object_value', 'status': 'status_value', 'type_': 'type__value', 'users': ['users_value1', 'users_value2'], 'zone': 'zone_value'}
    test_field = compute.InsertRegionDiskRequest.meta.fields['disk_resource']

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
    for (field, value) in request_init['disk_resource'].items():
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
                for i in range(0, len(request_init['disk_resource'][field])):
                    del request_init['disk_resource'][field][i][subfield]
            else:
                del request_init['disk_resource'][field][subfield]
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

def test_insert_rest_required_fields(request_type=compute.InsertRegionDiskRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegionDisksRestTransport
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
    assert not set(unset_fields) - set(('request_id', 'source_image'))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        print('Hello World!')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.insert._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'sourceImage')) & set(('diskResource', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_insert_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_insert') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_insert') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.InsertRegionDiskRequest.pb(compute.InsertRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.InsertRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.insert(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_insert_rest_bad_request(transport: str='rest', request_type=compute.InsertRegionDiskRequest):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        mock_args = dict(project='project_value', region='region_value', disk_resource=compute.Disk(architecture='architecture_value'))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks' % client.transport._host, args[1])

def test_insert_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.insert(compute.InsertRegionDiskRequest(), project='project_value', region='region_value', disk_resource=compute.Disk(architecture='architecture_value'))

def test_insert_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.InsertRegionDiskRequest, dict])
def test_insert_unary_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request_init['disk_resource'] = {'architecture': 'architecture_value', 'async_primary_disk': {'consistency_group_policy': 'consistency_group_policy_value', 'consistency_group_policy_id': 'consistency_group_policy_id_value', 'disk': 'disk_value', 'disk_id': 'disk_id_value'}, 'async_secondary_disks': {}, 'creation_timestamp': 'creation_timestamp_value', 'description': 'description_value', 'disk_encryption_key': {'kms_key_name': 'kms_key_name_value', 'kms_key_service_account': 'kms_key_service_account_value', 'raw_key': 'raw_key_value', 'rsa_encrypted_key': 'rsa_encrypted_key_value', 'sha256': 'sha256_value'}, 'guest_os_features': [{'type_': 'type__value'}], 'id': 205, 'kind': 'kind_value', 'label_fingerprint': 'label_fingerprint_value', 'labels': {}, 'last_attach_timestamp': 'last_attach_timestamp_value', 'last_detach_timestamp': 'last_detach_timestamp_value', 'license_codes': [1361, 1362], 'licenses': ['licenses_value1', 'licenses_value2'], 'location_hint': 'location_hint_value', 'name': 'name_value', 'options': 'options_value', 'params': {'resource_manager_tags': {}}, 'physical_block_size_bytes': 2663, 'provisioned_iops': 1740, 'provisioned_throughput': 2411, 'region': 'region_value', 'replica_zones': ['replica_zones_value1', 'replica_zones_value2'], 'resource_policies': ['resource_policies_value1', 'resource_policies_value2'], 'resource_status': {'async_primary_disk': {'state': 'state_value'}, 'async_secondary_disks': {}}, 'satisfies_pzs': True, 'self_link': 'self_link_value', 'size_gb': 739, 'source_consistency_group_policy': 'source_consistency_group_policy_value', 'source_consistency_group_policy_id': 'source_consistency_group_policy_id_value', 'source_disk': 'source_disk_value', 'source_disk_id': 'source_disk_id_value', 'source_image': 'source_image_value', 'source_image_encryption_key': {}, 'source_image_id': 'source_image_id_value', 'source_snapshot': 'source_snapshot_value', 'source_snapshot_encryption_key': {}, 'source_snapshot_id': 'source_snapshot_id_value', 'source_storage_object': 'source_storage_object_value', 'status': 'status_value', 'type_': 'type__value', 'users': ['users_value1', 'users_value2'], 'zone': 'zone_value'}
    test_field = compute.InsertRegionDiskRequest.meta.fields['disk_resource']

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
    for (field, value) in request_init['disk_resource'].items():
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
                for i in range(0, len(request_init['disk_resource'][field])):
                    del request_init['disk_resource'][field][i][subfield]
            else:
                del request_init['disk_resource'][field][subfield]
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

def test_insert_unary_rest_required_fields(request_type=compute.InsertRegionDiskRequest):
    if False:
        print('Hello World!')
    transport_class = transports.RegionDisksRestTransport
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
    assert not set(unset_fields) - set(('request_id', 'source_image'))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        return 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.insert._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId', 'sourceImage')) & set(('diskResource', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_insert_unary_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_insert') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_insert') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.InsertRegionDiskRequest.pb(compute.InsertRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.InsertRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.insert_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_insert_unary_rest_bad_request(transport: str='rest', request_type=compute.InsertRegionDiskRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        mock_args = dict(project='project_value', region='region_value', disk_resource=compute.Disk(architecture='architecture_value'))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks' % client.transport._host, args[1])

def test_insert_unary_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.insert_unary(compute.InsertRegionDiskRequest(), project='project_value', region='region_value', disk_resource=compute.Disk(architecture='architecture_value'))

def test_insert_unary_rest_error():
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.ListRegionDisksRequest, dict])
def test_list_rest(request_type):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.DiskList(id='id_value', kind='kind_value', next_page_token='next_page_token_value', self_link='self_link_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.DiskList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list(request)
    assert isinstance(response, pagers.ListPager)
    assert response.id == 'id_value'
    assert response.kind == 'kind_value'
    assert response.next_page_token == 'next_page_token_value'
    assert response.self_link == 'self_link_value'

def test_list_rest_required_fields(request_type=compute.ListRegionDisksRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegionDisksRestTransport
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
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.DiskList()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.DiskList.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess')) & set(('project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_list') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_list') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.ListRegionDisksRequest.pb(compute.ListRegionDisksRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.DiskList.to_json(compute.DiskList())
        request = compute.ListRegionDisksRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.DiskList()
        client.list(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_rest_bad_request(transport: str='rest', request_type=compute.ListRegionDisksRequest):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.DiskList()
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        mock_args = dict(project='project_value', region='region_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.DiskList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks' % client.transport._host, args[1])

def test_list_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list(compute.ListRegionDisksRequest(), project='project_value', region='region_value')

def test_list_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (compute.DiskList(items=[compute.Disk(), compute.Disk(), compute.Disk()], next_page_token='abc'), compute.DiskList(items=[], next_page_token='def'), compute.DiskList(items=[compute.Disk()], next_page_token='ghi'), compute.DiskList(items=[compute.Disk(), compute.Disk()]))
        response = response + response
        response = tuple((compute.DiskList.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        pager = client.list(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, compute.Disk) for i in results))
        pages = list(client.list(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [compute.RemoveResourcePoliciesRegionDiskRequest, dict])
def test_remove_resource_policies_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request_init['region_disks_remove_resource_policies_request_resource'] = {'resource_policies': ['resource_policies_value1', 'resource_policies_value2']}
    test_field = compute.RemoveResourcePoliciesRegionDiskRequest.meta.fields['region_disks_remove_resource_policies_request_resource']

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
    for (field, value) in request_init['region_disks_remove_resource_policies_request_resource'].items():
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
                for i in range(0, len(request_init['region_disks_remove_resource_policies_request_resource'][field])):
                    del request_init['region_disks_remove_resource_policies_request_resource'][field][i][subfield]
            else:
                del request_init['region_disks_remove_resource_policies_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.remove_resource_policies(request)
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

def test_remove_resource_policies_rest_required_fields(request_type=compute.RemoveResourcePoliciesRegionDiskRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['disk'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).remove_resource_policies._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['disk'] = 'disk_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).remove_resource_policies._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'disk' in jsonified_request
    assert jsonified_request['disk'] == 'disk_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.remove_resource_policies(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_remove_resource_policies_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.remove_resource_policies._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('disk', 'project', 'region', 'regionDisksRemoveResourcePoliciesRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_remove_resource_policies_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_remove_resource_policies') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_remove_resource_policies') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.RemoveResourcePoliciesRegionDiskRequest.pb(compute.RemoveResourcePoliciesRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.RemoveResourcePoliciesRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.remove_resource_policies(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_remove_resource_policies_rest_bad_request(transport: str='rest', request_type=compute.RemoveResourcePoliciesRegionDiskRequest):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.remove_resource_policies(request)

def test_remove_resource_policies_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', disk='disk_value', region_disks_remove_resource_policies_request_resource=compute.RegionDisksRemoveResourcePoliciesRequest(resource_policies=['resource_policies_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.remove_resource_policies(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{disk}/removeResourcePolicies' % client.transport._host, args[1])

def test_remove_resource_policies_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.remove_resource_policies(compute.RemoveResourcePoliciesRegionDiskRequest(), project='project_value', region='region_value', disk='disk_value', region_disks_remove_resource_policies_request_resource=compute.RegionDisksRemoveResourcePoliciesRequest(resource_policies=['resource_policies_value']))

def test_remove_resource_policies_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.RemoveResourcePoliciesRegionDiskRequest, dict])
def test_remove_resource_policies_unary_rest(request_type):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request_init['region_disks_remove_resource_policies_request_resource'] = {'resource_policies': ['resource_policies_value1', 'resource_policies_value2']}
    test_field = compute.RemoveResourcePoliciesRegionDiskRequest.meta.fields['region_disks_remove_resource_policies_request_resource']

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
    for (field, value) in request_init['region_disks_remove_resource_policies_request_resource'].items():
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
                for i in range(0, len(request_init['region_disks_remove_resource_policies_request_resource'][field])):
                    del request_init['region_disks_remove_resource_policies_request_resource'][field][i][subfield]
            else:
                del request_init['region_disks_remove_resource_policies_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.remove_resource_policies_unary(request)
    assert isinstance(response, compute.Operation)

def test_remove_resource_policies_unary_rest_required_fields(request_type=compute.RemoveResourcePoliciesRegionDiskRequest):
    if False:
        print('Hello World!')
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['disk'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).remove_resource_policies._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['disk'] = 'disk_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).remove_resource_policies._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'disk' in jsonified_request
    assert jsonified_request['disk'] == 'disk_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.remove_resource_policies_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_remove_resource_policies_unary_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.remove_resource_policies._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('disk', 'project', 'region', 'regionDisksRemoveResourcePoliciesRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_remove_resource_policies_unary_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_remove_resource_policies') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_remove_resource_policies') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.RemoveResourcePoliciesRegionDiskRequest.pb(compute.RemoveResourcePoliciesRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.RemoveResourcePoliciesRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.remove_resource_policies_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_remove_resource_policies_unary_rest_bad_request(transport: str='rest', request_type=compute.RemoveResourcePoliciesRegionDiskRequest):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.remove_resource_policies_unary(request)

def test_remove_resource_policies_unary_rest_flattened():
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', disk='disk_value', region_disks_remove_resource_policies_request_resource=compute.RegionDisksRemoveResourcePoliciesRequest(resource_policies=['resource_policies_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.remove_resource_policies_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{disk}/removeResourcePolicies' % client.transport._host, args[1])

def test_remove_resource_policies_unary_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.remove_resource_policies_unary(compute.RemoveResourcePoliciesRegionDiskRequest(), project='project_value', region='region_value', disk='disk_value', region_disks_remove_resource_policies_request_resource=compute.RegionDisksRemoveResourcePoliciesRequest(resource_policies=['resource_policies_value']))

def test_remove_resource_policies_unary_rest_error():
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.ResizeRegionDiskRequest, dict])
def test_resize_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request_init['region_disks_resize_request_resource'] = {'size_gb': 739}
    test_field = compute.ResizeRegionDiskRequest.meta.fields['region_disks_resize_request_resource']

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
    for (field, value) in request_init['region_disks_resize_request_resource'].items():
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
                for i in range(0, len(request_init['region_disks_resize_request_resource'][field])):
                    del request_init['region_disks_resize_request_resource'][field][i][subfield]
            else:
                del request_init['region_disks_resize_request_resource'][field][subfield]
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

def test_resize_rest_required_fields(request_type=compute.ResizeRegionDiskRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['disk'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resize._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['disk'] = 'disk_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resize._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'disk' in jsonified_request
    assert jsonified_request['disk'] == 'disk_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.resize(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_resize_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.resize._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('disk', 'project', 'region', 'regionDisksResizeRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_resize_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_resize') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_resize') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.ResizeRegionDiskRequest.pb(compute.ResizeRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.ResizeRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.resize(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_resize_rest_bad_request(transport: str='rest', request_type=compute.ResizeRegionDiskRequest):
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
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
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', disk='disk_value', region_disks_resize_request_resource=compute.RegionDisksResizeRequest(size_gb=739))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{disk}/resize' % client.transport._host, args[1])

def test_resize_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.resize(compute.ResizeRegionDiskRequest(), project='project_value', region='region_value', disk='disk_value', region_disks_resize_request_resource=compute.RegionDisksResizeRequest(size_gb=739))

def test_resize_rest_error():
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.ResizeRegionDiskRequest, dict])
def test_resize_unary_rest(request_type):
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request_init['region_disks_resize_request_resource'] = {'size_gb': 739}
    test_field = compute.ResizeRegionDiskRequest.meta.fields['region_disks_resize_request_resource']

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
    for (field, value) in request_init['region_disks_resize_request_resource'].items():
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
                for i in range(0, len(request_init['region_disks_resize_request_resource'][field])):
                    del request_init['region_disks_resize_request_resource'][field][i][subfield]
            else:
                del request_init['region_disks_resize_request_resource'][field][subfield]
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

def test_resize_unary_rest_required_fields(request_type=compute.ResizeRegionDiskRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['disk'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resize._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['disk'] = 'disk_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).resize._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'disk' in jsonified_request
    assert jsonified_request['disk'] == 'disk_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.resize_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_resize_unary_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.resize._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('disk', 'project', 'region', 'regionDisksResizeRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_resize_unary_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_resize') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_resize') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.ResizeRegionDiskRequest.pb(compute.ResizeRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.ResizeRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.resize_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_resize_unary_rest_bad_request(transport: str='rest', request_type=compute.ResizeRegionDiskRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.resize_unary(request)

def test_resize_unary_rest_flattened():
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', disk='disk_value', region_disks_resize_request_resource=compute.RegionDisksResizeRequest(size_gb=739))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{disk}/resize' % client.transport._host, args[1])

def test_resize_unary_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.resize_unary(compute.ResizeRegionDiskRequest(), project='project_value', region='region_value', disk='disk_value', region_disks_resize_request_resource=compute.RegionDisksResizeRequest(size_gb=739))

def test_resize_unary_rest_error():
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetIamPolicyRegionDiskRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'resource': 'sample3'}
    request_init['region_set_policy_request_resource'] = {'bindings': [{'binding_id': 'binding_id_value', 'condition': {'description': 'description_value', 'expression': 'expression_value', 'location': 'location_value', 'title': 'title_value'}, 'members': ['members_value1', 'members_value2'], 'role': 'role_value'}], 'etag': 'etag_value', 'policy': {'audit_configs': [{'audit_log_configs': [{'exempted_members': ['exempted_members_value1', 'exempted_members_value2'], 'ignore_child_exemptions': True, 'log_type': 'log_type_value'}], 'exempted_members': ['exempted_members_value1', 'exempted_members_value2'], 'service': 'service_value'}], 'bindings': {}, 'etag': 'etag_value', 'iam_owned': True, 'rules': [{'action': 'action_value', 'conditions': [{'iam': 'iam_value', 'op': 'op_value', 'svc': 'svc_value', 'sys': 'sys_value', 'values': ['values_value1', 'values_value2']}], 'description': 'description_value', 'ins': ['ins_value1', 'ins_value2'], 'log_configs': [{'cloud_audit': {'authorization_logging_options': {'permission_type': 'permission_type_value'}, 'log_name': 'log_name_value'}, 'counter': {'custom_fields': [{'name': 'name_value', 'value': 'value_value'}], 'field': 'field_value', 'metric': 'metric_value'}, 'data_access': {'log_mode': 'log_mode_value'}}], 'not_ins': ['not_ins_value1', 'not_ins_value2'], 'permissions': ['permissions_value1', 'permissions_value2']}], 'version': 774}}
    test_field = compute.SetIamPolicyRegionDiskRequest.meta.fields['region_set_policy_request_resource']

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
    for (field, value) in request_init['region_set_policy_request_resource'].items():
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
                for i in range(0, len(request_init['region_set_policy_request_resource'][field])):
                    del request_init['region_set_policy_request_resource'][field][i][subfield]
            else:
                del request_init['region_set_policy_request_resource'][field][subfield]
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

def test_set_iam_policy_rest_required_fields(request_type=compute.SetIamPolicyRegionDiskRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('project', 'region', 'regionSetPolicyRequestResource', 'resource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_iam_policy_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_set_iam_policy') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_set_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetIamPolicyRegionDiskRequest.pb(compute.SetIamPolicyRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Policy.to_json(compute.Policy())
        request = compute.SetIamPolicyRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Policy()
        client.set_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_iam_policy_rest_bad_request(transport: str='rest', request_type=compute.SetIamPolicyRegionDiskRequest):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'resource': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_iam_policy(request)

def test_set_iam_policy_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Policy()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'resource': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', resource='resource_value', region_set_policy_request_resource=compute.RegionSetPolicyRequest(bindings=[compute.Binding(binding_id='binding_id_value')]))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{resource}/setIamPolicy' % client.transport._host, args[1])

def test_set_iam_policy_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_iam_policy(compute.SetIamPolicyRegionDiskRequest(), project='project_value', region='region_value', resource='resource_value', region_set_policy_request_resource=compute.RegionSetPolicyRequest(bindings=[compute.Binding(binding_id='binding_id_value')]))

def test_set_iam_policy_rest_error():
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetLabelsRegionDiskRequest, dict])
def test_set_labels_rest(request_type):
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'resource': 'sample3'}
    request_init['region_set_labels_request_resource'] = {'label_fingerprint': 'label_fingerprint_value', 'labels': {}}
    test_field = compute.SetLabelsRegionDiskRequest.meta.fields['region_set_labels_request_resource']

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
    for (field, value) in request_init['region_set_labels_request_resource'].items():
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
                for i in range(0, len(request_init['region_set_labels_request_resource'][field])):
                    del request_init['region_set_labels_request_resource'][field][i][subfield]
            else:
                del request_init['region_set_labels_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_labels(request)
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

def test_set_labels_rest_required_fields(request_type=compute.SetLabelsRegionDiskRequest):
    if False:
        return 10
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_labels._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_labels._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.set_labels(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_labels_rest_unset_required_fields():
    if False:
        i = 10
        return i + 15
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_labels._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('project', 'region', 'regionSetLabelsRequestResource', 'resource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_labels_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_set_labels') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_set_labels') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetLabelsRegionDiskRequest.pb(compute.SetLabelsRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SetLabelsRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.set_labels(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_labels_rest_bad_request(transport: str='rest', request_type=compute.SetLabelsRegionDiskRequest):
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'resource': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_labels(request)

def test_set_labels_rest_flattened():
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'resource': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', resource='resource_value', region_set_labels_request_resource=compute.RegionSetLabelsRequest(label_fingerprint='label_fingerprint_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_labels(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{resource}/setLabels' % client.transport._host, args[1])

def test_set_labels_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_labels(compute.SetLabelsRegionDiskRequest(), project='project_value', region='region_value', resource='resource_value', region_set_labels_request_resource=compute.RegionSetLabelsRequest(label_fingerprint='label_fingerprint_value'))

def test_set_labels_rest_error():
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetLabelsRegionDiskRequest, dict])
def test_set_labels_unary_rest(request_type):
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'resource': 'sample3'}
    request_init['region_set_labels_request_resource'] = {'label_fingerprint': 'label_fingerprint_value', 'labels': {}}
    test_field = compute.SetLabelsRegionDiskRequest.meta.fields['region_set_labels_request_resource']

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
    for (field, value) in request_init['region_set_labels_request_resource'].items():
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
                for i in range(0, len(request_init['region_set_labels_request_resource'][field])):
                    del request_init['region_set_labels_request_resource'][field][i][subfield]
            else:
                del request_init['region_set_labels_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_labels_unary(request)
    assert isinstance(response, compute.Operation)

def test_set_labels_unary_rest_required_fields(request_type=compute.SetLabelsRegionDiskRequest):
    if False:
        print('Hello World!')
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_labels._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_labels._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.set_labels_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_labels_unary_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_labels._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('project', 'region', 'regionSetLabelsRequestResource', 'resource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_labels_unary_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_set_labels') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_set_labels') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetLabelsRegionDiskRequest.pb(compute.SetLabelsRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SetLabelsRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.set_labels_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_labels_unary_rest_bad_request(transport: str='rest', request_type=compute.SetLabelsRegionDiskRequest):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'resource': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_labels_unary(request)

def test_set_labels_unary_rest_flattened():
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'resource': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', resource='resource_value', region_set_labels_request_resource=compute.RegionSetLabelsRequest(label_fingerprint='label_fingerprint_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_labels_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{resource}/setLabels' % client.transport._host, args[1])

def test_set_labels_unary_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_labels_unary(compute.SetLabelsRegionDiskRequest(), project='project_value', region='region_value', resource='resource_value', region_set_labels_request_resource=compute.RegionSetLabelsRequest(label_fingerprint='label_fingerprint_value'))

def test_set_labels_unary_rest_error():
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.StartAsyncReplicationRegionDiskRequest, dict])
def test_start_async_replication_rest(request_type):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request_init['region_disks_start_async_replication_request_resource'] = {'async_secondary_disk': 'async_secondary_disk_value'}
    test_field = compute.StartAsyncReplicationRegionDiskRequest.meta.fields['region_disks_start_async_replication_request_resource']

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
    for (field, value) in request_init['region_disks_start_async_replication_request_resource'].items():
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
                for i in range(0, len(request_init['region_disks_start_async_replication_request_resource'][field])):
                    del request_init['region_disks_start_async_replication_request_resource'][field][i][subfield]
            else:
                del request_init['region_disks_start_async_replication_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.start_async_replication(request)
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

def test_start_async_replication_rest_required_fields(request_type=compute.StartAsyncReplicationRegionDiskRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['disk'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).start_async_replication._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['disk'] = 'disk_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).start_async_replication._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'disk' in jsonified_request
    assert jsonified_request['disk'] == 'disk_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.start_async_replication(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_start_async_replication_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.start_async_replication._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('disk', 'project', 'region', 'regionDisksStartAsyncReplicationRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_start_async_replication_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_start_async_replication') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_start_async_replication') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.StartAsyncReplicationRegionDiskRequest.pb(compute.StartAsyncReplicationRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.StartAsyncReplicationRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.start_async_replication(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_start_async_replication_rest_bad_request(transport: str='rest', request_type=compute.StartAsyncReplicationRegionDiskRequest):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.start_async_replication(request)

def test_start_async_replication_rest_flattened():
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', disk='disk_value', region_disks_start_async_replication_request_resource=compute.RegionDisksStartAsyncReplicationRequest(async_secondary_disk='async_secondary_disk_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.start_async_replication(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{disk}/startAsyncReplication' % client.transport._host, args[1])

def test_start_async_replication_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.start_async_replication(compute.StartAsyncReplicationRegionDiskRequest(), project='project_value', region='region_value', disk='disk_value', region_disks_start_async_replication_request_resource=compute.RegionDisksStartAsyncReplicationRequest(async_secondary_disk='async_secondary_disk_value'))

def test_start_async_replication_rest_error():
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.StartAsyncReplicationRegionDiskRequest, dict])
def test_start_async_replication_unary_rest(request_type):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request_init['region_disks_start_async_replication_request_resource'] = {'async_secondary_disk': 'async_secondary_disk_value'}
    test_field = compute.StartAsyncReplicationRegionDiskRequest.meta.fields['region_disks_start_async_replication_request_resource']

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
    for (field, value) in request_init['region_disks_start_async_replication_request_resource'].items():
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
                for i in range(0, len(request_init['region_disks_start_async_replication_request_resource'][field])):
                    del request_init['region_disks_start_async_replication_request_resource'][field][i][subfield]
            else:
                del request_init['region_disks_start_async_replication_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.start_async_replication_unary(request)
    assert isinstance(response, compute.Operation)

def test_start_async_replication_unary_rest_required_fields(request_type=compute.StartAsyncReplicationRegionDiskRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['disk'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).start_async_replication._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['disk'] = 'disk_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).start_async_replication._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'disk' in jsonified_request
    assert jsonified_request['disk'] == 'disk_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.start_async_replication_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_start_async_replication_unary_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.start_async_replication._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('disk', 'project', 'region', 'regionDisksStartAsyncReplicationRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_start_async_replication_unary_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_start_async_replication') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_start_async_replication') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.StartAsyncReplicationRegionDiskRequest.pb(compute.StartAsyncReplicationRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.StartAsyncReplicationRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.start_async_replication_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_start_async_replication_unary_rest_bad_request(transport: str='rest', request_type=compute.StartAsyncReplicationRegionDiskRequest):
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.start_async_replication_unary(request)

def test_start_async_replication_unary_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', disk='disk_value', region_disks_start_async_replication_request_resource=compute.RegionDisksStartAsyncReplicationRequest(async_secondary_disk='async_secondary_disk_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.start_async_replication_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{disk}/startAsyncReplication' % client.transport._host, args[1])

def test_start_async_replication_unary_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.start_async_replication_unary(compute.StartAsyncReplicationRegionDiskRequest(), project='project_value', region='region_value', disk='disk_value', region_disks_start_async_replication_request_resource=compute.RegionDisksStartAsyncReplicationRequest(async_secondary_disk='async_secondary_disk_value'))

def test_start_async_replication_unary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.StopAsyncReplicationRegionDiskRequest, dict])
def test_stop_async_replication_rest(request_type):
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.stop_async_replication(request)
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

def test_stop_async_replication_rest_required_fields(request_type=compute.StopAsyncReplicationRegionDiskRequest):
    if False:
        print('Hello World!')
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['disk'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).stop_async_replication._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['disk'] = 'disk_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).stop_async_replication._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'disk' in jsonified_request
    assert jsonified_request['disk'] == 'disk_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.stop_async_replication(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_stop_async_replication_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.stop_async_replication._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('disk', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_stop_async_replication_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_stop_async_replication') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_stop_async_replication') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.StopAsyncReplicationRegionDiskRequest.pb(compute.StopAsyncReplicationRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.StopAsyncReplicationRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.stop_async_replication(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_stop_async_replication_rest_bad_request(transport: str='rest', request_type=compute.StopAsyncReplicationRegionDiskRequest):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.stop_async_replication(request)

def test_stop_async_replication_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', disk='disk_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.stop_async_replication(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{disk}/stopAsyncReplication' % client.transport._host, args[1])

def test_stop_async_replication_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.stop_async_replication(compute.StopAsyncReplicationRegionDiskRequest(), project='project_value', region='region_value', disk='disk_value')

def test_stop_async_replication_rest_error():
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.StopAsyncReplicationRegionDiskRequest, dict])
def test_stop_async_replication_unary_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.stop_async_replication_unary(request)
    assert isinstance(response, compute.Operation)

def test_stop_async_replication_unary_rest_required_fields(request_type=compute.StopAsyncReplicationRegionDiskRequest):
    if False:
        return 10
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['disk'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).stop_async_replication._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['disk'] = 'disk_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).stop_async_replication._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'disk' in jsonified_request
    assert jsonified_request['disk'] == 'disk_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.stop_async_replication_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_stop_async_replication_unary_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.stop_async_replication._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('disk', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_stop_async_replication_unary_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_stop_async_replication') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_stop_async_replication') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.StopAsyncReplicationRegionDiskRequest.pb(compute.StopAsyncReplicationRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.StopAsyncReplicationRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.stop_async_replication_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_stop_async_replication_unary_rest_bad_request(transport: str='rest', request_type=compute.StopAsyncReplicationRegionDiskRequest):
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.stop_async_replication_unary(request)

def test_stop_async_replication_unary_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', disk='disk_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.stop_async_replication_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{disk}/stopAsyncReplication' % client.transport._host, args[1])

def test_stop_async_replication_unary_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.stop_async_replication_unary(compute.StopAsyncReplicationRegionDiskRequest(), project='project_value', region='region_value', disk='disk_value')

def test_stop_async_replication_unary_rest_error():
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.StopGroupAsyncReplicationRegionDiskRequest, dict])
def test_stop_group_async_replication_rest(request_type):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request_init['disks_stop_group_async_replication_resource_resource'] = {'resource_policy': 'resource_policy_value'}
    test_field = compute.StopGroupAsyncReplicationRegionDiskRequest.meta.fields['disks_stop_group_async_replication_resource_resource']

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
    for (field, value) in request_init['disks_stop_group_async_replication_resource_resource'].items():
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
                for i in range(0, len(request_init['disks_stop_group_async_replication_resource_resource'][field])):
                    del request_init['disks_stop_group_async_replication_resource_resource'][field][i][subfield]
            else:
                del request_init['disks_stop_group_async_replication_resource_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.stop_group_async_replication(request)
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

def test_stop_group_async_replication_rest_required_fields(request_type=compute.StopGroupAsyncReplicationRegionDiskRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).stop_group_async_replication._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).stop_group_async_replication._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.stop_group_async_replication(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_stop_group_async_replication_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.stop_group_async_replication._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('disksStopGroupAsyncReplicationResourceResource', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_stop_group_async_replication_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_stop_group_async_replication') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_stop_group_async_replication') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.StopGroupAsyncReplicationRegionDiskRequest.pb(compute.StopGroupAsyncReplicationRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.StopGroupAsyncReplicationRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.stop_group_async_replication(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_stop_group_async_replication_rest_bad_request(transport: str='rest', request_type=compute.StopGroupAsyncReplicationRegionDiskRequest):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.stop_group_async_replication(request)

def test_stop_group_async_replication_rest_flattened():
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        mock_args = dict(project='project_value', region='region_value', disks_stop_group_async_replication_resource_resource=compute.DisksStopGroupAsyncReplicationResource(resource_policy='resource_policy_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.stop_group_async_replication(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/stopGroupAsyncReplication' % client.transport._host, args[1])

def test_stop_group_async_replication_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.stop_group_async_replication(compute.StopGroupAsyncReplicationRegionDiskRequest(), project='project_value', region='region_value', disks_stop_group_async_replication_resource_resource=compute.DisksStopGroupAsyncReplicationResource(resource_policy='resource_policy_value'))

def test_stop_group_async_replication_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.StopGroupAsyncReplicationRegionDiskRequest, dict])
def test_stop_group_async_replication_unary_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request_init['disks_stop_group_async_replication_resource_resource'] = {'resource_policy': 'resource_policy_value'}
    test_field = compute.StopGroupAsyncReplicationRegionDiskRequest.meta.fields['disks_stop_group_async_replication_resource_resource']

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
    for (field, value) in request_init['disks_stop_group_async_replication_resource_resource'].items():
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
                for i in range(0, len(request_init['disks_stop_group_async_replication_resource_resource'][field])):
                    del request_init['disks_stop_group_async_replication_resource_resource'][field][i][subfield]
            else:
                del request_init['disks_stop_group_async_replication_resource_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.stop_group_async_replication_unary(request)
    assert isinstance(response, compute.Operation)

def test_stop_group_async_replication_unary_rest_required_fields(request_type=compute.StopGroupAsyncReplicationRegionDiskRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).stop_group_async_replication._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).stop_group_async_replication._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.stop_group_async_replication_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_stop_group_async_replication_unary_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.stop_group_async_replication._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('disksStopGroupAsyncReplicationResourceResource', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_stop_group_async_replication_unary_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_stop_group_async_replication') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_stop_group_async_replication') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.StopGroupAsyncReplicationRegionDiskRequest.pb(compute.StopGroupAsyncReplicationRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.StopGroupAsyncReplicationRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.stop_group_async_replication_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_stop_group_async_replication_unary_rest_bad_request(transport: str='rest', request_type=compute.StopGroupAsyncReplicationRegionDiskRequest):
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.stop_group_async_replication_unary(request)

def test_stop_group_async_replication_unary_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2'}
        mock_args = dict(project='project_value', region='region_value', disks_stop_group_async_replication_resource_resource=compute.DisksStopGroupAsyncReplicationResource(resource_policy='resource_policy_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.stop_group_async_replication_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/stopGroupAsyncReplication' % client.transport._host, args[1])

def test_stop_group_async_replication_unary_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.stop_group_async_replication_unary(compute.StopGroupAsyncReplicationRegionDiskRequest(), project='project_value', region='region_value', disks_stop_group_async_replication_resource_resource=compute.DisksStopGroupAsyncReplicationResource(resource_policy='resource_policy_value'))

def test_stop_group_async_replication_unary_rest_error():
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.TestIamPermissionsRegionDiskRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'resource': 'sample3'}
    request_init['test_permissions_request_resource'] = {'permissions': ['permissions_value1', 'permissions_value2']}
    test_field = compute.TestIamPermissionsRegionDiskRequest.meta.fields['test_permissions_request_resource']

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
    for (field, value) in request_init['test_permissions_request_resource'].items():
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
                for i in range(0, len(request_init['test_permissions_request_resource'][field])):
                    del request_init['test_permissions_request_resource'][field][i][subfield]
            else:
                del request_init['test_permissions_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.TestPermissionsResponse(permissions=['permissions_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.TestPermissionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.test_iam_permissions(request)
    assert isinstance(response, compute.TestPermissionsResponse)
    assert response.permissions == ['permissions_value']

def test_test_iam_permissions_rest_required_fields(request_type=compute.TestIamPermissionsRegionDiskRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['region'] = ''
    request_init['resource'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    jsonified_request['resource'] = 'resource_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.TestPermissionsResponse()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode_result['body'] = pb_request
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.TestPermissionsResponse.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.test_iam_permissions(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_test_iam_permissions_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.test_iam_permissions._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('project', 'region', 'resource', 'testPermissionsRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_test_iam_permissions_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_test_iam_permissions') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_test_iam_permissions') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.TestIamPermissionsRegionDiskRequest.pb(compute.TestIamPermissionsRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.TestPermissionsResponse.to_json(compute.TestPermissionsResponse())
        request = compute.TestIamPermissionsRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.TestPermissionsResponse()
        client.test_iam_permissions(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_test_iam_permissions_rest_bad_request(transport: str='rest', request_type=compute.TestIamPermissionsRegionDiskRequest):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'resource': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

def test_test_iam_permissions_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.TestPermissionsResponse()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'resource': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', resource='resource_value', test_permissions_request_resource=compute.TestPermissionsRequest(permissions=['permissions_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.TestPermissionsResponse.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.test_iam_permissions(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{resource}/testIamPermissions' % client.transport._host, args[1])

def test_test_iam_permissions_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.test_iam_permissions(compute.TestIamPermissionsRegionDiskRequest(), project='project_value', region='region_value', resource='resource_value', test_permissions_request_resource=compute.TestPermissionsRequest(permissions=['permissions_value']))

def test_test_iam_permissions_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.UpdateRegionDiskRequest, dict])
def test_update_rest(request_type):
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request_init['disk_resource'] = {'architecture': 'architecture_value', 'async_primary_disk': {'consistency_group_policy': 'consistency_group_policy_value', 'consistency_group_policy_id': 'consistency_group_policy_id_value', 'disk': 'disk_value', 'disk_id': 'disk_id_value'}, 'async_secondary_disks': {}, 'creation_timestamp': 'creation_timestamp_value', 'description': 'description_value', 'disk_encryption_key': {'kms_key_name': 'kms_key_name_value', 'kms_key_service_account': 'kms_key_service_account_value', 'raw_key': 'raw_key_value', 'rsa_encrypted_key': 'rsa_encrypted_key_value', 'sha256': 'sha256_value'}, 'guest_os_features': [{'type_': 'type__value'}], 'id': 205, 'kind': 'kind_value', 'label_fingerprint': 'label_fingerprint_value', 'labels': {}, 'last_attach_timestamp': 'last_attach_timestamp_value', 'last_detach_timestamp': 'last_detach_timestamp_value', 'license_codes': [1361, 1362], 'licenses': ['licenses_value1', 'licenses_value2'], 'location_hint': 'location_hint_value', 'name': 'name_value', 'options': 'options_value', 'params': {'resource_manager_tags': {}}, 'physical_block_size_bytes': 2663, 'provisioned_iops': 1740, 'provisioned_throughput': 2411, 'region': 'region_value', 'replica_zones': ['replica_zones_value1', 'replica_zones_value2'], 'resource_policies': ['resource_policies_value1', 'resource_policies_value2'], 'resource_status': {'async_primary_disk': {'state': 'state_value'}, 'async_secondary_disks': {}}, 'satisfies_pzs': True, 'self_link': 'self_link_value', 'size_gb': 739, 'source_consistency_group_policy': 'source_consistency_group_policy_value', 'source_consistency_group_policy_id': 'source_consistency_group_policy_id_value', 'source_disk': 'source_disk_value', 'source_disk_id': 'source_disk_id_value', 'source_image': 'source_image_value', 'source_image_encryption_key': {}, 'source_image_id': 'source_image_id_value', 'source_snapshot': 'source_snapshot_value', 'source_snapshot_encryption_key': {}, 'source_snapshot_id': 'source_snapshot_id_value', 'source_storage_object': 'source_storage_object_value', 'status': 'status_value', 'type_': 'type__value', 'users': ['users_value1', 'users_value2'], 'zone': 'zone_value'}
    test_field = compute.UpdateRegionDiskRequest.meta.fields['disk_resource']

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
    for (field, value) in request_init['disk_resource'].items():
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
                for i in range(0, len(request_init['disk_resource'][field])):
                    del request_init['disk_resource'][field][i][subfield]
            else:
                del request_init['disk_resource'][field][subfield]
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

def test_update_rest_required_fields(request_type=compute.UpdateRegionDiskRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['disk'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['disk'] = 'disk_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('paths', 'request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    assert 'disk' in jsonified_request
    assert jsonified_request['disk'] == 'disk_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_rest_unset_required_fields():
    if False:
        return 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update._get_unset_required_fields({})
    assert set(unset_fields) == set(('paths', 'requestId', 'updateMask')) & set(('disk', 'diskResource', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_update') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_update') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.UpdateRegionDiskRequest.pb(compute.UpdateRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.UpdateRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.update(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_rest_bad_request(transport: str='rest', request_type=compute.UpdateRegionDiskRequest):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update(request)

def test_update_rest_flattened():
    if False:
        while True:
            i = 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', disk='disk_value', disk_resource=compute.Disk(architecture='architecture_value'))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{disk}' % client.transport._host, args[1])

def test_update_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update(compute.UpdateRegionDiskRequest(), project='project_value', region='region_value', disk='disk_value', disk_resource=compute.Disk(architecture='architecture_value'))

def test_update_rest_error():
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.UpdateRegionDiskRequest, dict])
def test_update_unary_rest(request_type):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request_init['disk_resource'] = {'architecture': 'architecture_value', 'async_primary_disk': {'consistency_group_policy': 'consistency_group_policy_value', 'consistency_group_policy_id': 'consistency_group_policy_id_value', 'disk': 'disk_value', 'disk_id': 'disk_id_value'}, 'async_secondary_disks': {}, 'creation_timestamp': 'creation_timestamp_value', 'description': 'description_value', 'disk_encryption_key': {'kms_key_name': 'kms_key_name_value', 'kms_key_service_account': 'kms_key_service_account_value', 'raw_key': 'raw_key_value', 'rsa_encrypted_key': 'rsa_encrypted_key_value', 'sha256': 'sha256_value'}, 'guest_os_features': [{'type_': 'type__value'}], 'id': 205, 'kind': 'kind_value', 'label_fingerprint': 'label_fingerprint_value', 'labels': {}, 'last_attach_timestamp': 'last_attach_timestamp_value', 'last_detach_timestamp': 'last_detach_timestamp_value', 'license_codes': [1361, 1362], 'licenses': ['licenses_value1', 'licenses_value2'], 'location_hint': 'location_hint_value', 'name': 'name_value', 'options': 'options_value', 'params': {'resource_manager_tags': {}}, 'physical_block_size_bytes': 2663, 'provisioned_iops': 1740, 'provisioned_throughput': 2411, 'region': 'region_value', 'replica_zones': ['replica_zones_value1', 'replica_zones_value2'], 'resource_policies': ['resource_policies_value1', 'resource_policies_value2'], 'resource_status': {'async_primary_disk': {'state': 'state_value'}, 'async_secondary_disks': {}}, 'satisfies_pzs': True, 'self_link': 'self_link_value', 'size_gb': 739, 'source_consistency_group_policy': 'source_consistency_group_policy_value', 'source_consistency_group_policy_id': 'source_consistency_group_policy_id_value', 'source_disk': 'source_disk_value', 'source_disk_id': 'source_disk_id_value', 'source_image': 'source_image_value', 'source_image_encryption_key': {}, 'source_image_id': 'source_image_id_value', 'source_snapshot': 'source_snapshot_value', 'source_snapshot_encryption_key': {}, 'source_snapshot_id': 'source_snapshot_id_value', 'source_storage_object': 'source_storage_object_value', 'status': 'status_value', 'type_': 'type__value', 'users': ['users_value1', 'users_value2'], 'zone': 'zone_value'}
    test_field = compute.UpdateRegionDiskRequest.meta.fields['disk_resource']

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
    for (field, value) in request_init['disk_resource'].items():
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
                for i in range(0, len(request_init['disk_resource'][field])):
                    del request_init['disk_resource'][field][i][subfield]
            else:
                del request_init['disk_resource'][field][subfield]
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

def test_update_unary_rest_required_fields(request_type=compute.UpdateRegionDiskRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.RegionDisksRestTransport
    request_init = {}
    request_init['disk'] = ''
    request_init['project'] = ''
    request_init['region'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['disk'] = 'disk_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['region'] = 'region_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).update._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('paths', 'request_id', 'update_mask'))
    jsonified_request.update(unset_fields)
    assert 'disk' in jsonified_request
    assert jsonified_request['disk'] == 'disk_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'region' in jsonified_request
    assert jsonified_request['region'] == 'region_value'
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.update_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_update_unary_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.update._get_unset_required_fields({})
    assert set(unset_fields) == set(('paths', 'requestId', 'updateMask')) & set(('disk', 'diskResource', 'project', 'region'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_update_unary_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.RegionDisksRestInterceptor())
    client = RegionDisksClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.RegionDisksRestInterceptor, 'post_update') as post, mock.patch.object(transports.RegionDisksRestInterceptor, 'pre_update') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.UpdateRegionDiskRequest.pb(compute.UpdateRegionDiskRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.UpdateRegionDiskRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.update_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_update_unary_rest_bad_request(transport: str='rest', request_type=compute.UpdateRegionDiskRequest):
    if False:
        i = 10
        return i + 15
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.update_unary(request)

def test_update_unary_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'region': 'sample2', 'disk': 'sample3'}
        mock_args = dict(project='project_value', region='region_value', disk='disk_value', disk_resource=compute.Disk(architecture='architecture_value'))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/regions/{region}/disks/{disk}' % client.transport._host, args[1])

def test_update_unary_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.update_unary(compute.UpdateRegionDiskRequest(), project='project_value', region='region_value', disk='disk_value', disk_resource=compute.Disk(architecture='architecture_value'))

def test_update_unary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = RegionDisksClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = RegionDisksClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = RegionDisksClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = RegionDisksClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        while True:
            i = 10
    transport = transports.RegionDisksRestTransport(credentials=ga_credentials.AnonymousCredentials())
    client = RegionDisksClient(transport=transport)
    assert client.transport is transport

@pytest.mark.parametrize('transport_class', [transports.RegionDisksRestTransport])
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
    transport = RegionDisksClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_region_disks_base_transport_error():
    if False:
        return 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.RegionDisksTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_region_disks_base_transport():
    if False:
        print('Hello World!')
    with mock.patch('google.cloud.compute_v1.services.region_disks.transports.RegionDisksTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.RegionDisksTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('add_resource_policies', 'bulk_insert', 'create_snapshot', 'delete', 'get', 'get_iam_policy', 'insert', 'list', 'remove_resource_policies', 'resize', 'set_iam_policy', 'set_labels', 'start_async_replication', 'stop_async_replication', 'stop_group_async_replication', 'test_iam_permissions', 'update')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_region_disks_base_transport_with_credentials_file():
    if False:
        return 10
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.compute_v1.services.region_disks.transports.RegionDisksTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.RegionDisksTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id='octopus')

def test_region_disks_base_transport_with_adc():
    if False:
        for i in range(10):
            print('nop')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.compute_v1.services.region_disks.transports.RegionDisksTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.RegionDisksTransport()
        adc.assert_called_once()

def test_region_disks_auth_adc():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        RegionDisksClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id=None)

def test_region_disks_http_transport_client_cert_source_for_mtls():
    if False:
        i = 10
        return i + 15
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.RegionDisksRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['rest'])
def test_region_disks_host_no_port(transport_name):
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='compute.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('compute.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_region_disks_host_with_port(transport_name):
    if False:
        return 10
    client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='compute.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('compute.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_region_disks_client_transport_session_collision(transport_name):
    if False:
        while True:
            i = 10
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = RegionDisksClient(credentials=creds1, transport=transport_name)
    client2 = RegionDisksClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.add_resource_policies._session
    session2 = client2.transport.add_resource_policies._session
    assert session1 != session2
    session1 = client1.transport.bulk_insert._session
    session2 = client2.transport.bulk_insert._session
    assert session1 != session2
    session1 = client1.transport.create_snapshot._session
    session2 = client2.transport.create_snapshot._session
    assert session1 != session2
    session1 = client1.transport.delete._session
    session2 = client2.transport.delete._session
    assert session1 != session2
    session1 = client1.transport.get._session
    session2 = client2.transport.get._session
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
    session1 = client1.transport.remove_resource_policies._session
    session2 = client2.transport.remove_resource_policies._session
    assert session1 != session2
    session1 = client1.transport.resize._session
    session2 = client2.transport.resize._session
    assert session1 != session2
    session1 = client1.transport.set_iam_policy._session
    session2 = client2.transport.set_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.set_labels._session
    session2 = client2.transport.set_labels._session
    assert session1 != session2
    session1 = client1.transport.start_async_replication._session
    session2 = client2.transport.start_async_replication._session
    assert session1 != session2
    session1 = client1.transport.stop_async_replication._session
    session2 = client2.transport.stop_async_replication._session
    assert session1 != session2
    session1 = client1.transport.stop_group_async_replication._session
    session2 = client2.transport.stop_group_async_replication._session
    assert session1 != session2
    session1 = client1.transport.test_iam_permissions._session
    session2 = client2.transport.test_iam_permissions._session
    assert session1 != session2
    session1 = client1.transport.update._session
    session2 = client2.transport.update._session
    assert session1 != session2

def test_common_billing_account_path():
    if False:
        print('Hello World!')
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = RegionDisksClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'billing_account': 'clam'}
    path = RegionDisksClient.common_billing_account_path(**expected)
    actual = RegionDisksClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        for i in range(10):
            print('nop')
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = RegionDisksClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        i = 10
        return i + 15
    expected = {'folder': 'octopus'}
    path = RegionDisksClient.common_folder_path(**expected)
    actual = RegionDisksClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        i = 10
        return i + 15
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = RegionDisksClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        while True:
            i = 10
    expected = {'organization': 'nudibranch'}
    path = RegionDisksClient.common_organization_path(**expected)
    actual = RegionDisksClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        return 10
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = RegionDisksClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        i = 10
        return i + 15
    expected = {'project': 'mussel'}
    path = RegionDisksClient.common_project_path(**expected)
    actual = RegionDisksClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        for i in range(10):
            print('nop')
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = RegionDisksClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        return 10
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = RegionDisksClient.common_location_path(**expected)
    actual = RegionDisksClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        print('Hello World!')
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.RegionDisksTransport, '_prep_wrapped_messages') as prep:
        client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.RegionDisksTransport, '_prep_wrapped_messages') as prep:
        transport_class = RegionDisksClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

def test_transport_close():
    if False:
        i = 10
        return i + 15
    transports = {'rest': '_session'}
    for (transport, close_name) in transports.items():
        client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(getattr(client.transport, close_name)), 'close') as close:
            with client:
                close.assert_not_called()
            close.assert_called_once()

def test_client_ctx():
    if False:
        return 10
    transports = ['rest']
    for transport in transports:
        client = RegionDisksClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(RegionDisksClient, transports.RegionDisksRestTransport)])
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