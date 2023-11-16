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
from google.cloud.compute_v1.services.node_groups import NodeGroupsClient, pagers, transports
from google.cloud.compute_v1.types import compute

def client_cert_source_callback():
    if False:
        while True:
            i = 10
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
    assert NodeGroupsClient._get_default_mtls_endpoint(None) is None
    assert NodeGroupsClient._get_default_mtls_endpoint(api_endpoint) == api_mtls_endpoint
    assert NodeGroupsClient._get_default_mtls_endpoint(api_mtls_endpoint) == api_mtls_endpoint
    assert NodeGroupsClient._get_default_mtls_endpoint(sandbox_endpoint) == sandbox_mtls_endpoint
    assert NodeGroupsClient._get_default_mtls_endpoint(sandbox_mtls_endpoint) == sandbox_mtls_endpoint
    assert NodeGroupsClient._get_default_mtls_endpoint(non_googleapi) == non_googleapi

@pytest.mark.parametrize('client_class,transport_name', [(NodeGroupsClient, 'rest')])
def test_node_groups_client_from_service_account_info(client_class, transport_name):
    if False:
        print('Hello World!')
    creds = ga_credentials.AnonymousCredentials()
    with mock.patch.object(service_account.Credentials, 'from_service_account_info') as factory:
        factory.return_value = creds
        info = {'valid': True}
        client = client_class.from_service_account_info(info, transport=transport_name)
        assert client.transport._credentials == creds
        assert isinstance(client, client_class)
        assert client.transport._host == ('compute.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com')

@pytest.mark.parametrize('transport_class,transport_name', [(transports.NodeGroupsRestTransport, 'rest')])
def test_node_groups_client_service_account_always_use_jwt(transport_class, transport_name):
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

@pytest.mark.parametrize('client_class,transport_name', [(NodeGroupsClient, 'rest')])
def test_node_groups_client_from_service_account_file(client_class, transport_name):
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
        assert client.transport._host == ('compute.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com')

def test_node_groups_client_get_transport_class():
    if False:
        for i in range(10):
            print('nop')
    transport = NodeGroupsClient.get_transport_class()
    available_transports = [transports.NodeGroupsRestTransport]
    assert transport in available_transports
    transport = NodeGroupsClient.get_transport_class('rest')
    assert transport == transports.NodeGroupsRestTransport

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(NodeGroupsClient, transports.NodeGroupsRestTransport, 'rest')])
@mock.patch.object(NodeGroupsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NodeGroupsClient))
def test_node_groups_client_client_options(client_class, transport_class, transport_name):
    if False:
        return 10
    with mock.patch.object(NodeGroupsClient, 'get_transport_class') as gtc:
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials())
        client = client_class(transport=transport)
        gtc.assert_not_called()
    with mock.patch.object(NodeGroupsClient, 'get_transport_class') as gtc:
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

@pytest.mark.parametrize('client_class,transport_class,transport_name,use_client_cert_env', [(NodeGroupsClient, transports.NodeGroupsRestTransport, 'rest', 'true'), (NodeGroupsClient, transports.NodeGroupsRestTransport, 'rest', 'false')])
@mock.patch.object(NodeGroupsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NodeGroupsClient))
@mock.patch.dict(os.environ, {'GOOGLE_API_USE_MTLS_ENDPOINT': 'auto'})
def test_node_groups_client_mtls_env_auto(client_class, transport_class, transport_name, use_client_cert_env):
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

@pytest.mark.parametrize('client_class', [NodeGroupsClient])
@mock.patch.object(NodeGroupsClient, 'DEFAULT_ENDPOINT', modify_default_endpoint(NodeGroupsClient))
def test_node_groups_client_get_mtls_endpoint_and_cert_source(client_class):
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

@pytest.mark.parametrize('client_class,transport_class,transport_name', [(NodeGroupsClient, transports.NodeGroupsRestTransport, 'rest')])
def test_node_groups_client_client_options_scopes(client_class, transport_class, transport_name):
    if False:
        while True:
            i = 10
    options = client_options.ClientOptions(scopes=['1', '2'])
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file=None, host=client.DEFAULT_ENDPOINT, scopes=['1', '2'], client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('client_class,transport_class,transport_name,grpc_helpers', [(NodeGroupsClient, transports.NodeGroupsRestTransport, 'rest', None)])
def test_node_groups_client_client_options_credentials_file(client_class, transport_class, transport_name, grpc_helpers):
    if False:
        print('Hello World!')
    options = client_options.ClientOptions(credentials_file='credentials.json')
    with mock.patch.object(transport_class, '__init__') as patched:
        patched.return_value = None
        client = client_class(client_options=options, transport=transport_name)
        patched.assert_called_once_with(credentials=None, credentials_file='credentials.json', host=client.DEFAULT_ENDPOINT, scopes=None, client_cert_source_for_mtls=None, quota_project_id=None, client_info=transports.base.DEFAULT_CLIENT_INFO, always_use_jwt_access=True, api_audience=None)

@pytest.mark.parametrize('request_type', [compute.AddNodesNodeGroupRequest, dict])
def test_add_nodes_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request_init['node_groups_add_nodes_request_resource'] = {'additional_node_count': 2214}
    test_field = compute.AddNodesNodeGroupRequest.meta.fields['node_groups_add_nodes_request_resource']

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
    for (field, value) in request_init['node_groups_add_nodes_request_resource'].items():
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
                for i in range(0, len(request_init['node_groups_add_nodes_request_resource'][field])):
                    del request_init['node_groups_add_nodes_request_resource'][field][i][subfield]
            else:
                del request_init['node_groups_add_nodes_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.add_nodes(request)
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

def test_add_nodes_rest_required_fields(request_type=compute.AddNodesNodeGroupRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['node_group'] = ''
    request_init['project'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_nodes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['nodeGroup'] = 'node_group_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_nodes._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'nodeGroup' in jsonified_request
    assert jsonified_request['nodeGroup'] == 'node_group_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.add_nodes(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_add_nodes_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.add_nodes._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('nodeGroup', 'nodeGroupsAddNodesRequestResource', 'project', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_add_nodes_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_add_nodes') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_add_nodes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.AddNodesNodeGroupRequest.pb(compute.AddNodesNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.AddNodesNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.add_nodes(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_add_nodes_rest_bad_request(transport: str='rest', request_type=compute.AddNodesNodeGroupRequest):
    if False:
        print('Hello World!')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.add_nodes(request)

def test_add_nodes_rest_flattened():
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
        mock_args = dict(project='project_value', zone='zone_value', node_group='node_group_value', node_groups_add_nodes_request_resource=compute.NodeGroupsAddNodesRequest(additional_node_count=2214))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.add_nodes(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups/{node_group}/addNodes' % client.transport._host, args[1])

def test_add_nodes_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.add_nodes(compute.AddNodesNodeGroupRequest(), project='project_value', zone='zone_value', node_group='node_group_value', node_groups_add_nodes_request_resource=compute.NodeGroupsAddNodesRequest(additional_node_count=2214))

def test_add_nodes_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.AddNodesNodeGroupRequest, dict])
def test_add_nodes_unary_rest(request_type):
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request_init['node_groups_add_nodes_request_resource'] = {'additional_node_count': 2214}
    test_field = compute.AddNodesNodeGroupRequest.meta.fields['node_groups_add_nodes_request_resource']

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
    for (field, value) in request_init['node_groups_add_nodes_request_resource'].items():
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
                for i in range(0, len(request_init['node_groups_add_nodes_request_resource'][field])):
                    del request_init['node_groups_add_nodes_request_resource'][field][i][subfield]
            else:
                del request_init['node_groups_add_nodes_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.add_nodes_unary(request)
    assert isinstance(response, compute.Operation)

def test_add_nodes_unary_rest_required_fields(request_type=compute.AddNodesNodeGroupRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['node_group'] = ''
    request_init['project'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_nodes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['nodeGroup'] = 'node_group_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).add_nodes._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'nodeGroup' in jsonified_request
    assert jsonified_request['nodeGroup'] == 'node_group_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.add_nodes_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_add_nodes_unary_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.add_nodes._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('nodeGroup', 'nodeGroupsAddNodesRequestResource', 'project', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_add_nodes_unary_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_add_nodes') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_add_nodes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.AddNodesNodeGroupRequest.pb(compute.AddNodesNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.AddNodesNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.add_nodes_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_add_nodes_unary_rest_bad_request(transport: str='rest', request_type=compute.AddNodesNodeGroupRequest):
    if False:
        return 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.add_nodes_unary(request)

def test_add_nodes_unary_rest_flattened():
    if False:
        return 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
        mock_args = dict(project='project_value', zone='zone_value', node_group='node_group_value', node_groups_add_nodes_request_resource=compute.NodeGroupsAddNodesRequest(additional_node_count=2214))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.add_nodes_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups/{node_group}/addNodes' % client.transport._host, args[1])

def test_add_nodes_unary_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.add_nodes_unary(compute.AddNodesNodeGroupRequest(), project='project_value', zone='zone_value', node_group='node_group_value', node_groups_add_nodes_request_resource=compute.NodeGroupsAddNodesRequest(additional_node_count=2214))

def test_add_nodes_unary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.AggregatedListNodeGroupsRequest, dict])
def test_aggregated_list_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.NodeGroupAggregatedList(id='id_value', kind='kind_value', next_page_token='next_page_token_value', self_link='self_link_value', unreachables=['unreachables_value'])
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.NodeGroupAggregatedList.pb(return_value)
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

def test_aggregated_list_rest_required_fields(request_type=compute.AggregatedListNodeGroupsRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NodeGroupsRestTransport
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
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.NodeGroupAggregatedList()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.NodeGroupAggregatedList.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.aggregated_list(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_aggregated_list_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.aggregated_list._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess')) & set(('project',))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_aggregated_list_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_aggregated_list') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_aggregated_list') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.AggregatedListNodeGroupsRequest.pb(compute.AggregatedListNodeGroupsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.NodeGroupAggregatedList.to_json(compute.NodeGroupAggregatedList())
        request = compute.AggregatedListNodeGroupsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.NodeGroupAggregatedList()
        client.aggregated_list(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_aggregated_list_rest_bad_request(transport: str='rest', request_type=compute.AggregatedListNodeGroupsRequest):
    if False:
        print('Hello World!')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.NodeGroupAggregatedList()
        sample_request = {'project': 'sample1'}
        mock_args = dict(project='project_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.NodeGroupAggregatedList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.aggregated_list(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/aggregated/nodeGroups' % client.transport._host, args[1])

def test_aggregated_list_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.aggregated_list(compute.AggregatedListNodeGroupsRequest(), project='project_value')

def test_aggregated_list_rest_pager(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (compute.NodeGroupAggregatedList(items={'a': compute.NodeGroupsScopedList(), 'b': compute.NodeGroupsScopedList(), 'c': compute.NodeGroupsScopedList()}, next_page_token='abc'), compute.NodeGroupAggregatedList(items={}, next_page_token='def'), compute.NodeGroupAggregatedList(items={'g': compute.NodeGroupsScopedList()}, next_page_token='ghi'), compute.NodeGroupAggregatedList(items={'h': compute.NodeGroupsScopedList(), 'i': compute.NodeGroupsScopedList()}))
        response = response + response
        response = tuple((compute.NodeGroupAggregatedList.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'project': 'sample1'}
        pager = client.aggregated_list(request=sample_request)
        assert isinstance(pager.get('a'), compute.NodeGroupsScopedList)
        assert pager.get('h') is None
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, tuple) for i in results))
        for result in results:
            assert isinstance(result, tuple)
            assert tuple((type(t) for t in result)) == (str, compute.NodeGroupsScopedList)
        assert pager.get('a') is None
        assert isinstance(pager.get('h'), compute.NodeGroupsScopedList)
        pages = list(client.aggregated_list(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [compute.DeleteNodeGroupRequest, dict])
def test_delete_rest(request_type):
    if False:
        return 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
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

def test_delete_rest_required_fields(request_type=compute.DeleteNodeGroupRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['node_group'] = ''
    request_init['project'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['nodeGroup'] = 'node_group_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'nodeGroup' in jsonified_request
    assert jsonified_request['nodeGroup'] == 'node_group_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('nodeGroup', 'project', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_delete') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_delete') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteNodeGroupRequest.pb(compute.DeleteNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_rest_bad_request(transport: str='rest', request_type=compute.DeleteNodeGroupRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete(request)

def test_delete_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
        mock_args = dict(project='project_value', zone='zone_value', node_group='node_group_value')
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
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups/{node_group}' % client.transport._host, args[1])

def test_delete_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete(compute.DeleteNodeGroupRequest(), project='project_value', zone='zone_value', node_group='node_group_value')

def test_delete_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.DeleteNodeGroupRequest, dict])
def test_delete_unary_rest(request_type):
    if False:
        print('Hello World!')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
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

def test_delete_unary_rest_required_fields(request_type=compute.DeleteNodeGroupRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['node_group'] = ''
    request_init['project'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['nodeGroup'] = 'node_group_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'nodeGroup' in jsonified_request
    assert jsonified_request['nodeGroup'] == 'node_group_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('nodeGroup', 'project', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_unary_rest_interceptors(null_interceptor):
    if False:
        i = 10
        return i + 15
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_delete') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_delete') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteNodeGroupRequest.pb(compute.DeleteNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_unary_rest_bad_request(transport: str='rest', request_type=compute.DeleteNodeGroupRequest):
    if False:
        return 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_unary(request)

def test_delete_unary_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
        mock_args = dict(project='project_value', zone='zone_value', node_group='node_group_value')
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
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups/{node_group}' % client.transport._host, args[1])

def test_delete_unary_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_unary(compute.DeleteNodeGroupRequest(), project='project_value', zone='zone_value', node_group='node_group_value')

def test_delete_unary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.DeleteNodesNodeGroupRequest, dict])
def test_delete_nodes_rest(request_type):
    if False:
        i = 10
        return i + 15
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request_init['node_groups_delete_nodes_request_resource'] = {'nodes': ['nodes_value1', 'nodes_value2']}
    test_field = compute.DeleteNodesNodeGroupRequest.meta.fields['node_groups_delete_nodes_request_resource']

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
    for (field, value) in request_init['node_groups_delete_nodes_request_resource'].items():
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
                for i in range(0, len(request_init['node_groups_delete_nodes_request_resource'][field])):
                    del request_init['node_groups_delete_nodes_request_resource'][field][i][subfield]
            else:
                del request_init['node_groups_delete_nodes_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_nodes(request)
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

def test_delete_nodes_rest_required_fields(request_type=compute.DeleteNodesNodeGroupRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['node_group'] = ''
    request_init['project'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_nodes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['nodeGroup'] = 'node_group_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_nodes._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'nodeGroup' in jsonified_request
    assert jsonified_request['nodeGroup'] == 'node_group_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_nodes(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_nodes_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_nodes._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('nodeGroup', 'nodeGroupsDeleteNodesRequestResource', 'project', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_nodes_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_delete_nodes') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_delete_nodes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteNodesNodeGroupRequest.pb(compute.DeleteNodesNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteNodesNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete_nodes(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_nodes_rest_bad_request(transport: str='rest', request_type=compute.DeleteNodesNodeGroupRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_nodes(request)

def test_delete_nodes_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
        mock_args = dict(project='project_value', zone='zone_value', node_group='node_group_value', node_groups_delete_nodes_request_resource=compute.NodeGroupsDeleteNodesRequest(nodes=['nodes_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_nodes(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups/{node_group}/deleteNodes' % client.transport._host, args[1])

def test_delete_nodes_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_nodes(compute.DeleteNodesNodeGroupRequest(), project='project_value', zone='zone_value', node_group='node_group_value', node_groups_delete_nodes_request_resource=compute.NodeGroupsDeleteNodesRequest(nodes=['nodes_value']))

def test_delete_nodes_rest_error():
    if False:
        return 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.DeleteNodesNodeGroupRequest, dict])
def test_delete_nodes_unary_rest(request_type):
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request_init['node_groups_delete_nodes_request_resource'] = {'nodes': ['nodes_value1', 'nodes_value2']}
    test_field = compute.DeleteNodesNodeGroupRequest.meta.fields['node_groups_delete_nodes_request_resource']

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
    for (field, value) in request_init['node_groups_delete_nodes_request_resource'].items():
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
                for i in range(0, len(request_init['node_groups_delete_nodes_request_resource'][field])):
                    del request_init['node_groups_delete_nodes_request_resource'][field][i][subfield]
            else:
                del request_init['node_groups_delete_nodes_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.delete_nodes_unary(request)
    assert isinstance(response, compute.Operation)

def test_delete_nodes_unary_rest_required_fields(request_type=compute.DeleteNodesNodeGroupRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['node_group'] = ''
    request_init['project'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_nodes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['nodeGroup'] = 'node_group_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).delete_nodes._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'nodeGroup' in jsonified_request
    assert jsonified_request['nodeGroup'] == 'node_group_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.delete_nodes_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_delete_nodes_unary_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.delete_nodes._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('nodeGroup', 'nodeGroupsDeleteNodesRequestResource', 'project', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_delete_nodes_unary_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_delete_nodes') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_delete_nodes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.DeleteNodesNodeGroupRequest.pb(compute.DeleteNodesNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.DeleteNodesNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.delete_nodes_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_delete_nodes_unary_rest_bad_request(transport: str='rest', request_type=compute.DeleteNodesNodeGroupRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.delete_nodes_unary(request)

def test_delete_nodes_unary_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
        mock_args = dict(project='project_value', zone='zone_value', node_group='node_group_value', node_groups_delete_nodes_request_resource=compute.NodeGroupsDeleteNodesRequest(nodes=['nodes_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.delete_nodes_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups/{node_group}/deleteNodes' % client.transport._host, args[1])

def test_delete_nodes_unary_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.delete_nodes_unary(compute.DeleteNodesNodeGroupRequest(), project='project_value', zone='zone_value', node_group='node_group_value', node_groups_delete_nodes_request_resource=compute.NodeGroupsDeleteNodesRequest(nodes=['nodes_value']))

def test_delete_nodes_unary_rest_error():
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.GetNodeGroupRequest, dict])
def test_get_rest(request_type):
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.NodeGroup(creation_timestamp='creation_timestamp_value', description='description_value', fingerprint='fingerprint_value', id=205, kind='kind_value', location_hint='location_hint_value', maintenance_policy='maintenance_policy_value', name='name_value', node_template='node_template_value', self_link='self_link_value', size=443, status='status_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.NodeGroup.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.get(request)
    assert isinstance(response, compute.NodeGroup)
    assert response.creation_timestamp == 'creation_timestamp_value'
    assert response.description == 'description_value'
    assert response.fingerprint == 'fingerprint_value'
    assert response.id == 205
    assert response.kind == 'kind_value'
    assert response.location_hint == 'location_hint_value'
    assert response.maintenance_policy == 'maintenance_policy_value'
    assert response.name == 'name_value'
    assert response.node_template == 'node_template_value'
    assert response.self_link == 'self_link_value'
    assert response.size == 443
    assert response.status == 'status_value'
    assert response.zone == 'zone_value'

def test_get_rest_required_fields(request_type=compute.GetNodeGroupRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['node_group'] = ''
    request_init['project'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['nodeGroup'] = 'node_group_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'nodeGroup' in jsonified_request
    assert jsonified_request['nodeGroup'] == 'node_group_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.NodeGroup()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.NodeGroup.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.get(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_get_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('nodeGroup', 'project', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_get') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_get') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.GetNodeGroupRequest.pb(compute.GetNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.NodeGroup.to_json(compute.NodeGroup())
        request = compute.GetNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.NodeGroup()
        client.get(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_rest_bad_request(transport: str='rest', request_type=compute.GetNodeGroupRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
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
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.NodeGroup()
        sample_request = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
        mock_args = dict(project='project_value', zone='zone_value', node_group='node_group_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.NodeGroup.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.get(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups/{node_group}' % client.transport._host, args[1])

def test_get_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get(compute.GetNodeGroupRequest(), project='project_value', zone='zone_value', node_group='node_group_value')

def test_get_rest_error():
    if False:
        return 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.GetIamPolicyNodeGroupRequest, dict])
def test_get_iam_policy_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2', 'resource': 'sample3'}
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

def test_get_iam_policy_rest_required_fields(request_type=compute.GetIamPolicyNodeGroupRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['resource'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['resource'] = 'resource_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).get_iam_policy._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('options_requested_policy_version',))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.get_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(('optionsRequestedPolicyVersion',)) & set(('project', 'resource', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_get_iam_policy_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_get_iam_policy') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_get_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.GetIamPolicyNodeGroupRequest.pb(compute.GetIamPolicyNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Policy.to_json(compute.Policy())
        request = compute.GetIamPolicyNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Policy()
        client.get_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_get_iam_policy_rest_bad_request(transport: str='rest', request_type=compute.GetIamPolicyNodeGroupRequest):
    if False:
        i = 10
        return i + 15
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2', 'resource': 'sample3'}
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
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Policy()
        sample_request = {'project': 'sample1', 'zone': 'sample2', 'resource': 'sample3'}
        mock_args = dict(project='project_value', zone='zone_value', resource='resource_value')
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
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups/{resource}/getIamPolicy' % client.transport._host, args[1])

def test_get_iam_policy_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.get_iam_policy(compute.GetIamPolicyNodeGroupRequest(), project='project_value', zone='zone_value', resource='resource_value')

def test_get_iam_policy_rest_error():
    if False:
        return 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.InsertNodeGroupRequest, dict])
def test_insert_rest(request_type):
    if False:
        print('Hello World!')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2'}
    request_init['node_group_resource'] = {'autoscaling_policy': {'max_nodes': 958, 'min_nodes': 956, 'mode': 'mode_value'}, 'creation_timestamp': 'creation_timestamp_value', 'description': 'description_value', 'fingerprint': 'fingerprint_value', 'id': 205, 'kind': 'kind_value', 'location_hint': 'location_hint_value', 'maintenance_policy': 'maintenance_policy_value', 'maintenance_window': {'maintenance_duration': {'nanos': 543, 'seconds': 751}, 'start_time': 'start_time_value'}, 'name': 'name_value', 'node_template': 'node_template_value', 'self_link': 'self_link_value', 'share_settings': {'project_map': {}, 'share_type': 'share_type_value'}, 'size': 443, 'status': 'status_value', 'zone': 'zone_value'}
    test_field = compute.InsertNodeGroupRequest.meta.fields['node_group_resource']

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
    for (field, value) in request_init['node_group_resource'].items():
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
                for i in range(0, len(request_init['node_group_resource'][field])):
                    del request_init['node_group_resource'][field][i][subfield]
            else:
                del request_init['node_group_resource'][field][subfield]
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

def test_insert_rest_required_fields(request_type=compute.InsertNodeGroupRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['initial_node_count'] = 0
    request_init['project'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'initialNodeCount' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).insert._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'initialNodeCount' in jsonified_request
    assert jsonified_request['initialNodeCount'] == request_init['initial_node_count']
    jsonified_request['initialNodeCount'] = 1911
    jsonified_request['project'] = 'project_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).insert._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('initial_node_count', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'initialNodeCount' in jsonified_request
    assert jsonified_request['initialNodeCount'] == 1911
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            expected_params = [('initialNodeCount', str(0))]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_insert_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.insert._get_unset_required_fields({})
    assert set(unset_fields) == set(('initialNodeCount', 'requestId')) & set(('initialNodeCount', 'nodeGroupResource', 'project', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_insert_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_insert') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_insert') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.InsertNodeGroupRequest.pb(compute.InsertNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.InsertNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.insert(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_insert_rest_bad_request(transport: str='rest', request_type=compute.InsertNodeGroupRequest):
    if False:
        i = 10
        return i + 15
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2'}
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
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'zone': 'sample2'}
        mock_args = dict(project='project_value', zone='zone_value', initial_node_count=1911, node_group_resource=compute.NodeGroup(autoscaling_policy=compute.NodeGroupAutoscalingPolicy(max_nodes=958)))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups' % client.transport._host, args[1])

def test_insert_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.insert(compute.InsertNodeGroupRequest(), project='project_value', zone='zone_value', initial_node_count=1911, node_group_resource=compute.NodeGroup(autoscaling_policy=compute.NodeGroupAutoscalingPolicy(max_nodes=958)))

def test_insert_rest_error():
    if False:
        i = 10
        return i + 15
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.InsertNodeGroupRequest, dict])
def test_insert_unary_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2'}
    request_init['node_group_resource'] = {'autoscaling_policy': {'max_nodes': 958, 'min_nodes': 956, 'mode': 'mode_value'}, 'creation_timestamp': 'creation_timestamp_value', 'description': 'description_value', 'fingerprint': 'fingerprint_value', 'id': 205, 'kind': 'kind_value', 'location_hint': 'location_hint_value', 'maintenance_policy': 'maintenance_policy_value', 'maintenance_window': {'maintenance_duration': {'nanos': 543, 'seconds': 751}, 'start_time': 'start_time_value'}, 'name': 'name_value', 'node_template': 'node_template_value', 'self_link': 'self_link_value', 'share_settings': {'project_map': {}, 'share_type': 'share_type_value'}, 'size': 443, 'status': 'status_value', 'zone': 'zone_value'}
    test_field = compute.InsertNodeGroupRequest.meta.fields['node_group_resource']

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
    for (field, value) in request_init['node_group_resource'].items():
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
                for i in range(0, len(request_init['node_group_resource'][field])):
                    del request_init['node_group_resource'][field][i][subfield]
            else:
                del request_init['node_group_resource'][field][subfield]
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

def test_insert_unary_rest_required_fields(request_type=compute.InsertNodeGroupRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['initial_node_count'] = 0
    request_init['project'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    assert 'initialNodeCount' not in jsonified_request
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).insert._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'initialNodeCount' in jsonified_request
    assert jsonified_request['initialNodeCount'] == request_init['initial_node_count']
    jsonified_request['initialNodeCount'] = 1911
    jsonified_request['project'] = 'project_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).insert._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('initial_node_count', 'request_id'))
    jsonified_request.update(unset_fields)
    assert 'initialNodeCount' in jsonified_request
    assert jsonified_request['initialNodeCount'] == 1911
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            expected_params = [('initialNodeCount', str(0))]
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_insert_unary_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.insert._get_unset_required_fields({})
    assert set(unset_fields) == set(('initialNodeCount', 'requestId')) & set(('initialNodeCount', 'nodeGroupResource', 'project', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_insert_unary_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_insert') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_insert') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.InsertNodeGroupRequest.pb(compute.InsertNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.InsertNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.insert_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_insert_unary_rest_bad_request(transport: str='rest', request_type=compute.InsertNodeGroupRequest):
    if False:
        return 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2'}
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
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'zone': 'sample2'}
        mock_args = dict(project='project_value', zone='zone_value', initial_node_count=1911, node_group_resource=compute.NodeGroup(autoscaling_policy=compute.NodeGroupAutoscalingPolicy(max_nodes=958)))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups' % client.transport._host, args[1])

def test_insert_unary_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.insert_unary(compute.InsertNodeGroupRequest(), project='project_value', zone='zone_value', initial_node_count=1911, node_group_resource=compute.NodeGroup(autoscaling_policy=compute.NodeGroupAutoscalingPolicy(max_nodes=958)))

def test_insert_unary_rest_error():
    if False:
        print('Hello World!')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.ListNodeGroupsRequest, dict])
def test_list_rest(request_type):
    if False:
        return 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.NodeGroupList(id='id_value', kind='kind_value', next_page_token='next_page_token_value', self_link='self_link_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.NodeGroupList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list(request)
    assert isinstance(response, pagers.ListPager)
    assert response.id == 'id_value'
    assert response.kind == 'kind_value'
    assert response.next_page_token == 'next_page_token_value'
    assert response.self_link == 'self_link_value'

def test_list_rest_required_fields(request_type=compute.ListNodeGroupsRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'max_results', 'order_by', 'page_token', 'return_partial_success'))
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.NodeGroupList()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'get', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.NodeGroupList.pb(return_value)
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
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess')) & set(('project', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_list') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_list') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.ListNodeGroupsRequest.pb(compute.ListNodeGroupsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.NodeGroupList.to_json(compute.NodeGroupList())
        request = compute.ListNodeGroupsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.NodeGroupList()
        client.list(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_rest_bad_request(transport: str='rest', request_type=compute.ListNodeGroupsRequest):
    if False:
        i = 10
        return i + 15
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2'}
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
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.NodeGroupList()
        sample_request = {'project': 'sample1', 'zone': 'sample2'}
        mock_args = dict(project='project_value', zone='zone_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.NodeGroupList.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups' % client.transport._host, args[1])

def test_list_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list(compute.ListNodeGroupsRequest(), project='project_value', zone='zone_value')

def test_list_rest_pager(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (compute.NodeGroupList(items=[compute.NodeGroup(), compute.NodeGroup(), compute.NodeGroup()], next_page_token='abc'), compute.NodeGroupList(items=[], next_page_token='def'), compute.NodeGroupList(items=[compute.NodeGroup()], next_page_token='ghi'), compute.NodeGroupList(items=[compute.NodeGroup(), compute.NodeGroup()]))
        response = response + response
        response = tuple((compute.NodeGroupList.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'project': 'sample1', 'zone': 'sample2'}
        pager = client.list(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, compute.NodeGroup) for i in results))
        pages = list(client.list(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [compute.ListNodesNodeGroupsRequest, dict])
def test_list_nodes_rest(request_type):
    if False:
        print('Hello World!')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.NodeGroupsListNodes(id='id_value', kind='kind_value', next_page_token='next_page_token_value', self_link='self_link_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.NodeGroupsListNodes.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.list_nodes(request)
    assert isinstance(response, pagers.ListNodesPager)
    assert response.id == 'id_value'
    assert response.kind == 'kind_value'
    assert response.next_page_token == 'next_page_token_value'
    assert response.self_link == 'self_link_value'

def test_list_nodes_rest_required_fields(request_type=compute.ListNodesNodeGroupsRequest):
    if False:
        while True:
            i = 10
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['node_group'] = ''
    request_init['project'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_nodes._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['nodeGroup'] = 'node_group_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).list_nodes._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('filter', 'max_results', 'order_by', 'page_token', 'return_partial_success'))
    jsonified_request.update(unset_fields)
    assert 'nodeGroup' in jsonified_request
    assert jsonified_request['nodeGroup'] == 'node_group_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request = request_type(**request_init)
    return_value = compute.NodeGroupsListNodes()
    with mock.patch.object(Session, 'request') as req:
        with mock.patch.object(path_template, 'transcode') as transcode:
            pb_request = request_type.pb(request)
            transcode_result = {'uri': 'v1/sample_method', 'method': 'post', 'query_params': pb_request}
            transcode.return_value = transcode_result
            response_value = Response()
            response_value.status_code = 200
            return_value = compute.NodeGroupsListNodes.pb(return_value)
            json_return_value = json_format.MessageToJson(return_value)
            response_value._content = json_return_value.encode('UTF-8')
            req.return_value = response_value
            response = client.list_nodes(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_list_nodes_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.list_nodes._get_unset_required_fields({})
    assert set(unset_fields) == set(('filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess')) & set(('nodeGroup', 'project', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_list_nodes_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_list_nodes') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_list_nodes') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.ListNodesNodeGroupsRequest.pb(compute.ListNodesNodeGroupsRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.NodeGroupsListNodes.to_json(compute.NodeGroupsListNodes())
        request = compute.ListNodesNodeGroupsRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.NodeGroupsListNodes()
        client.list_nodes(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_list_nodes_rest_bad_request(transport: str='rest', request_type=compute.ListNodesNodeGroupsRequest):
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.list_nodes(request)

def test_list_nodes_rest_flattened():
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.NodeGroupsListNodes()
        sample_request = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
        mock_args = dict(project='project_value', zone='zone_value', node_group='node_group_value')
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.NodeGroupsListNodes.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.list_nodes(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups/{node_group}/listNodes' % client.transport._host, args[1])

def test_list_nodes_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.list_nodes(compute.ListNodesNodeGroupsRequest(), project='project_value', zone='zone_value', node_group='node_group_value')

def test_list_nodes_rest_pager(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with mock.patch.object(Session, 'request') as req:
        response = (compute.NodeGroupsListNodes(items=[compute.NodeGroupNode(), compute.NodeGroupNode(), compute.NodeGroupNode()], next_page_token='abc'), compute.NodeGroupsListNodes(items=[], next_page_token='def'), compute.NodeGroupsListNodes(items=[compute.NodeGroupNode()], next_page_token='ghi'), compute.NodeGroupsListNodes(items=[compute.NodeGroupNode(), compute.NodeGroupNode()]))
        response = response + response
        response = tuple((compute.NodeGroupsListNodes.to_json(x) for x in response))
        return_values = tuple((Response() for i in response))
        for (return_val, response_val) in zip(return_values, response):
            return_val._content = response_val.encode('UTF-8')
            return_val.status_code = 200
        req.side_effect = return_values
        sample_request = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
        pager = client.list_nodes(request=sample_request)
        results = list(pager)
        assert len(results) == 6
        assert all((isinstance(i, compute.NodeGroupNode) for i in results))
        pages = list(client.list_nodes(request=sample_request).pages)
        for (page_, token) in zip(pages, ['abc', 'def', 'ghi', '']):
            assert page_.raw_page.next_page_token == token

@pytest.mark.parametrize('request_type', [compute.PatchNodeGroupRequest, dict])
def test_patch_rest(request_type):
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request_init['node_group_resource'] = {'autoscaling_policy': {'max_nodes': 958, 'min_nodes': 956, 'mode': 'mode_value'}, 'creation_timestamp': 'creation_timestamp_value', 'description': 'description_value', 'fingerprint': 'fingerprint_value', 'id': 205, 'kind': 'kind_value', 'location_hint': 'location_hint_value', 'maintenance_policy': 'maintenance_policy_value', 'maintenance_window': {'maintenance_duration': {'nanos': 543, 'seconds': 751}, 'start_time': 'start_time_value'}, 'name': 'name_value', 'node_template': 'node_template_value', 'self_link': 'self_link_value', 'share_settings': {'project_map': {}, 'share_type': 'share_type_value'}, 'size': 443, 'status': 'status_value', 'zone': 'zone_value'}
    test_field = compute.PatchNodeGroupRequest.meta.fields['node_group_resource']

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
    for (field, value) in request_init['node_group_resource'].items():
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
                for i in range(0, len(request_init['node_group_resource'][field])):
                    del request_init['node_group_resource'][field][i][subfield]
            else:
                del request_init['node_group_resource'][field][subfield]
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

def test_patch_rest_required_fields(request_type=compute.PatchNodeGroupRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['node_group'] = ''
    request_init['project'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['nodeGroup'] = 'node_group_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'nodeGroup' in jsonified_request
    assert jsonified_request['nodeGroup'] == 'node_group_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        i = 10
        return i + 15
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.patch._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('nodeGroup', 'nodeGroupResource', 'project', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_patch_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_patch') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_patch') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.PatchNodeGroupRequest.pb(compute.PatchNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.PatchNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.patch(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_patch_rest_bad_request(transport: str='rest', request_type=compute.PatchNodeGroupRequest):
    if False:
        print('Hello World!')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.patch(request)

def test_patch_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
        mock_args = dict(project='project_value', zone='zone_value', node_group='node_group_value', node_group_resource=compute.NodeGroup(autoscaling_policy=compute.NodeGroupAutoscalingPolicy(max_nodes=958)))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups/{node_group}' % client.transport._host, args[1])

def test_patch_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.patch(compute.PatchNodeGroupRequest(), project='project_value', zone='zone_value', node_group='node_group_value', node_group_resource=compute.NodeGroup(autoscaling_policy=compute.NodeGroupAutoscalingPolicy(max_nodes=958)))

def test_patch_rest_error():
    if False:
        i = 10
        return i + 15
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.PatchNodeGroupRequest, dict])
def test_patch_unary_rest(request_type):
    if False:
        print('Hello World!')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request_init['node_group_resource'] = {'autoscaling_policy': {'max_nodes': 958, 'min_nodes': 956, 'mode': 'mode_value'}, 'creation_timestamp': 'creation_timestamp_value', 'description': 'description_value', 'fingerprint': 'fingerprint_value', 'id': 205, 'kind': 'kind_value', 'location_hint': 'location_hint_value', 'maintenance_policy': 'maintenance_policy_value', 'maintenance_window': {'maintenance_duration': {'nanos': 543, 'seconds': 751}, 'start_time': 'start_time_value'}, 'name': 'name_value', 'node_template': 'node_template_value', 'self_link': 'self_link_value', 'share_settings': {'project_map': {}, 'share_type': 'share_type_value'}, 'size': 443, 'status': 'status_value', 'zone': 'zone_value'}
    test_field = compute.PatchNodeGroupRequest.meta.fields['node_group_resource']

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
    for (field, value) in request_init['node_group_resource'].items():
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
                for i in range(0, len(request_init['node_group_resource'][field])):
                    del request_init['node_group_resource'][field][i][subfield]
            else:
                del request_init['node_group_resource'][field][subfield]
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

def test_patch_unary_rest_required_fields(request_type=compute.PatchNodeGroupRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['node_group'] = ''
    request_init['project'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['nodeGroup'] = 'node_group_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).patch._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'nodeGroup' in jsonified_request
    assert jsonified_request['nodeGroup'] == 'node_group_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.patch._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('nodeGroup', 'nodeGroupResource', 'project', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_patch_unary_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_patch') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_patch') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.PatchNodeGroupRequest.pb(compute.PatchNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.PatchNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.patch_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_patch_unary_rest_bad_request(transport: str='rest', request_type=compute.PatchNodeGroupRequest):
    if False:
        return 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.patch_unary(request)

def test_patch_unary_rest_flattened():
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
        mock_args = dict(project='project_value', zone='zone_value', node_group='node_group_value', node_group_resource=compute.NodeGroup(autoscaling_policy=compute.NodeGroupAutoscalingPolicy(max_nodes=958)))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups/{node_group}' % client.transport._host, args[1])

def test_patch_unary_rest_flattened_error(transport: str='rest'):
    if False:
        return 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.patch_unary(compute.PatchNodeGroupRequest(), project='project_value', zone='zone_value', node_group='node_group_value', node_group_resource=compute.NodeGroup(autoscaling_policy=compute.NodeGroupAutoscalingPolicy(max_nodes=958)))

def test_patch_unary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetIamPolicyNodeGroupRequest, dict])
def test_set_iam_policy_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2', 'resource': 'sample3'}
    request_init['zone_set_policy_request_resource'] = {'bindings': [{'binding_id': 'binding_id_value', 'condition': {'description': 'description_value', 'expression': 'expression_value', 'location': 'location_value', 'title': 'title_value'}, 'members': ['members_value1', 'members_value2'], 'role': 'role_value'}], 'etag': 'etag_value', 'policy': {'audit_configs': [{'audit_log_configs': [{'exempted_members': ['exempted_members_value1', 'exempted_members_value2'], 'ignore_child_exemptions': True, 'log_type': 'log_type_value'}], 'exempted_members': ['exempted_members_value1', 'exempted_members_value2'], 'service': 'service_value'}], 'bindings': {}, 'etag': 'etag_value', 'iam_owned': True, 'rules': [{'action': 'action_value', 'conditions': [{'iam': 'iam_value', 'op': 'op_value', 'svc': 'svc_value', 'sys': 'sys_value', 'values': ['values_value1', 'values_value2']}], 'description': 'description_value', 'ins': ['ins_value1', 'ins_value2'], 'log_configs': [{'cloud_audit': {'authorization_logging_options': {'permission_type': 'permission_type_value'}, 'log_name': 'log_name_value'}, 'counter': {'custom_fields': [{'name': 'name_value', 'value': 'value_value'}], 'field': 'field_value', 'metric': 'metric_value'}, 'data_access': {'log_mode': 'log_mode_value'}}], 'not_ins': ['not_ins_value1', 'not_ins_value2'], 'permissions': ['permissions_value1', 'permissions_value2']}], 'version': 774}}
    test_field = compute.SetIamPolicyNodeGroupRequest.meta.fields['zone_set_policy_request_resource']

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
    for (field, value) in request_init['zone_set_policy_request_resource'].items():
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
                for i in range(0, len(request_init['zone_set_policy_request_resource'][field])):
                    del request_init['zone_set_policy_request_resource'][field][i][subfield]
            else:
                del request_init['zone_set_policy_request_resource'][field][subfield]
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

def test_set_iam_policy_rest_required_fields(request_type=compute.SetIamPolicyNodeGroupRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['resource'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['resource'] = 'resource_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_iam_policy._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_iam_policy._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('project', 'resource', 'zone', 'zoneSetPolicyRequestResource'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_iam_policy_rest_interceptors(null_interceptor):
    if False:
        while True:
            i = 10
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_set_iam_policy') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_set_iam_policy') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetIamPolicyNodeGroupRequest.pb(compute.SetIamPolicyNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Policy.to_json(compute.Policy())
        request = compute.SetIamPolicyNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Policy()
        client.set_iam_policy(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_iam_policy_rest_bad_request(transport: str='rest', request_type=compute.SetIamPolicyNodeGroupRequest):
    if False:
        print('Hello World!')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2', 'resource': 'sample3'}
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
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Policy()
        sample_request = {'project': 'sample1', 'zone': 'sample2', 'resource': 'sample3'}
        mock_args = dict(project='project_value', zone='zone_value', resource='resource_value', zone_set_policy_request_resource=compute.ZoneSetPolicyRequest(bindings=[compute.Binding(binding_id='binding_id_value')]))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups/{resource}/setIamPolicy' % client.transport._host, args[1])

def test_set_iam_policy_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_iam_policy(compute.SetIamPolicyNodeGroupRequest(), project='project_value', zone='zone_value', resource='resource_value', zone_set_policy_request_resource=compute.ZoneSetPolicyRequest(bindings=[compute.Binding(binding_id='binding_id_value')]))

def test_set_iam_policy_rest_error():
    if False:
        print('Hello World!')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetNodeTemplateNodeGroupRequest, dict])
def test_set_node_template_rest(request_type):
    if False:
        return 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request_init['node_groups_set_node_template_request_resource'] = {'node_template': 'node_template_value'}
    test_field = compute.SetNodeTemplateNodeGroupRequest.meta.fields['node_groups_set_node_template_request_resource']

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
    for (field, value) in request_init['node_groups_set_node_template_request_resource'].items():
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
                for i in range(0, len(request_init['node_groups_set_node_template_request_resource'][field])):
                    del request_init['node_groups_set_node_template_request_resource'][field][i][subfield]
            else:
                del request_init['node_groups_set_node_template_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_node_template(request)
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

def test_set_node_template_rest_required_fields(request_type=compute.SetNodeTemplateNodeGroupRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['node_group'] = ''
    request_init['project'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_node_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['nodeGroup'] = 'node_group_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_node_template._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'nodeGroup' in jsonified_request
    assert jsonified_request['nodeGroup'] == 'node_group_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.set_node_template(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_node_template_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_node_template._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('nodeGroup', 'nodeGroupsSetNodeTemplateRequestResource', 'project', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_node_template_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_set_node_template') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_set_node_template') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetNodeTemplateNodeGroupRequest.pb(compute.SetNodeTemplateNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SetNodeTemplateNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.set_node_template(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_node_template_rest_bad_request(transport: str='rest', request_type=compute.SetNodeTemplateNodeGroupRequest):
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_node_template(request)

def test_set_node_template_rest_flattened():
    if False:
        print('Hello World!')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
        mock_args = dict(project='project_value', zone='zone_value', node_group='node_group_value', node_groups_set_node_template_request_resource=compute.NodeGroupsSetNodeTemplateRequest(node_template='node_template_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_node_template(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups/{node_group}/setNodeTemplate' % client.transport._host, args[1])

def test_set_node_template_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_node_template(compute.SetNodeTemplateNodeGroupRequest(), project='project_value', zone='zone_value', node_group='node_group_value', node_groups_set_node_template_request_resource=compute.NodeGroupsSetNodeTemplateRequest(node_template='node_template_value'))

def test_set_node_template_rest_error():
    if False:
        i = 10
        return i + 15
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SetNodeTemplateNodeGroupRequest, dict])
def test_set_node_template_unary_rest(request_type):
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request_init['node_groups_set_node_template_request_resource'] = {'node_template': 'node_template_value'}
    test_field = compute.SetNodeTemplateNodeGroupRequest.meta.fields['node_groups_set_node_template_request_resource']

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
    for (field, value) in request_init['node_groups_set_node_template_request_resource'].items():
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
                for i in range(0, len(request_init['node_groups_set_node_template_request_resource'][field])):
                    del request_init['node_groups_set_node_template_request_resource'][field][i][subfield]
            else:
                del request_init['node_groups_set_node_template_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.set_node_template_unary(request)
    assert isinstance(response, compute.Operation)

def test_set_node_template_unary_rest_required_fields(request_type=compute.SetNodeTemplateNodeGroupRequest):
    if False:
        return 10
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['node_group'] = ''
    request_init['project'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_node_template._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['nodeGroup'] = 'node_group_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).set_node_template._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'nodeGroup' in jsonified_request
    assert jsonified_request['nodeGroup'] == 'node_group_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.set_node_template_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_set_node_template_unary_rest_unset_required_fields():
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.set_node_template._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('nodeGroup', 'nodeGroupsSetNodeTemplateRequestResource', 'project', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_set_node_template_unary_rest_interceptors(null_interceptor):
    if False:
        return 10
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_set_node_template') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_set_node_template') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SetNodeTemplateNodeGroupRequest.pb(compute.SetNodeTemplateNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SetNodeTemplateNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.set_node_template_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_set_node_template_unary_rest_bad_request(transport: str='rest', request_type=compute.SetNodeTemplateNodeGroupRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.set_node_template_unary(request)

def test_set_node_template_unary_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
        mock_args = dict(project='project_value', zone='zone_value', node_group='node_group_value', node_groups_set_node_template_request_resource=compute.NodeGroupsSetNodeTemplateRequest(node_template='node_template_value'))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.set_node_template_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups/{node_group}/setNodeTemplate' % client.transport._host, args[1])

def test_set_node_template_unary_rest_flattened_error(transport: str='rest'):
    if False:
        print('Hello World!')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.set_node_template_unary(compute.SetNodeTemplateNodeGroupRequest(), project='project_value', zone='zone_value', node_group='node_group_value', node_groups_set_node_template_request_resource=compute.NodeGroupsSetNodeTemplateRequest(node_template='node_template_value'))

def test_set_node_template_unary_rest_error():
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SimulateMaintenanceEventNodeGroupRequest, dict])
def test_simulate_maintenance_event_rest(request_type):
    if False:
        print('Hello World!')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request_init['node_groups_simulate_maintenance_event_request_resource'] = {'nodes': ['nodes_value1', 'nodes_value2']}
    test_field = compute.SimulateMaintenanceEventNodeGroupRequest.meta.fields['node_groups_simulate_maintenance_event_request_resource']

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
    for (field, value) in request_init['node_groups_simulate_maintenance_event_request_resource'].items():
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
                for i in range(0, len(request_init['node_groups_simulate_maintenance_event_request_resource'][field])):
                    del request_init['node_groups_simulate_maintenance_event_request_resource'][field][i][subfield]
            else:
                del request_init['node_groups_simulate_maintenance_event_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.simulate_maintenance_event(request)
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

def test_simulate_maintenance_event_rest_required_fields(request_type=compute.SimulateMaintenanceEventNodeGroupRequest):
    if False:
        for i in range(10):
            print('nop')
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['node_group'] = ''
    request_init['project'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).simulate_maintenance_event._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['nodeGroup'] = 'node_group_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).simulate_maintenance_event._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'nodeGroup' in jsonified_request
    assert jsonified_request['nodeGroup'] == 'node_group_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.simulate_maintenance_event(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_simulate_maintenance_event_rest_unset_required_fields():
    if False:
        print('Hello World!')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.simulate_maintenance_event._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('nodeGroup', 'nodeGroupsSimulateMaintenanceEventRequestResource', 'project', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_simulate_maintenance_event_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_simulate_maintenance_event') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_simulate_maintenance_event') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SimulateMaintenanceEventNodeGroupRequest.pb(compute.SimulateMaintenanceEventNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SimulateMaintenanceEventNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.simulate_maintenance_event(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_simulate_maintenance_event_rest_bad_request(transport: str='rest', request_type=compute.SimulateMaintenanceEventNodeGroupRequest):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.simulate_maintenance_event(request)

def test_simulate_maintenance_event_rest_flattened():
    if False:
        print('Hello World!')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
        mock_args = dict(project='project_value', zone='zone_value', node_group='node_group_value', node_groups_simulate_maintenance_event_request_resource=compute.NodeGroupsSimulateMaintenanceEventRequest(nodes=['nodes_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.simulate_maintenance_event(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups/{node_group}/simulateMaintenanceEvent' % client.transport._host, args[1])

def test_simulate_maintenance_event_rest_flattened_error(transport: str='rest'):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.simulate_maintenance_event(compute.SimulateMaintenanceEventNodeGroupRequest(), project='project_value', zone='zone_value', node_group='node_group_value', node_groups_simulate_maintenance_event_request_resource=compute.NodeGroupsSimulateMaintenanceEventRequest(nodes=['nodes_value']))

def test_simulate_maintenance_event_rest_error():
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.SimulateMaintenanceEventNodeGroupRequest, dict])
def test_simulate_maintenance_event_unary_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request_init['node_groups_simulate_maintenance_event_request_resource'] = {'nodes': ['nodes_value1', 'nodes_value2']}
    test_field = compute.SimulateMaintenanceEventNodeGroupRequest.meta.fields['node_groups_simulate_maintenance_event_request_resource']

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
    for (field, value) in request_init['node_groups_simulate_maintenance_event_request_resource'].items():
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
                for i in range(0, len(request_init['node_groups_simulate_maintenance_event_request_resource'][field])):
                    del request_init['node_groups_simulate_maintenance_event_request_resource'][field][i][subfield]
            else:
                del request_init['node_groups_simulate_maintenance_event_request_resource'][field][subfield]
    request = request_type(**request_init)
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation(client_operation_id='client_operation_id_value', creation_timestamp='creation_timestamp_value', description='description_value', end_time='end_time_value', http_error_message='http_error_message_value', http_error_status_code=2374, id=205, insert_time='insert_time_value', kind='kind_value', name='name_value', operation_group_id='operation_group_id_value', operation_type='operation_type_value', progress=885, region='region_value', self_link='self_link_value', start_time='start_time_value', status=compute.Operation.Status.DONE, status_message='status_message_value', target_id=947, target_link='target_link_value', user='user_value', zone='zone_value')
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        response = client.simulate_maintenance_event_unary(request)
    assert isinstance(response, compute.Operation)

def test_simulate_maintenance_event_unary_rest_required_fields(request_type=compute.SimulateMaintenanceEventNodeGroupRequest):
    if False:
        i = 10
        return i + 15
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['node_group'] = ''
    request_init['project'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).simulate_maintenance_event._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['nodeGroup'] = 'node_group_value'
    jsonified_request['project'] = 'project_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).simulate_maintenance_event._get_unset_required_fields(jsonified_request)
    assert not set(unset_fields) - set(('request_id',))
    jsonified_request.update(unset_fields)
    assert 'nodeGroup' in jsonified_request
    assert jsonified_request['nodeGroup'] == 'node_group_value'
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
            response = client.simulate_maintenance_event_unary(request)
            expected_params = []
            actual_params = req.call_args.kwargs['params']
            assert expected_params == actual_params

def test_simulate_maintenance_event_unary_rest_unset_required_fields():
    if False:
        while True:
            i = 10
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.simulate_maintenance_event._get_unset_required_fields({})
    assert set(unset_fields) == set(('requestId',)) & set(('nodeGroup', 'nodeGroupsSimulateMaintenanceEventRequestResource', 'project', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_simulate_maintenance_event_unary_rest_interceptors(null_interceptor):
    if False:
        for i in range(10):
            print('nop')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_simulate_maintenance_event') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_simulate_maintenance_event') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.SimulateMaintenanceEventNodeGroupRequest.pb(compute.SimulateMaintenanceEventNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.Operation.to_json(compute.Operation())
        request = compute.SimulateMaintenanceEventNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.Operation()
        client.simulate_maintenance_event_unary(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_simulate_maintenance_event_unary_rest_bad_request(transport: str='rest', request_type=compute.SimulateMaintenanceEventNodeGroupRequest):
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.simulate_maintenance_event_unary(request)

def test_simulate_maintenance_event_unary_rest_flattened():
    if False:
        print('Hello World!')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.Operation()
        sample_request = {'project': 'sample1', 'zone': 'sample2', 'node_group': 'sample3'}
        mock_args = dict(project='project_value', zone='zone_value', node_group='node_group_value', node_groups_simulate_maintenance_event_request_resource=compute.NodeGroupsSimulateMaintenanceEventRequest(nodes=['nodes_value']))
        mock_args.update(sample_request)
        response_value = Response()
        response_value.status_code = 200
        return_value = compute.Operation.pb(return_value)
        json_return_value = json_format.MessageToJson(return_value)
        response_value._content = json_return_value.encode('UTF-8')
        req.return_value = response_value
        client.simulate_maintenance_event_unary(**mock_args)
        assert len(req.mock_calls) == 1
        (_, args, _) = req.mock_calls[0]
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups/{node_group}/simulateMaintenanceEvent' % client.transport._host, args[1])

def test_simulate_maintenance_event_unary_rest_flattened_error(transport: str='rest'):
    if False:
        i = 10
        return i + 15
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.simulate_maintenance_event_unary(compute.SimulateMaintenanceEventNodeGroupRequest(), project='project_value', zone='zone_value', node_group='node_group_value', node_groups_simulate_maintenance_event_request_resource=compute.NodeGroupsSimulateMaintenanceEventRequest(nodes=['nodes_value']))

def test_simulate_maintenance_event_unary_rest_error():
    if False:
        return 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

@pytest.mark.parametrize('request_type', [compute.TestIamPermissionsNodeGroupRequest, dict])
def test_test_iam_permissions_rest(request_type):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    request_init = {'project': 'sample1', 'zone': 'sample2', 'resource': 'sample3'}
    request_init['test_permissions_request_resource'] = {'permissions': ['permissions_value1', 'permissions_value2']}
    test_field = compute.TestIamPermissionsNodeGroupRequest.meta.fields['test_permissions_request_resource']

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

def test_test_iam_permissions_rest_required_fields(request_type=compute.TestIamPermissionsNodeGroupRequest):
    if False:
        print('Hello World!')
    transport_class = transports.NodeGroupsRestTransport
    request_init = {}
    request_init['project'] = ''
    request_init['resource'] = ''
    request_init['zone'] = ''
    request = request_type(**request_init)
    pb_request = request_type.pb(request)
    jsonified_request = json.loads(json_format.MessageToJson(pb_request, including_default_value_fields=False, use_integers_for_enums=False))
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    jsonified_request['project'] = 'project_value'
    jsonified_request['resource'] = 'resource_value'
    jsonified_request['zone'] = 'zone_value'
    unset_fields = transport_class(credentials=ga_credentials.AnonymousCredentials()).test_iam_permissions._get_unset_required_fields(jsonified_request)
    jsonified_request.update(unset_fields)
    assert 'project' in jsonified_request
    assert jsonified_request['project'] == 'project_value'
    assert 'resource' in jsonified_request
    assert jsonified_request['resource'] == 'resource_value'
    assert 'zone' in jsonified_request
    assert jsonified_request['zone'] == 'zone_value'
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
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
        for i in range(10):
            print('nop')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials)
    unset_fields = transport.test_iam_permissions._get_unset_required_fields({})
    assert set(unset_fields) == set(()) & set(('project', 'resource', 'testPermissionsRequestResource', 'zone'))

@pytest.mark.parametrize('null_interceptor', [True, False])
def test_test_iam_permissions_rest_interceptors(null_interceptor):
    if False:
        print('Hello World!')
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials(), interceptor=None if null_interceptor else transports.NodeGroupsRestInterceptor())
    client = NodeGroupsClient(transport=transport)
    with mock.patch.object(type(client.transport._session), 'request') as req, mock.patch.object(path_template, 'transcode') as transcode, mock.patch.object(transports.NodeGroupsRestInterceptor, 'post_test_iam_permissions') as post, mock.patch.object(transports.NodeGroupsRestInterceptor, 'pre_test_iam_permissions') as pre:
        pre.assert_not_called()
        post.assert_not_called()
        pb_message = compute.TestIamPermissionsNodeGroupRequest.pb(compute.TestIamPermissionsNodeGroupRequest())
        transcode.return_value = {'method': 'post', 'uri': 'my_uri', 'body': pb_message, 'query_params': pb_message}
        req.return_value = Response()
        req.return_value.status_code = 200
        req.return_value.request = PreparedRequest()
        req.return_value._content = compute.TestPermissionsResponse.to_json(compute.TestPermissionsResponse())
        request = compute.TestIamPermissionsNodeGroupRequest()
        metadata = [('key', 'val'), ('cephalopod', 'squid')]
        pre.return_value = (request, metadata)
        post.return_value = compute.TestPermissionsResponse()
        client.test_iam_permissions(request, metadata=[('key', 'val'), ('cephalopod', 'squid')])
        pre.assert_called_once()
        post.assert_called_once()

def test_test_iam_permissions_rest_bad_request(transport: str='rest', request_type=compute.TestIamPermissionsNodeGroupRequest):
    if False:
        i = 10
        return i + 15
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    request_init = {'project': 'sample1', 'zone': 'sample2', 'resource': 'sample3'}
    request = request_type(**request_init)
    with mock.patch.object(Session, 'request') as req, pytest.raises(core_exceptions.BadRequest):
        response_value = Response()
        response_value.status_code = 400
        response_value.request = Request()
        req.return_value = response_value
        client.test_iam_permissions(request)

def test_test_iam_permissions_rest_flattened():
    if False:
        i = 10
        return i + 15
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')
    with mock.patch.object(type(client.transport._session), 'request') as req:
        return_value = compute.TestPermissionsResponse()
        sample_request = {'project': 'sample1', 'zone': 'sample2', 'resource': 'sample3'}
        mock_args = dict(project='project_value', zone='zone_value', resource='resource_value', test_permissions_request_resource=compute.TestPermissionsRequest(permissions=['permissions_value']))
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
        assert path_template.validate('%s/compute/v1/projects/{project}/zones/{zone}/nodeGroups/{resource}/testIamPermissions' % client.transport._host, args[1])

def test_test_iam_permissions_rest_flattened_error(transport: str='rest'):
    if False:
        while True:
            i = 10
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    with pytest.raises(ValueError):
        client.test_iam_permissions(compute.TestIamPermissionsNodeGroupRequest(), project='project_value', zone='zone_value', resource='resource_value', test_permissions_request_resource=compute.TestPermissionsRequest(permissions=['permissions_value']))

def test_test_iam_permissions_rest_error():
    if False:
        i = 10
        return i + 15
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport='rest')

def test_credentials_transport_error():
    if False:
        while True:
            i = 10
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = NodeGroupsClient(client_options={'credentials_file': 'credentials.json'}, transport=transport)
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials())
    options = client_options.ClientOptions()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = NodeGroupsClient(client_options=options, transport=transport)
    options = mock.Mock()
    options.api_key = 'api_key'
    with pytest.raises(ValueError):
        client = NodeGroupsClient(client_options=options, credentials=ga_credentials.AnonymousCredentials())
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials())
    with pytest.raises(ValueError):
        client = NodeGroupsClient(client_options={'scopes': ['1', '2']}, transport=transport)

def test_transport_instance():
    if False:
        i = 10
        return i + 15
    transport = transports.NodeGroupsRestTransport(credentials=ga_credentials.AnonymousCredentials())
    client = NodeGroupsClient(transport=transport)
    assert client.transport is transport

@pytest.mark.parametrize('transport_class', [transports.NodeGroupsRestTransport])
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
    transport = NodeGroupsClient.get_transport_class(transport_name)(credentials=ga_credentials.AnonymousCredentials())
    assert transport.kind == transport_name

def test_node_groups_base_transport_error():
    if False:
        while True:
            i = 10
    with pytest.raises(core_exceptions.DuplicateCredentialArgs):
        transport = transports.NodeGroupsTransport(credentials=ga_credentials.AnonymousCredentials(), credentials_file='credentials.json')

def test_node_groups_base_transport():
    if False:
        while True:
            i = 10
    with mock.patch('google.cloud.compute_v1.services.node_groups.transports.NodeGroupsTransport.__init__') as Transport:
        Transport.return_value = None
        transport = transports.NodeGroupsTransport(credentials=ga_credentials.AnonymousCredentials())
    methods = ('add_nodes', 'aggregated_list', 'delete', 'delete_nodes', 'get', 'get_iam_policy', 'insert', 'list', 'list_nodes', 'patch', 'set_iam_policy', 'set_node_template', 'simulate_maintenance_event', 'test_iam_permissions')
    for method in methods:
        with pytest.raises(NotImplementedError):
            getattr(transport, method)(request=object())
    with pytest.raises(NotImplementedError):
        transport.close()
    remainder = ['kind']
    for r in remainder:
        with pytest.raises(NotImplementedError):
            getattr(transport, r)()

def test_node_groups_base_transport_with_credentials_file():
    if False:
        print('Hello World!')
    with mock.patch.object(google.auth, 'load_credentials_from_file', autospec=True) as load_creds, mock.patch('google.cloud.compute_v1.services.node_groups.transports.NodeGroupsTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        load_creds.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.NodeGroupsTransport(credentials_file='credentials.json', quota_project_id='octopus')
        load_creds.assert_called_once_with('credentials.json', scopes=None, default_scopes=('https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id='octopus')

def test_node_groups_base_transport_with_adc():
    if False:
        while True:
            i = 10
    with mock.patch.object(google.auth, 'default', autospec=True) as adc, mock.patch('google.cloud.compute_v1.services.node_groups.transports.NodeGroupsTransport._prep_wrapped_messages') as Transport:
        Transport.return_value = None
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        transport = transports.NodeGroupsTransport()
        adc.assert_called_once()

def test_node_groups_auth_adc():
    if False:
        i = 10
        return i + 15
    with mock.patch.object(google.auth, 'default', autospec=True) as adc:
        adc.return_value = (ga_credentials.AnonymousCredentials(), None)
        NodeGroupsClient()
        adc.assert_called_once_with(scopes=None, default_scopes=('https://www.googleapis.com/auth/compute', 'https://www.googleapis.com/auth/cloud-platform'), quota_project_id=None)

def test_node_groups_http_transport_client_cert_source_for_mtls():
    if False:
        print('Hello World!')
    cred = ga_credentials.AnonymousCredentials()
    with mock.patch('google.auth.transport.requests.AuthorizedSession.configure_mtls_channel') as mock_configure_mtls_channel:
        transports.NodeGroupsRestTransport(credentials=cred, client_cert_source_for_mtls=client_cert_source_callback)
        mock_configure_mtls_channel.assert_called_once_with(client_cert_source_callback)

@pytest.mark.parametrize('transport_name', ['rest'])
def test_node_groups_host_no_port(transport_name):
    if False:
        for i in range(10):
            print('nop')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='compute.googleapis.com'), transport=transport_name)
    assert client.transport._host == ('compute.googleapis.com:443' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_node_groups_host_with_port(transport_name):
    if False:
        print('Hello World!')
    client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), client_options=client_options.ClientOptions(api_endpoint='compute.googleapis.com:8000'), transport=transport_name)
    assert client.transport._host == ('compute.googleapis.com:8000' if transport_name in ['grpc', 'grpc_asyncio'] else 'https://compute.googleapis.com:8000')

@pytest.mark.parametrize('transport_name', ['rest'])
def test_node_groups_client_transport_session_collision(transport_name):
    if False:
        for i in range(10):
            print('nop')
    creds1 = ga_credentials.AnonymousCredentials()
    creds2 = ga_credentials.AnonymousCredentials()
    client1 = NodeGroupsClient(credentials=creds1, transport=transport_name)
    client2 = NodeGroupsClient(credentials=creds2, transport=transport_name)
    session1 = client1.transport.add_nodes._session
    session2 = client2.transport.add_nodes._session
    assert session1 != session2
    session1 = client1.transport.aggregated_list._session
    session2 = client2.transport.aggregated_list._session
    assert session1 != session2
    session1 = client1.transport.delete._session
    session2 = client2.transport.delete._session
    assert session1 != session2
    session1 = client1.transport.delete_nodes._session
    session2 = client2.transport.delete_nodes._session
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
    session1 = client1.transport.list_nodes._session
    session2 = client2.transport.list_nodes._session
    assert session1 != session2
    session1 = client1.transport.patch._session
    session2 = client2.transport.patch._session
    assert session1 != session2
    session1 = client1.transport.set_iam_policy._session
    session2 = client2.transport.set_iam_policy._session
    assert session1 != session2
    session1 = client1.transport.set_node_template._session
    session2 = client2.transport.set_node_template._session
    assert session1 != session2
    session1 = client1.transport.simulate_maintenance_event._session
    session2 = client2.transport.simulate_maintenance_event._session
    assert session1 != session2
    session1 = client1.transport.test_iam_permissions._session
    session2 = client2.transport.test_iam_permissions._session
    assert session1 != session2

def test_common_billing_account_path():
    if False:
        for i in range(10):
            print('nop')
    billing_account = 'squid'
    expected = 'billingAccounts/{billing_account}'.format(billing_account=billing_account)
    actual = NodeGroupsClient.common_billing_account_path(billing_account)
    assert expected == actual

def test_parse_common_billing_account_path():
    if False:
        while True:
            i = 10
    expected = {'billing_account': 'clam'}
    path = NodeGroupsClient.common_billing_account_path(**expected)
    actual = NodeGroupsClient.parse_common_billing_account_path(path)
    assert expected == actual

def test_common_folder_path():
    if False:
        while True:
            i = 10
    folder = 'whelk'
    expected = 'folders/{folder}'.format(folder=folder)
    actual = NodeGroupsClient.common_folder_path(folder)
    assert expected == actual

def test_parse_common_folder_path():
    if False:
        return 10
    expected = {'folder': 'octopus'}
    path = NodeGroupsClient.common_folder_path(**expected)
    actual = NodeGroupsClient.parse_common_folder_path(path)
    assert expected == actual

def test_common_organization_path():
    if False:
        return 10
    organization = 'oyster'
    expected = 'organizations/{organization}'.format(organization=organization)
    actual = NodeGroupsClient.common_organization_path(organization)
    assert expected == actual

def test_parse_common_organization_path():
    if False:
        i = 10
        return i + 15
    expected = {'organization': 'nudibranch'}
    path = NodeGroupsClient.common_organization_path(**expected)
    actual = NodeGroupsClient.parse_common_organization_path(path)
    assert expected == actual

def test_common_project_path():
    if False:
        print('Hello World!')
    project = 'cuttlefish'
    expected = 'projects/{project}'.format(project=project)
    actual = NodeGroupsClient.common_project_path(project)
    assert expected == actual

def test_parse_common_project_path():
    if False:
        for i in range(10):
            print('nop')
    expected = {'project': 'mussel'}
    path = NodeGroupsClient.common_project_path(**expected)
    actual = NodeGroupsClient.parse_common_project_path(path)
    assert expected == actual

def test_common_location_path():
    if False:
        return 10
    project = 'winkle'
    location = 'nautilus'
    expected = 'projects/{project}/locations/{location}'.format(project=project, location=location)
    actual = NodeGroupsClient.common_location_path(project, location)
    assert expected == actual

def test_parse_common_location_path():
    if False:
        while True:
            i = 10
    expected = {'project': 'scallop', 'location': 'abalone'}
    path = NodeGroupsClient.common_location_path(**expected)
    actual = NodeGroupsClient.parse_common_location_path(path)
    assert expected == actual

def test_client_with_default_client_info():
    if False:
        while True:
            i = 10
    client_info = gapic_v1.client_info.ClientInfo()
    with mock.patch.object(transports.NodeGroupsTransport, '_prep_wrapped_messages') as prep:
        client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)
    with mock.patch.object(transports.NodeGroupsTransport, '_prep_wrapped_messages') as prep:
        transport_class = NodeGroupsClient.get_transport_class()
        transport = transport_class(credentials=ga_credentials.AnonymousCredentials(), client_info=client_info)
        prep.assert_called_once_with(client_info)

def test_transport_close():
    if False:
        for i in range(10):
            print('nop')
    transports = {'rest': '_session'}
    for (transport, close_name) in transports.items():
        client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
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
        client = NodeGroupsClient(credentials=ga_credentials.AnonymousCredentials(), transport=transport)
        with mock.patch.object(type(client.transport), 'close') as close:
            close.assert_not_called()
            with client:
                pass
            close.assert_called()

@pytest.mark.parametrize('client_class,transport_class', [(NodeGroupsClient, transports.NodeGroupsRestTransport)])
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